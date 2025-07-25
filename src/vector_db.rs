use anyhow::{Result, anyhow};
use hnsw::Searcher;
use ordered_float::NotNan;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::metrics::Index;
use crate::params::Params;
use crate::storage::{Header, Storage};
use crate::storage::{StoredEntry, VERSION};
use crate::types::{Metadata, Metric, SearchResult};

#[derive(Clone)]
struct Entry {
    id: usize,
    metadata: Metadata,
    deleted: bool,
}

pub struct VectorDB<const M: usize = 12, const M0: usize = 24> {
    storage: Storage,
    path: PathBuf,
    metric: Metric,
    dim: usize,
    index: Index<M, M0>,
    searcher: Searcher<u32>,
    entries: Vec<Entry>,
    ids: HashSet<usize>,
    params: Params,
}

impl<const M: usize, const M0: usize> VectorDB<M, M0> {
    pub fn open<P: AsRef<Path>>(path: P, metric: Metric) -> Result<Self> {
        Self::open_with_params(path, metric, Params::default())
    }

    pub fn open_with_params<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        params: Params,
    ) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        if path.as_ref().exists() {
            let (storage, header, stored_entries) = Storage::open(&path_buf)?;
            if header.metric != metric {
                return Err(anyhow!("Metric mismatch"));
            }
            let mut db = Self::new_empty(storage, path_buf, metric, header.dim as usize, params);
            for e in stored_entries {
                db.apply_entry(e)?;
            }
            Ok(db)
        } else {
            let storage = Storage::create(&path_buf, metric)?;
            let db = Self::new_empty(storage, path_buf, metric, 0, params);
            Ok(db)
        }
    }

    fn new_empty(
        storage: Storage,
        path: PathBuf,
        metric: Metric,
        dim: usize,
        params: Params,
    ) -> Self {
        Self {
            storage,
            path,
            metric,
            dim,
            index: Index::new_params(metric, params.ef_construction),
            searcher: Searcher::default(),
            entries: Vec::new(),
            ids: HashSet::new(),
            params,
        }
    }

    fn apply_entry(&mut self, entry: StoredEntry) -> Result<()> {
        if entry.deleted {
            if let Some(pos) = self.entries.iter_mut().position(|e| e.id == entry.id && !e.deleted) {
                self.entries[pos].deleted = true;
                self.ids.remove(&entry.id);
            }
            return Ok(());
        }

        if self.ids.contains(&entry.id) {
            // previous value exists, mark deleted
            if let Some(pos) = self.entries.iter_mut().position(|e| e.id == entry.id && !e.deleted) {
                self.entries[pos].deleted = true;
                self.ids.remove(&entry.id);
            }
        }

        if self.dim == 0 {
            self.dim = entry.vector.len();
        } else if entry.vector.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        self.index.insert(entry.vector, &mut self.searcher);
        self.entries.push(Entry { id: entry.id, metadata: entry.metadata, deleted: false });
        self.ids.insert(entry.id);
        Ok(())
    }

    pub fn add(&mut self, id: usize, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        if self.ids.contains(&id) {
            return Err(anyhow!("duplicate id"));
        }
        if self.dim == 0 {
            self.dim = vector.len();
            let header = Header {
                magic: crate::storage::MAGIC,
                version: VERSION,
                metric: self.metric,
                dim: self.dim as u32,
            };
            self.storage.update_header(&header)?;
        } else if vector.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        let stored = StoredEntry {
            id,
            vector: vector.clone(),
            metadata: metadata.clone(),
            deleted: false,
        };
        self.index.insert(vector, &mut self.searcher);
        self.storage.append_entry(&stored)?;
        self.entries.push(Entry { id, metadata, deleted: false });
        self.ids.insert(id);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        let valid = self.entries.iter().filter(|e| !e.deleted).count();
        let real_k = k.min(valid);
        let mut neighbors = vec![
            space::Neighbor {
                index: !0,
                distance: 0
            };
            self.entries.len()
        ];
        let mut searcher = Searcher::default();
        let q = query.to_vec();
        let found = self.index.nearest(
            &q,
            self.params.ef_search.max(real_k * 2),
            &mut searcher,
            &mut neighbors,
        );
        let mut results: Vec<SearchResult> = found
            .iter()
            .filter_map(|n| {
                let entry = &self.entries[n.index];
                if entry.deleted {
                    None
                } else {
                    Some(SearchResult {
                        id: entry.id,
                        distance: f32::from_bits(n.distance),
                        metadata: entry.metadata.clone(),
                    })
                }
            })
            .collect();
        results.sort_by_key(|r| NotNan::new(r.distance).unwrap());
        results.truncate(real_k);
        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }


    pub fn remove(&mut self, id: usize) -> Result<()> {
        let pos = self.entries.iter_mut().position(|e| e.id == id && !e.deleted)
            .ok_or(anyhow!("not found"))?;
        self.entries[pos].deleted = true;
        self.ids.remove(&id);
        let tomb = StoredEntry { id, vector: Vec::new(), metadata: Metadata::default(), deleted: true };
        self.storage.append_entry(&tomb)?;
        Ok(())
    }

    pub fn update(&mut self, id: usize, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        if vector.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        let pos = self.entries.iter_mut().position(|e| e.id == id && !e.deleted)
            .ok_or(anyhow!("not found"))?;
        self.entries[pos].deleted = true;
        self.ids.remove(&id);
        let tomb = StoredEntry { id, vector: Vec::new(), metadata: Metadata::default(), deleted: true };
        self.storage.append_entry(&tomb)?;
        self.add(id, vector, metadata)
    }

    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        queries.par_iter().map(|q| self.search(q, k)).collect()
    }
}
