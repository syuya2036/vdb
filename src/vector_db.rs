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

pub struct VectorDB<const M: usize = 12, const M0: usize = 24> {
    storage: Storage,
    path: PathBuf,
    metric: Metric,
    dim: usize,
    index: Index<M, M0>,
    searcher: Searcher<u32>,
    entries: Vec<(usize, Metadata)>,
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
                db.add_loaded(e.id, e.vector, e.metadata)?;
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

    fn add_loaded(&mut self, id: usize, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        if self.ids.contains(&id) {
            return Err(anyhow!("duplicate id in file"));
        }
        if self.dim == 0 {
            self.dim = vector.len();
        } else if vector.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        self.index.insert(vector, &mut self.searcher);
        self.entries.push((id, metadata));
        self.ids.insert(id);
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
        self.index.insert(vector.clone(), &mut self.searcher);
        self.storage.append_entry(&StoredEntry {
            id,
            vector,
            metadata: metadata.clone(),
        })?;
        self.entries.push((id, metadata));
        self.ids.insert(id);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        let real_k = k.min(self.entries.len());
        let mut neighbors = vec![
            space::Neighbor {
                index: !0,
                distance: 0
            };
            real_k
        ];
        let mut searcher = Searcher::default();
        let found = self.index.nearest(
            &query.to_vec(),
            self.params.ef_search.max(real_k * 2),
            &mut searcher,
            &mut neighbors,
        );
        let mut results: Vec<SearchResult> = found
            .iter()
            .map(|n| {
                let (id, meta) = &self.entries[n.index];
                SearchResult {
                    id: *id,
                    distance: f32::from_bits(n.distance),
                    metadata: meta.clone(),
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

    fn all_entries(&self) -> Vec<StoredEntry> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, (id, meta))| StoredEntry {
                id: *id,
                vector: self.index.feature(i).clone(),
                metadata: meta.clone(),
            })
            .collect()
    }

    fn reload_from_entries(&mut self, entries: Vec<StoredEntry>) -> Result<()> {
        self.index = Index::new_params(self.metric, self.params.ef_construction);
        self.searcher = Searcher::default();
        self.entries.clear();
        self.ids.clear();
        let header = Header {
            magic: crate::storage::MAGIC,
            version: VERSION,
            metric: self.metric,
            dim: self.dim as u32,
        };
        self.storage.rewrite(&header, &entries)?;
        for e in entries {
            self.add_loaded(e.id, e.vector, e.metadata)?;
        }
        Ok(())
    }

    pub fn remove(&mut self, id: usize) -> Result<()> {
        let mut entries = self.all_entries();
        let pos = entries
            .iter()
            .position(|e| e.id == id)
            .ok_or(anyhow!("not found"))?;
        entries.remove(pos);
        if let Some(e) = entries.first() {
            self.dim = e.vector.len();
        } else {
            self.dim = 0;
        }
        self.reload_from_entries(entries)
    }

    pub fn update(&mut self, id: usize, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let mut entries = self.all_entries();
        let pos = entries
            .iter()
            .position(|e| e.id == id)
            .ok_or(anyhow!("not found"))?;
        if vector.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        entries[pos] = StoredEntry {
            id,
            vector,
            metadata,
        };
        self.reload_from_entries(entries)
    }

    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        queries.par_iter().map(|q| self.search(q, k)).collect()
    }
}
