use anyhow::{Result, anyhow};
use hnsw::Searcher;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::metrics::Index;
use crate::storage::{Header, Storage};
use crate::storage::{StoredEntry, VERSION};
use crate::types::{Metadata, Metric, SearchResult};

pub struct VectorDB {
    storage: Storage,
    path: PathBuf,
    metric: Metric,
    dim: usize,
    index: Index,
    searcher: Searcher<u32>,
    entries: Vec<(usize, Metadata)>,
    ids: HashSet<usize>,
}

impl VectorDB {
    pub fn open<P: AsRef<Path>>(path: P, metric: Metric) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        if path.as_ref().exists() {
            let (storage, header, stored_entries) = Storage::open(&path_buf)?;
            if header.metric != metric {
                return Err(anyhow!("Metric mismatch"));
            }
            let mut db = Self::new_empty(storage, path_buf, metric, header.dim as usize);
            for e in stored_entries {
                db.add_loaded(e.id, e.vector, e.metadata)?;
            }
            Ok(db)
        } else {
            let storage = Storage::create(&path_buf, metric)?;
            let db = Self::new_empty(storage, path_buf, metric, 0);
            Ok(db)
        }
    }

    fn new_empty(storage: Storage, path: PathBuf, metric: Metric, dim: usize) -> Self {
        Self {
            storage,
            path,
            metric,
            dim,
            index: Index::new(metric),
            searcher: Searcher::default(),
            entries: Vec::new(),
            ids: HashSet::new(),
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

    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(anyhow!("dimension mismatch"));
        }
        let mut neighbors = vec![
            space::Neighbor {
                index: !0,
                distance: 0
            };
            k
        ];
        let found = self
            .index
            .nearest(&query.to_vec(), k * 2, &mut self.searcher, &mut neighbors);
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
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn basic_usage() -> Result<()> {
        let path = "test.vdb";
        let _ = fs::remove_file(path);
        {
            let mut db = VectorDB::open(path, Metric::Cosine)?;
            let vector = vec![0.1, 0.2, 0.3, 0.4];
            let metadata = Metadata {
                label: "sample".into(),
                description: Some("desc".into()),
            };
            db.add(1, vector.clone(), metadata.clone())?;
            let results = db.search(&vector, 1)?;
            assert_eq!(results[0].id, 1);
            assert_eq!(results[0].metadata.label, metadata.label);
        }
        {
            let mut db = VectorDB::open(path, Metric::Cosine)?;
            let query = vec![0.1, 0.2, 0.3, 0.4];
            let results = db.search(&query, 1)?;
            assert_eq!(results[0].id, 1);
        }
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn duplicate_id() -> Result<()> {
        let path = "dup.vdb";
        let _ = fs::remove_file(path);
        let mut db = VectorDB::open(path, Metric::Cosine)?;
        let v = vec![0.0, 0.0, 0.0];
        let m = Metadata {
            label: "a".into(),
            description: None,
        };
        db.add(1, v.clone(), m.clone())?;
        let err = db.add(1, v, m).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn dimension_mismatch() -> Result<()> {
        let path = "dim.vdb";
        let _ = fs::remove_file(path);
        let mut db = VectorDB::open(path, Metric::Cosine)?;
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![0.0, 0.0];
        db.add(
            1,
            v1,
            Metadata {
                label: "a".into(),
                description: None,
            },
        )?;
        let err = db
            .add(
                2,
                v2,
                Metadata {
                    label: "b".into(),
                    description: None,
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("dimension"));
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn metric_mismatch() -> Result<()> {
        let path = "metric.vdb";
        let _ = fs::remove_file(path);
        {
            let mut db = VectorDB::open(path, Metric::Cosine)?;
            db.add(
                1,
                vec![0.0, 0.0, 0.0],
                Metadata {
                    label: "a".into(),
                    description: None,
                },
            )?;
        }
        let err = VectorDB::open(path, Metric::Euclidean);
        assert!(err.is_err());
        fs::remove_file(path)?;
        Ok(())
    }
}
