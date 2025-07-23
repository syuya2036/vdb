use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use anyhow::Result;
use hnsw::{Hnsw, Searcher};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use space::{Metric as SpaceMetric, Neighbor};

const M: usize = 12;
const M0: usize = 24;

#[derive(Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub label: String,
    pub description: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Metric {
    Cosine,
    Euclidean,
}

#[derive(Clone, Serialize, Deserialize)]
struct StoredEntry {
    id: usize,
    vector: Vec<f32>,
    metadata: Metadata,
}

#[derive(Serialize, Deserialize)]
struct StoredData {
    metric: Metric,
    entries: Vec<StoredEntry>,
}

#[derive(Clone, Copy)]
struct CosineMetric;

impl SpaceMetric<Vec<f32>> for CosineMetric {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) };
        let dist = 1.0 - cos;
        dist.to_bits()
    }
}

#[derive(Clone, Copy)]
struct EuclideanMetric;

impl SpaceMetric<Vec<f32>> for EuclideanMetric {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt().to_bits()
    }
}

struct Entry {
    id: usize,
    metadata: Metadata,
}

enum Index {
    Cosine(Hnsw<CosineMetric, Vec<f32>, Pcg64, M, M0>),
    Euclidean(Hnsw<EuclideanMetric, Vec<f32>, Pcg64, M, M0>),
}

pub struct VectorDB {
    path: PathBuf,
    metric: Metric,
    index: Index,
    searcher: Searcher<u32>,
    entries: Vec<Entry>,
}

impl VectorDB {
    pub fn open<P: AsRef<Path>>(path: P, metric: Metric) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            let file = File::open(&path)?;
            let reader = BufReader::new(file);
            let stored: StoredData = bincode::deserialize_from(reader)?;
            let mut db = Self::new_empty(path.clone(), stored.metric);
            for entry in stored.entries {
                db.add(entry.id, entry.vector, entry.metadata)?;
            }
            Ok(db)
        } else {
            let db = Self::new_empty(path.clone(), metric);
            db.save()?;
            Ok(db)
        }
    }

    fn new_empty(path: PathBuf, metric: Metric) -> Self {
        let index = match metric {
            Metric::Cosine => Index::Cosine(Hnsw::new(CosineMetric)),
            Metric::Euclidean => Index::Euclidean(Hnsw::new(EuclideanMetric)),
        };
        Self {
            path,
            metric,
            index,
            searcher: Searcher::default(),
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, id: usize, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        match &mut self.index {
            Index::Cosine(h) => { h.insert(vector, &mut self.searcher); }
            Index::Euclidean(h) => { h.insert(vector, &mut self.searcher); }
        }
        self.entries.push(Entry { id, metadata });
        self.save()?;
        Ok(())
    }

    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut neighbors = vec![Neighbor { index: !0, distance: 0 }; k];
        let found = match &self.index {
            Index::Cosine(h) => h.nearest(&query.to_vec(), k * 2, &mut self.searcher, &mut neighbors),
            Index::Euclidean(h) => h.nearest(&query.to_vec(), k * 2, &mut self.searcher, &mut neighbors),
        };
        let mut results: Vec<SearchResult> = found
            .iter()
            .map(|n| {
                let entry = &self.entries[n.index];
                SearchResult { id: entry.id, distance: f32::from_bits(n.distance), metadata: entry.metadata.clone() }
            })
            .collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        Ok(results)
    }

    fn save(&self) -> Result<()> {
        let mut entries = Vec::with_capacity(self.entries.len());
        for i in 0..self.entries.len() {
            let vector = match &self.index {
                Index::Cosine(h) => h.feature(i).clone(),
                Index::Euclidean(h) => h.feature(i).clone(),
            };
            let entry = &self.entries[i];
            entries.push(StoredEntry { id: entry.id, vector, metadata: entry.metadata.clone() });
        }
        let stored = StoredData { metric: self.metric, entries };
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(&self.path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &stored)?;
        Ok(())
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
            let metadata = Metadata { label: "sample".to_string(), description: Some("これはサンプルです".to_string()) };
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
}
