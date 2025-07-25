use crate::types::Metric;
use hnsw::Hnsw;
use rand_pcg::Pcg64;
use space::{Metric as SpaceMetric, Neighbor};

#[derive(Clone, Copy)]
pub struct CosineMetric;

impl SpaceMetric<Vec<f32>> for CosineMetric {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        const EPSILON: f32 = 1e-6;
        let cos = if na < EPSILON || nb < EPSILON {
            0.0
        } else {
            dot / (na * nb)
        };
        let dist = 1.0 - cos;
        dist.to_bits()
    }
}

#[derive(Clone, Copy)]
pub struct EuclideanMetric;

impl SpaceMetric<Vec<f32>> for EuclideanMetric {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt().to_bits()
    }
}

pub enum Index<const M: usize, const M0: usize> {
    Cosine(Hnsw<CosineMetric, Vec<f32>, Pcg64, M, M0>),
    Euclidean(Hnsw<EuclideanMetric, Vec<f32>, Pcg64, M, M0>),
}

impl<const M: usize, const M0: usize> Index<M, M0> {
    pub fn new(metric: Metric) -> Self {
        match metric {
            Metric::Cosine => Index::Cosine(Hnsw::new(CosineMetric)),
            Metric::Euclidean => Index::Euclidean(Hnsw::new(EuclideanMetric)),
        }
    }

    pub fn new_params(metric: Metric, ef_construction: usize) -> Self {
        let params = hnsw::Params::new().ef_construction(ef_construction);
        match metric {
            Metric::Cosine => Index::Cosine(Hnsw::new_params(CosineMetric, params)),
            Metric::Euclidean => Index::Euclidean(Hnsw::new_params(EuclideanMetric, params)),
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, searcher: &mut hnsw::Searcher<u32>) {
        match self {
            Index::Cosine(h) => h.insert(vector, searcher),
            Index::Euclidean(h) => h.insert(vector, searcher),
        };
    }

    pub fn feature(&self, i: usize) -> &Vec<f32> {
        match self {
            Index::Cosine(h) => h.feature(i),
            Index::Euclidean(h) => h.feature(i),
        }
    }

    pub fn nearest<'a>(
        &self,
        query: &Vec<f32>,
        ef: usize,
        searcher: &mut hnsw::Searcher<u32>,
        neighbors: &'a mut [Neighbor<u32>],
    ) -> &'a mut [Neighbor<u32>] {
        match self {
            Index::Cosine(h) => h.nearest(query, ef, searcher, neighbors),
            Index::Euclidean(h) => h.nearest(query, ef, searcher, neighbors),
        }
    }
}
