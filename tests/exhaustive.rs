use anyhow::Result;
use std::fs;
use vdb::{Metadata, Metric, VectorDB};

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn exhaustive_search() -> Result<()> {
    let path = "exhaustive.vdb";
    let _ = fs::remove_file(path);
    let mut db = VectorDB::<12, 24>::open(path, Metric::Euclidean)?;
    let vectors = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    for (i, v) in vectors.iter().enumerate() {
        db.add(
            i,
            v.clone(),
            Metadata {
                label: i.to_string(),
                description: None,
            },
        )?;
    }
    let query = vec![1.0, 0.5];
    let results = db.search(&query, 3)?;
    let mut expected: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, distance(&query, v)))
        .collect();
    expected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (res, exp) in results.iter().zip(expected.iter()) {
        assert_eq!(res.id, exp.0);
    }
    fs::remove_file(path)?;
    Ok(())
}
