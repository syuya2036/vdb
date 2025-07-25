use anyhow::Result;
use std::fs;
use vdb::{Metadata, Metric, VectorDB};

#[test]
fn remove_update() -> Result<()> {
    let path = "crud.vdb";
    let _ = fs::remove_file(path);
    let mut db = VectorDB::<12, 24>::open(path, Metric::Cosine)?;
    db.add(
        1,
        vec![0.0, 0.0],
        Metadata {
            label: "a".into(),
            description: None,
        },
    )?;
    db.add(
        2,
        vec![1.0, 1.0],
        Metadata {
            label: "b".into(),
            description: None,
        },
    )?;
    db.remove(1)?;
    let results = db.search(&vec![1.0, 1.0], 2)?;
    assert_eq!(results[0].id, 2);
    db.update(
        2,
        vec![0.0, 1.0],
        Metadata {
            label: "c".into(),
            description: None,
        },
    )?;
    let results = db.search(&vec![0.0, 1.0], 1)?;
    assert_eq!(results[0].id, 2);
    fs::remove_file(path)?;
    Ok(())
}
