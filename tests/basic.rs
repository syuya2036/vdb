use anyhow::Result;
use std::fs;
use vdb::{Metadata, Metric, VectorDB};

#[test]
fn basic_usage() -> Result<()> {
    let path = "test.vdb";
    let _ = fs::remove_file(path);
    {
        let mut db = VectorDB::<12, 24>::open(path, Metric::Cosine)?;
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
        let mut db = VectorDB::<12, 24>::open(path, Metric::Cosine)?;
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
    let mut db = VectorDB::<12, 24>::open(path, Metric::Cosine)?;
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
    let mut db = VectorDB::<12, 24>::open(path, Metric::Cosine)?;
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
        let mut db = VectorDB::<12, 24>::open(path, Metric::Cosine)?;
        db.add(
            1,
            vec![0.0, 0.0, 0.0],
            Metadata {
                label: "a".into(),
                description: None,
            },
        )?;
    }
    let err = VectorDB::<12, 24>::open(path, Metric::Euclidean);
    assert!(err.is_err());
    fs::remove_file(path)?;
    Ok(())
}
