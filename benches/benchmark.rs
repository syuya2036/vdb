use criterion::{criterion_group, criterion_main, Criterion};
use vdb::{VectorDB, Metadata, Metric};

fn search_benchmark(c: &mut Criterion) {
    let path = "bench.vdb";
    let _ = std::fs::remove_file(path);
    let mut db = VectorDB::open(path, Metric::Cosine).unwrap();
    for i in 0..1000 {
        let vector = vec![i as f32, i as f32 / 2.0, i as f32 / 3.0];
        let metadata = Metadata { label: i.to_string(), description: None };
        db.add(i as usize, vector, metadata).unwrap();
    }
    let query = vec![1.0, 0.5, 0.33];
    c.bench_function("search 10 nn", |b| {
        b.iter(|| db.search(&query, 10).unwrap())
    });
    std::fs::remove_file(path).unwrap();
}

criterion_group!(benches, search_benchmark);
criterion_main!(benches);
