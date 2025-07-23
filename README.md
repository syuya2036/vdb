# vdb
`vdb` は Rust 製の軽量ベクトルデータベースです。  
高次元ベクトルに対して、高速な近傍検索（ANN）を提供します。  
すべてのデータは SQLite のように単一ファイルで永続化されます。

## 特徴

- 高速な近傍検索（HNSWアルゴリズム）
- ベクトルとメタデータの登録・検索に対応
- 単一の `.vdb` ファイルにすべて保存
- 明示的な save/load は不要。

## 使い方

```rust
use vdb::{VectorDB, Metadata, Metric};

let mut db = VectorDB::open("example.vdb", Metric::Cosine)?;

let vector = vec![0.1, 0.2, 0.3, 0.4];
let metadata = Metadata {
    label: "sample".to_string(),
    description: Some("これはサンプルです".to_string()),
};

db.add(1, vector, metadata)?;

let query = vec![0.1, 0.2, 0.3, 0.4];
let results = db.search(&query, 5)?;

for result in results {
    println!("ID: {}, 距離: {}, ラベル: {}", result.id, result.distance, result.metadata.label);
}
```

## データ構造

```rust
struct Metadata {
    label: String,
    description: Option<String>,
}

struct SearchResult {
    id: usize,
    distance: f32,
    metadata: Metadata,
}
```

## ベンチマーク

`cargo bench` を実行すると簡単なベンチマークが走ります。1000 件のベクトルを登録した後、10 個の近傍を検索する処理を計測した結果は次の通りです。

```
search 10 nn            time:   [7.8869 µs 7.9252 µs 7.9752 µs]
```