use clap::{Parser, Subcommand};
use vdb::{Metadata, Metric, VectorDB};

#[derive(Parser)]
#[command(name = "vdb")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add {
        path: String,
        id: usize,
        vector: String,
        label: String,
    },
    Search {
        path: String,
        vector: String,
        k: usize,
    },
    Remove {
        path: String,
        id: usize,
    },
}

fn parse_vector(s: &str) -> Vec<f32> {
    s.split(',').filter_map(|x| x.parse().ok()).collect()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Add {
            path,
            id,
            vector,
            label,
        } => {
            let mut db = VectorDB::<12, 24>::open(&path, Metric::Cosine)?;
            let vec = parse_vector(&vector);
            db.add(
                id,
                vec,
                Metadata {
                    label,
                    description: None,
                },
            )?;
        }
        Commands::Search { path, vector, k } => {
            let mut db = VectorDB::<12, 24>::open(&path, Metric::Cosine)?;
            let vec = parse_vector(&vector);
            let results = db.search(&vec, k)?;
            for r in results {
                println!("{} {}", r.id, r.distance);
            }
        }
        Commands::Remove { path, id } => {
            let mut db = VectorDB::<12, 24>::open(&path, Metric::Cosine)?;
            db.remove(id)?;
        }
    }
    Ok(())
}
