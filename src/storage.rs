use crate::types::{Metadata, Metric};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: [u8; 4] = *b"VDB0";
pub const VERSION: u8 = 1;

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u8,
    pub metric: Metric,
    pub dim: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StoredEntry {
    pub id: usize,
    pub vector: Vec<f32>,
    pub metadata: Metadata,
}

pub struct Storage {
    path: PathBuf,
}

impl Storage {
    pub fn create<P: AsRef<Path>>(path: P, metric: Metric) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            metric,
            dim: 0,
        };
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &header)?;
        writer.flush()?;
        Ok(Self { path })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<(Self, Header, Vec<StoredEntry>)> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let header: Header = bincode::deserialize_from(&mut reader)?;
        if header.magic != MAGIC {
            return Err(anyhow!("invalid magic"));
        }
        if header.version != VERSION {
            return Err(anyhow!("unsupported version"));
        }
        let mut entries = Vec::new();
        loop {
            match bincode::deserialize_from::<_, StoredEntry>(&mut reader) {
                Ok(e) => entries.push(e),
                Err(e) => {
                    if let bincode::ErrorKind::Io(ref io_err) = *e {
                        if io_err.kind() == std::io::ErrorKind::UnexpectedEof {
                            break;
                        }
                    }
                    return Err(e.into());
                }
            }
        }
        Ok((Self { path }, header, entries))
    }

    pub fn append_entry(&self, entry: &StoredEntry) -> Result<()> {
        let file = OpenOptions::new().append(true).open(&self.path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, entry)?;
        writer.flush()?;
        Ok(())
    }

    pub fn update_header(&self, header: &Header) -> Result<()> {
        let file = OpenOptions::new().write(true).open(&self.path)?;
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::Start(0))?;
        bincode::serialize_into(&mut writer, header)?;
        writer.flush()?;
        Ok(())
    }
}
