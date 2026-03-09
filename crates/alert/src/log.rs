//! Append-only JSON-line alert log.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use quake_vector_types::Alert;

/// Append-only log of emitted alerts, one JSON line per alert.
pub struct AlertLog {
    file: File,
    path: PathBuf,
}

impl AlertLog {
    /// Create or open an alert log in the given directory.
    pub fn new(log_dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(log_dir)?;
        let path = log_dir.join("alerts.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self { file, path })
    }

    /// Append an alert as a JSON line.
    pub fn append(&mut self, alert: &Alert) -> io::Result<()> {
        let json = serde_json::to_string(alert)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        writeln!(self.file, "{}", json)?;
        self.file.flush()
    }

    /// Path to the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}
