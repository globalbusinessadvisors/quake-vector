//! Watchdog manager: hardware watchdog for production, mock for development.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};

use tracing::warn;

/// Manages hardware watchdog timer (or mock mode on dev machines).
pub struct WatchdogManager {
    file: Option<File>,
    timeout_secs: u8,
}

impl WatchdogManager {
    /// Open /dev/watchdog. Falls back to mock mode if unavailable.
    pub fn new(timeout_secs: u8) -> io::Result<Self> {
        match OpenOptions::new().write(true).open("/dev/watchdog") {
            Ok(file) => Ok(Self {
                file: Some(file),
                timeout_secs,
            }),
            Err(_) => {
                warn!("watchdog: /dev/watchdog not available, operating in mock mode");
                Ok(Self {
                    file: None,
                    timeout_secs,
                })
            }
        }
    }

    /// Send heartbeat to the watchdog.
    pub fn heartbeat(&mut self) -> io::Result<()> {
        if let Some(ref mut f) = self.file {
            f.write_all(b"1")?;
            f.flush()?;
        }
        Ok(())
    }

    /// Returns true if running in mock mode (no real watchdog).
    pub fn is_mock(&self) -> bool {
        self.file.is_none()
    }

    /// Configured timeout in seconds.
    pub fn timeout_secs(&self) -> u8 {
        self.timeout_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_mode_heartbeat_succeeds() {
        // On dev machines, /dev/watchdog won't exist
        let mut wdg = WatchdogManager::new(10).unwrap();
        assert!(wdg.is_mock());
        wdg.heartbeat().unwrap();
        assert_eq!(wdg.timeout_secs(), 10);
    }
}
