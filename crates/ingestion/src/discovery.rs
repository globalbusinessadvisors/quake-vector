//! Sensor discovery service: trait-based sensor enumeration so real hardware
//! discovery (I2C/SPI/USB) can be swapped in later.

use quake_vector_types::SensorId;

use crate::sensor::{MockSensor, SensorInterface, WavePattern};

/// Trait for discovering available sensors.
pub trait SensorDiscoveryService {
    fn discover(&self) -> Vec<Box<dyn SensorInterface>>;
}

/// A mock discovery service that returns a configurable list of MockSensors.
pub struct MockDiscoveryService {
    configs: Vec<MockSensorConfig>,
}

/// Configuration for a single mock sensor.
pub struct MockSensorConfig {
    pub sensor_id: SensorId,
    pub sample_rate: u16,
    pub pattern: WavePattern,
    pub amplitude: f64,
    pub noise_level: f64,
}

impl MockDiscoveryService {
    pub fn new(configs: Vec<MockSensorConfig>) -> Self {
        Self { configs }
    }
}

impl SensorDiscoveryService for MockDiscoveryService {
    fn discover(&self) -> Vec<Box<dyn SensorInterface>> {
        self.configs
            .iter()
            .map(|cfg| {
                Box::new(MockSensor::new(
                    cfg.sensor_id,
                    cfg.sample_rate,
                    cfg.pattern,
                    cfg.amplitude,
                    cfg.noise_level,
                )) as Box<dyn SensorInterface>
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_returns_configured_sensors() {
        let service = MockDiscoveryService::new(vec![
            MockSensorConfig {
                sensor_id: SensorId(1),
                sample_rate: 100,
                pattern: WavePattern::PWave,
                amplitude: 1000.0,
                noise_level: 0.0,
            },
            MockSensorConfig {
                sensor_id: SensorId(2),
                sample_rate: 200,
                pattern: WavePattern::SWave,
                amplitude: 500.0,
                noise_level: 0.1,
            },
        ]);

        let sensors = service.discover();
        assert_eq!(sensors.len(), 2);
        assert_eq!(sensors[0].sensor_id(), SensorId(1));
        assert_eq!(sensors[0].sample_rate(), 100);
        assert_eq!(sensors[1].sensor_id(), SensorId(2));
        assert_eq!(sensors[1].sample_rate(), 200);
    }
}
