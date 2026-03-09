//! Seismic sensor ingestion: receives raw waveform data from sensors and
//! produces validated WaveformWindow values for downstream processing.

mod ring_buffer;
mod sensor;
mod classifier;
mod health;
mod stream;
mod discovery;
mod simulator;

pub use ring_buffer::RingBuffer;
pub use sensor::{SensorInterface, SensorError, MockSensor, WavePattern};
pub use classifier::WaveClassifierService;
pub use health::SensorHealthMonitor;
pub use stream::SensorStream;
pub use discovery::{SensorDiscoveryService, MockDiscoveryService};
pub use simulator::{SeismicEventSimulator, SimulatedSensor};
