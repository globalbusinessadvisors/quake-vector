//! GPIO controller for physical alert indicators (LEDs, buzzer).

use std::io;

use quake_vector_types::AlertLevel;

/// Trait for GPIO output operations.
pub trait GpioOutput {
    fn set_high(&mut self, pin: u8) -> io::Result<()>;
    fn set_low(&mut self, pin: u8) -> io::Result<()>;
}

/// Pin state change record for testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinChange {
    pub pin: u8,
    pub high: bool,
}

/// Mock GPIO that records all pin state changes.
pub struct MockGpio {
    pub changes: Vec<PinChange>,
    pub pin_states: [bool; 32],
}

impl MockGpio {
    pub fn new() -> Self {
        Self {
            changes: Vec::new(),
            pin_states: [false; 32],
        }
    }
}

impl Default for MockGpio {
    fn default() -> Self {
        Self::new()
    }
}

impl GpioOutput for MockGpio {
    fn set_high(&mut self, pin: u8) -> io::Result<()> {
        self.changes.push(PinChange { pin, high: true });
        if (pin as usize) < self.pin_states.len() {
            self.pin_states[pin as usize] = true;
        }
        Ok(())
    }

    fn set_low(&mut self, pin: u8) -> io::Result<()> {
        self.changes.push(PinChange { pin, high: false });
        if (pin as usize) < self.pin_states.len() {
            self.pin_states[pin as usize] = false;
        }
        Ok(())
    }
}

/// Linux sysfs GPIO output.
pub struct LinuxGpio;

impl LinuxGpio {
    pub fn new() -> Self {
        Self
    }

    fn write_sysfs(path: &str, value: &str) -> io::Result<()> {
        std::fs::write(path, value)
    }

    fn ensure_exported(pin: u8) -> io::Result<()> {
        let dir_path = format!("/sys/class/gpio/gpio{pin}");
        if !std::path::Path::new(&dir_path).exists() {
            Self::write_sysfs("/sys/class/gpio/export", &pin.to_string())?;
        }
        Self::write_sysfs(&format!("{dir_path}/direction"), "out")?;
        Ok(())
    }
}

impl Default for LinuxGpio {
    fn default() -> Self {
        Self::new()
    }
}

impl GpioOutput for LinuxGpio {
    fn set_high(&mut self, pin: u8) -> io::Result<()> {
        Self::ensure_exported(pin)?;
        Self::write_sysfs(&format!("/sys/class/gpio/gpio{pin}/value"), "1")
    }

    fn set_low(&mut self, pin: u8) -> io::Result<()> {
        Self::ensure_exported(pin)?;
        Self::write_sysfs(&format!("/sys/class/gpio/gpio{pin}/value"), "0")
    }
}

/// Pin assignments.
const PIN_AMBER_LED: u8 = 17;
const PIN_RED_LED: u8 = 27;
const PIN_BUZZER: u8 = 22;

/// GPIO-based alert service mapping AlertLevel to physical outputs.
pub struct GpioAlertService {
    gpio: Box<dyn GpioOutput>,
}

impl GpioAlertService {
    pub fn new(gpio: Box<dyn GpioOutput>) -> Self {
        Self { gpio }
    }

    /// Trigger GPIO outputs based on alert level.
    ///
    /// - LOW: no GPIO action
    /// - MEDIUM: pin 17 high (amber LED)
    /// - HIGH: pin 27 high (red LED blink — caller schedules low after 500ms)
    /// - CRITICAL: pin 27 high (steady red), pin 22 high (buzzer)
    pub fn trigger(&mut self, level: AlertLevel) -> io::Result<()> {
        match level {
            AlertLevel::Low => {
                // No GPIO action
            }
            AlertLevel::Medium => {
                self.gpio.set_high(PIN_AMBER_LED)?;
            }
            AlertLevel::High => {
                self.gpio.set_high(PIN_RED_LED)?;
                // Caller responsible for scheduling set_low after 500ms
            }
            AlertLevel::Critical => {
                self.gpio.set_high(PIN_RED_LED)?;
                self.gpio.set_high(PIN_BUZZER)?;
            }
        }
        Ok(())
    }

    /// Clear all alert pins.
    pub fn clear(&mut self) -> io::Result<()> {
        self.gpio.set_low(PIN_AMBER_LED)?;
        self.gpio.set_low(PIN_RED_LED)?;
        self.gpio.set_low(PIN_BUZZER)?;
        Ok(())
    }

    /// Access the underlying GPIO for testing.
    pub fn gpio(&self) -> &dyn GpioOutput {
        &*self.gpio
    }

    /// Access the underlying GPIO mutably for testing.
    pub fn gpio_mut(&mut self) -> &mut dyn GpioOutput {
        &mut *self.gpio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_alert_no_gpio_action() {
        let mock = MockGpio::new();
        let mut svc = GpioAlertService::new(Box::new(mock));
        svc.trigger(AlertLevel::Low).unwrap();
        // Downcast to check
        let changes = get_changes(&svc);
        assert!(changes.is_empty(), "LOW should produce no GPIO changes");
    }

    #[test]
    fn medium_alert_amber_led() {
        let mock = MockGpio::new();
        let mut svc = GpioAlertService::new(Box::new(mock));
        svc.trigger(AlertLevel::Medium).unwrap();
        let changes = get_changes(&svc);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], PinChange { pin: 17, high: true });
    }

    #[test]
    fn high_alert_red_led() {
        let mock = MockGpio::new();
        let mut svc = GpioAlertService::new(Box::new(mock));
        svc.trigger(AlertLevel::High).unwrap();
        let changes = get_changes(&svc);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], PinChange { pin: 27, high: true });
    }

    #[test]
    fn critical_alert_red_led_and_buzzer() {
        let mock = MockGpio::new();
        let mut svc = GpioAlertService::new(Box::new(mock));
        svc.trigger(AlertLevel::Critical).unwrap();
        let changes = get_changes(&svc);
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0], PinChange { pin: 27, high: true });
        assert_eq!(changes[1], PinChange { pin: 22, high: true });
    }

    #[test]
    fn clear_sets_all_pins_low() {
        let mock = MockGpio::new();
        let mut svc = GpioAlertService::new(Box::new(mock));
        svc.trigger(AlertLevel::Critical).unwrap();
        svc.clear().unwrap();
        let changes = get_changes(&svc);
        // After trigger(Critical): pin27 high, pin22 high
        // After clear: pin17 low, pin27 low, pin22 low
        assert_eq!(changes.len(), 5);
        assert_eq!(changes[2], PinChange { pin: 17, high: false });
        assert_eq!(changes[3], PinChange { pin: 27, high: false });
        assert_eq!(changes[4], PinChange { pin: 22, high: false });
    }

    /// Helper to extract changes from the service's MockGpio.
    fn get_changes(svc: &GpioAlertService) -> Vec<PinChange> {
        // We need to use unsafe to downcast since we know it's a MockGpio
        let gpio_ref = svc.gpio();
        let mock_ptr = gpio_ref as *const dyn GpioOutput as *const MockGpio;
        // SAFETY: we know the underlying type is MockGpio in tests
        unsafe { (*mock_ptr).changes.clone() }
    }
}
