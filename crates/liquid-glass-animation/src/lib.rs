//! Frame-rate independent animation primitives.

#![deny(unsafe_code)]

/// A critically-damped-friendly spring configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    pub stiffness: f32,
    pub damping: f32,
    pub mass: f32,
}

impl Spring {
    #[must_use]
    pub const fn gentle() -> Self {
        Self { stiffness: 220.0, damping: 26.0, mass: 1.0 }
    }

    #[must_use]
    pub const fn responsive() -> Self {
        Self { stiffness: 420.0, damping: 32.0, mass: 1.0 }
    }
}

impl Default for Spring {
    fn default() -> Self {
        Self::responsive()
    }
}

/// A scalar spring state advanced with a real frame delta.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpringValue {
    pub value: f32,
    pub velocity: f32,
}

impl SpringValue {
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self { value, velocity: 0.0 }
    }

    pub fn step(&mut self, target: f32, spring: Spring, delta_seconds: f32) {
        let delta = delta_seconds.clamp(0.0, 0.1);
        let acceleration = (target - self.value) * spring.stiffness / spring.mass
            - self.velocity * spring.damping / spring.mass;
        self.velocity += acceleration * delta;
        self.value += self.velocity * delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spring_moves_towards_target() {
        let mut value = SpringValue::new(0.0);
        for _ in 0..120 {
            value.step(1.0, Spring::responsive(), 1.0 / 60.0);
        }

        assert!(value.value > 0.95);
    }
}
