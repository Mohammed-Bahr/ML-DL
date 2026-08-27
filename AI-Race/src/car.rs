use crate::track::Track;
use crate::vec2::Vec2;

pub const CAR_W: f32 = 28.0; // car body length (drawn as a rotated rectangle)
pub const CAR_H: f32 = 15.0; // car body width

const ACCEL: f32 = 260.0; // engine power, pixels/s^2
const BRAKE: f32 = 500.0;
const TURN_RATE: f32 = 3.2; // radians per second at full speed
const DRAG: f32 = 0.6; // natural slow-down per second
const MAX_SPEED: f32 = 340.0;

// --- Sensors -------------------------------------------------------------
/// Ray directions relative to the car's heading, in radians.
/// 0 = front ("top"), then left, right, front-left, front-right.
pub const SENSORS: [f32; 5] = [
    0.0,
    -std::f32::consts::FRAC_PI_2, // left
    std::f32::consts::FRAC_PI_2,  // right
    -std::f32::consts::FRAC_PI_4, // front-left
    std::f32::consts::FRAC_PI_4,  // front-right
];
pub const SENSOR_RANGE: f32 = 220.0;
const SENSOR_STEP: f32 = 4.0; // ray-march resolution in pixels

/// Simple RGB so car.rs stays independent of raylib (main.rs converts it).
#[derive(Clone, Copy)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// One car. Both players use this same struct.
pub struct Car {
    pub color: Rgb,
    pub pos: Vec2,
    pub angle: f32, // heading in radians (0 = right, PI/2 = down)
    pub speed: f32, // pixels per second (negative = reversing)
}

impl Car {
    pub fn new(color: Rgb, pos: Vec2, angle: f32) -> Self {
        Self { color, pos, angle, speed: 0.0 }
    }

    /// Move the car for one frame.
    ///
    /// * `throttle`: +1 = gas, -1 = brake/reverse
    /// * `steer`:    -1 = left, +1 = right
    /// * `on_track`: driving on grass is much slower
    pub fn update(&mut self, throttle: f32, steer: f32, on_track: bool, dt: f32) {
        // Engine / braking
        if throttle > 0.0 {
            self.speed += ACCEL * throttle * dt;
        } else if throttle < 0.0 {
            self.speed -= BRAKE * (-throttle) * dt;
        }

        // Drag is stronger on grass, which punishes cutting corners
        let drag = if on_track { DRAG } else { DRAG * 5.0 };
        let max_speed = if on_track { MAX_SPEED } else { MAX_SPEED * 0.45 };

        self.speed -= self.speed * drag * dt;
        self.speed = self.speed.clamp(-MAX_SPEED * 0.35, max_speed);

        // Steering only works while moving; direction flips when reversing.
        let speed_factor = (self.speed.abs() / MAX_SPEED).clamp(0.2, 1.0);
        self.angle += steer * TURN_RATE * speed_factor * dt * self.speed.signum();

        // Move along the heading
        let dir = Vec2::new(self.angle.cos(), self.angle.sin());
        self.pos = self.pos + dir * (self.speed * dt);
    }

    /// Cast the 5 sensor rays and return the distance to the first wall
    /// each one hits (or SENSOR_RANGE if nothing is in reach).
    pub fn sensor_distances(&self, track: &Track) -> [f32; 5] {
        let mut result = [SENSOR_RANGE; 5];

        for (i, offset) in SENSORS.iter().enumerate() {
            let dir = Vec2::new((self.angle + offset).cos(), (self.angle + offset).sin());
            let mut dist = CAR_W / 2.0; // start just outside the body

            while dist <= SENSOR_RANGE {
                let point = self.pos + dir * dist;
                if !track.on_track(point) {
                    break; // this ray found a wall
                }
                dist += SENSOR_STEP;
            }
            result[i] = dist.min(SENSOR_RANGE);
        }

        result
    }

    /// Push overlapping cars apart and make them lose a bit of speed.
    pub fn resolve_car_collision(a: &mut Car, b: &mut Car) {
        let diff = b.pos - a.pos;
        let dist = diff.length();
        if dist < CAR_W && dist > 0.001 {
            let push = diff.normalize() * (CAR_W - dist) * 0.5;
            a.pos = a.pos - push;
            b.pos = b.pos + push;
            a.speed *= 0.97;
            b.speed *= 0.97;
        }
    }
}
