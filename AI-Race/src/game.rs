use crate::car::{Car, Rgb};
use crate::track::Track;
use crate::vec2::Vec2;

const LAPS_NEEDED: u32 = 3;

/// Controls for one car this frame (built from the keyboard in main.rs).
#[derive(Clone, Copy)]
pub struct Input {
    pub throttle: f32, // +1 gas, -1 brake
    pub steer: f32,    // -1 left, +1 right
}

/// What phase the race is in.
pub enum State {
    /// Counts down from 3 before the start.
    Countdown(f32),
    Racing,
    /// Someone won; holds their name.
    Finished(&'static str),
}

/// All race logic — no raylib here. main.rs reads keys and draws.
pub struct Game {
    track: Track,
    player1: Car,
    player2: Car,
    state: State,

    // Lap tracking: total radians traveled forward around the oval.
    // One full lap = TAU radians. Only counts while on asphalt.
    p1_progress: f32,
    p2_progress: f32,
    p1_prev_angle: f32,
    p2_prev_angle: f32,
}

impl Game {
    pub fn new() -> Self {
        let track = Track::new();

        // Both cars start at the bottom on the start/finish line,
        // stacked across the road width, facing left (angle = PI).
        let start_angle = std::f32::consts::PI;
        let player1 = Car::new(
            Rgb { r: 100, g: 200, b: 255 },
            Vec2::new(track.center.x, track.center.y + track.ry * 0.80),
            start_angle,
        );
        let player2 = Car::new(
            Rgb { r: 255, g: 80, b: 80 },
            Vec2::new(track.center.x + 40.0, track.center.y + track.ry * 0.93),
            start_angle,
        );

        Self {
            p1_prev_angle: track.angle_of(player1.pos),
            p2_prev_angle: track.angle_of(player2.pos),
            p1_progress: 0.0,
            p2_progress: 0.0,
            track,
            player1,
            player2,
            state: State::Countdown(3.999),
        }
    }

    pub fn track(&self) -> &Track {
        &self.track
    }

    pub fn cars(&self) -> (&Car, &Car) {
        (&self.player1, &self.player2)
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn laps_needed(&self) -> u32 {
        LAPS_NEEDED
    }

    pub fn laps_p1(&self) -> u32 {
        self.laps(self.p1_progress)
    }

    pub fn laps_p2(&self) -> u32 {
        self.laps(self.p2_progress)
    }

    pub fn update(&mut self, dt: f32, in1: Input, in2: Input) {
        match &mut self.state {
            State::Countdown(t) => {
                *t -= dt;
                if *t <= 0.0 {
                    self.state = State::Racing;
                }
            }
            State::Racing => {
                self.player1.update(
                    in1.throttle,
                    in1.steer,
                    self.track.on_track(self.player1.pos),
                    dt,
                );
                self.player2.update(
                    in2.throttle,
                    in2.steer,
                    self.track.on_track(self.player2.pos),
                    dt,
                );

                // Border collisions (walls at both edges of the road)
                let track = &self.track;
                Self::handle_border(track, &mut self.player1);
                Self::handle_border(track, &mut self.player2);

                Car::resolve_car_collision(&mut self.player1, &mut self.player2);

                // Laps
                let track = &self.track;
                Self::track_lap(
                    track,
                    &mut self.p1_progress,
                    &mut self.p1_prev_angle,
                    &self.player1,
                );
                Self::track_lap(
                    track,
                    &mut self.p2_progress,
                    &mut self.p2_prev_angle,
                    &self.player2,
                );

                let done1 = self.laps(self.p1_progress) >= LAPS_NEEDED;
                let done2 = self.laps(self.p2_progress) >= LAPS_NEEDED;
                if done1 || done2 {
                    let winner = if done1 { "P1" } else { "P2" };
                    self.state = State::Finished(winner);
                }
            }
            State::Finished(_) => {}
        }
    }

    /// Bounce a car off a wall it just crossed.
    fn handle_border(track: &Track, car: &mut Car) {
        if let Some(fixed) = track.clamp_to_road(car.pos) {
            car.pos = fixed;
            car.speed *= -0.25; // small bounce back
        }
    }

    /// Add up how far around the oval a car has moved.
    ///
    /// We compare this frame's angle with last frame's angle and accumulate
    /// the difference. Driving backwards subtracts progress, so you cannot
    /// cheat by wiggling over the line. Progress only counts on asphalt.
    fn track_lap(track: &Track, progress: &mut f32, prev_angle: &mut f32, car: &Car) {
        let angle = track.angle_of(car.pos);
        let mut delta = angle - *prev_angle;
        // Wrap the delta to [-PI, PI] so crossing the line doesn't jump
        if delta > std::f32::consts::PI {
            delta -= std::f32::consts::TAU;
        } else if delta < -std::f32::consts::PI {
            delta += std::f32::consts::TAU;
        }
        *prev_angle = angle;

        if track.on_track(car.pos) {
            *progress += delta;
        }
    }

    /// Completed laps for a given accumulated progress.
    fn laps(&self, progress: f32) -> u32 {
        (progress / std::f32::consts::TAU).floor().max(0.0) as u32
    }
}
