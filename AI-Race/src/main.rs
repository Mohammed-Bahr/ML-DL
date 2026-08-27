// All raylib code lives in this file only.
// track.rs, car.rs and game.rs are pure game logic with no raylib.

mod car;
mod game;
mod track;
mod vec2;

use raylib::prelude::*;
use raylib::RaylibHandle;

use crate::car::{Car, SENSOR_RANGE, SENSORS};
use crate::game::{Game, Input, State};
use crate::track::Track;

pub const SCREEN_W: i32 = 1200;
pub const SCREEN_H: i32 = 700;

fn main() {
    // 1. Open the window
    let (mut rl, thread) = raylib::init()
        .size(SCREEN_W, SCREEN_H)
        .title("Race - P1 WASD vs P2 Arrows")
        .build();
    rl.set_target_fps(60);

    // 2. Create the race
    let mut game = Game::new();

    // 3. Main loop: read input -> update logic -> draw one frame
    while !rl.window_should_close() {
        let dt = rl.get_frame_time();

        // R restarts at any time
        if rl.is_key_pressed(KeyboardKey::KEY_R) {
            game = Game::new();
        }

        let p1 = read_input(
            &rl,
            KeyboardKey::KEY_W,
            KeyboardKey::KEY_S,
            KeyboardKey::KEY_A,
            KeyboardKey::KEY_D,
        );
        let p2 = read_input(
            &rl,
            KeyboardKey::KEY_UP,
            KeyboardKey::KEY_DOWN,
            KeyboardKey::KEY_LEFT,
            KeyboardKey::KEY_RIGHT,
        );

        game.update(dt, p1, p2);

        let mut d = rl.begin_drawing(&thread);
        draw_scene(&game, &mut d);
    }
}

/// Turn four keys into a throttle/steer pair.
fn read_input(
    rl: &RaylibHandle,
    up: KeyboardKey,
    down: KeyboardKey,
    left: KeyboardKey,
    right: KeyboardKey,
) -> Input {
    Input {
        throttle: if rl.is_key_down(up) {
            1.0
        } else if rl.is_key_down(down) {
            -1.0
        } else {
            0.0
        },
        steer: if rl.is_key_down(left) {
            -1.0
        } else if rl.is_key_down(right) {
            1.0
        } else {
            0.0
        },
    }
}

// --- Drawing --------------------------------------------------------------

fn draw_scene<D: RaylibDraw>(game: &Game, d: &mut D) {
    draw_track(game.track(), d);

    for car in [game.cars().0, game.cars().1] {
        draw_sensors(car, game.track(), d);
        draw_car(car, d);
    }

    draw_hud(game, d);
}

fn draw_track<D: RaylibDraw>(track: &Track, d: &mut D) {
    let cx = track.center.x as i32;
    let cy = track.center.y as i32;

    // Grass background
    d.draw_rectangle(0, 0, SCREEN_W, SCREEN_H, Color::DARKGREEN);

    // Road: big gray ellipse, then cover the middle with grass again
    d.draw_ellipse(cx, cy, track.rx, track.ry, Color::GRAY);
    d.draw_ellipse(
        cx,
        cy,
        track.rx * Track::INNER,
        track.ry * Track::INNER,
        Color::DARKGREEN,
    );

    // Border walls (thick lines on both edges)
    for scale in [1.0, Track::INNER] {
        d.draw_ellipse_lines(cx, cy, track.rx * scale, track.ry * scale, Color::DARKGRAY);
        d.draw_ellipse_lines(cx, cy, track.rx * scale - 4.0, track.ry * scale - 4.0, Color::BLACK);
        d.draw_ellipse_lines(cx, cy, track.rx * scale - 8.0, track.ry * scale - 8.0, Color::DARKGRAY);
    }

    // Start/finish line at the bottom: checkered strip across the road
    let y_top = track.center.y + track.ry * Track::INNER;
    let y_bot = track.center.y + track.ry;
    let rows = ((y_bot - y_top) / 12.0) as i32;
    for row in 0..rows {
        let color = if row % 2 == 0 { Color::WHITE } else { Color::BLACK };
        d.draw_rectangle(cx - 8, (y_top + row as f32 * 12.0) as i32, 16, 12, color);
    }
}

fn draw_car<D: RaylibDraw>(car: &Car, d: &mut D) {
    // draw_rectangle_pro rotates around the given origin, so we pass the
    // car center as origin to rotate it in place around its middle.
    let rec = Rectangle::new(car.pos.x, car.pos.y, car::CAR_W, car::CAR_H);
    let origin = Vector2::new(car::CAR_W / 2.0, car::CAR_H / 2.0);
    let degrees = car.angle.to_degrees();
    let color = Color::new(car.color.r, car.color.g, car.color.b, 255);

    d.draw_rectangle_pro(rec, origin, degrees, color);

    // A small dark "windshield" near the front so direction is visible
    let front = car.pos + vec2::Vec2::new(car.angle.cos(), car.angle.sin()) * 6.0;
    let wrec = Rectangle::new(front.x, front.y, 8.0, car::CAR_H - 6.0);
    d.draw_rectangle_pro(wrec, Vector2::new(4.0, (car::CAR_H - 6.0) / 2.0), degrees, Color::DARKGRAY);
}

fn draw_sensors<D: RaylibDraw>(car: &Car, track: &Track, d: &mut D) {
    let distances = car.sensor_distances(track);

    for (i, offset) in SENSORS.iter().enumerate() {
        let dir = vec2::Vec2::new((car.angle + offset).cos(), (car.angle + offset).sin());
        let end = car.pos + dir * distances[i];

        let alpha = if distances[i] < SENSOR_RANGE { 200 } else { 60 };
        d.draw_line(
            car.pos.x as i32,
            car.pos.y as i32,
            end.x as i32,
            end.y as i32,
            Color::new(255, 255, 0, alpha),
        );
    }
}

fn draw_hud<D: RaylibDraw>(game: &Game, d: &mut D) {
    d.draw_text(
        &format!("P1   lap {} / {}", game.laps_p1(), game.laps_needed()),
        20,
        15,
        25,
        Color::SKYBLUE,
    );
    d.draw_text(
        &format!("P2   lap {} / {}", game.laps_p2(), game.laps_needed()),
        20,
        45,
        25,
        Color::RED,
    );
    d.draw_text(
        "WASD vs Arrows | R restart",
        SCREEN_W - 300,
        SCREEN_H - 30,
        20,
        Color::WHITE,
    );

    match game.state() {
        State::Countdown(t) => {
            let n = (*t as i32).clamp(1, 3);
            let msg = if n > 1 { n.to_string() } else { "GO!".to_string() };
            d.draw_text(&msg, SCREEN_W / 2 - 40, 60, 100, Color::YELLOW);
        }
        State::Finished(winner) => {
            d.draw_rectangle(0, 250, SCREEN_W, 160, Color::new(0, 0, 0, 180));
            let msg = if *winner == "P1" { "PLAYER 1 WINS!" } else { "PLAYER 2 WINS!" };
            let color = if *winner == "P1" { Color::SKYBLUE } else { Color::RED };
            d.draw_text(msg, SCREEN_W / 2 - 250, 280, 70, color);
            d.draw_text("Press R to race again", SCREEN_W / 2 - 130, 370, 25, Color::WHITE);
        }
        State::Racing => {}
    }
}
