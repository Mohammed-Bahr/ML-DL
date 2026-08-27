"""ANN controller for one Rust car.

Rust sends one JSON object per line:
    {"sensors": [120, 80, 90, 110, 100], "speed": 150}

This program returns one JSON object per line:
    {"throttle": 0.73, "steer": -0.18}

The weights are random until a trained checkpoint is loaded. This file shows
the input -> neural network -> controls data flow.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


# These values match car.rs.
SENSOR_RANGE = 220.0
MAX_SPEED = 340.0


class CarANN(nn.Module):
    """6 inputs -> hidden layers -> throttle and steering."""

    def __init__(self):
        super().__init__()

        # Input order: front, left, right, front_left, front_right, speed.
        self.hidden = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.throttle_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),       # throttle: 0..1
        )
        self.steer_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),          # steer: -1..1
        )

    def forward(self, inputs):
        features = self.hidden(inputs)
        throttle = self.throttle_head(features)
        steer = self.steer_head(features)
        return torch.cat((throttle, steer), dim=1)


def make_inputs(sensors, speed):
    """Normalize Rust's raw values into one ANN input row."""
    if len(sensors) != 5:
        raise ValueError("expected exactly 5 sensor distances")

    normalized_sensors = [
        max(0.0, min(float(distance), SENSOR_RANGE)) / SENSOR_RANGE
        for distance in sensors
    ]
    normalized_speed = max(-MAX_SPEED, min(float(speed), MAX_SPEED)) / MAX_SPEED

    # Shape: [batch=1, features=6].
    return torch.tensor(
        [normalized_sensors + [normalized_speed]],
        dtype=torch.float32,
    )


def predict(model, sensors, speed):
    """Process one observation and return the controls for Rust."""
    inputs = make_inputs(sensors, speed)

    with torch.no_grad():
        output = model(inputs)[0]

    return {
        "throttle": float(output[0]),
        "steer": float(output[1]),
    }


def run_controller(model):
    """Read observations from Rust and write controls back to Rust."""
    for line in sys.stdin:
        if not line.strip():
            continue

        try:
            message = json.loads(line)
            controls = predict(model, message["sensors"], message["speed"])
            print(json.dumps(controls), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)


def main():
    model = CarANN()

    # Optional usage: python3 src/ANN/ANN.py model.pt
    if len(sys.argv) > 1:
        checkpoint = Path(sys.argv[1])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    model.eval()
    run_controller(model)


if __name__ == "__main__":
    main()
