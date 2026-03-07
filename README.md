# precise-reserve

An actuarial reserving engine built in Python.

## Overview

`precise-reserve` is a Python-based actuarial reserving engine designed to estimate insurance loss reserves using established actuarial methods. It provides a structured framework for loading claims data, running reserving algorithms, and generating output reports.

## Project Structure

```
precise-reserve/
├── data/           # Input data (triangles, claims, exposure data)
├── engine/         # Core reserving logic and actuarial methods
├── tests/          # Unit and integration tests
├── output/         # Generated reports, charts, and results
├── README.md
└── requirements.txt
```

## Methods (Planned)

- **Chain-Ladder (Development)** — projects loss development factors to ultimate
- **Bornhuetter-Ferguson** — blends development and a priori loss ratios
- **Cape Cod** — derives expected loss ratios from the data itself
- **Average Development** — smoothed factor selection methods

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- `pandas` — data manipulation and triangle management
- `numpy` — numerical computations
- `matplotlib` — visualization of development patterns and reserve estimates
- `scipy` — statistical methods and curve fitting

## Usage

Place loss triangle or claims data in `/data`, run the engine from `/engine`, and find results in `/output`.

## License

MIT
