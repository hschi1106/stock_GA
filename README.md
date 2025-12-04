# 📈 Genetic Algorithm for Stock Trading Strategy Optimization

This project implements a **Genetic Algorithm (GA)** to evolve and optimize stock trading strategies using historical market data. It uses combinations of technical indicators to construct rules and improves them through evolutionary principles like selection, crossover, and mutation.

---

## 🧠 Key Features

* **Modular Genetic Algorithm** implementation:

  * **Tournament Selection** with configurable pressure
  * **Single-point Crossover** between parent strategies
  * **Bitwise Mutation** with adjustable mutation rate
  * **Early Stopping** based on patience
* Support for custom or randomly initialized gene populations
* Easily customizable **fitness function** to define profitability
* **Progress visualization** using matplotlib
* Optional **warm-up period** before early stopping
* Can run on any gene structure: binary, integer, or real-valued

---

## 🛠️ Prerequisites

* Python 3.x
* Libraries:

  ```bash
  pip install numpy pandas tqdm matplotlib
  ```

---

## 📁 Data Preparation

1. Store historical stock price CSV files in the `./stock_data_dayK/` directory.
2. Each CSV should include at least the following columns:

   * `date`
   * `open`
   * `close`

---

## 🚀 Getting Started

1. Clone the repository.
2. Customize or provide a fitness function in `main.py` or your script.
3. Run the optimization:

```bash
python main.py
```

4. Monitor terminal logs for progress and view:

   * `fitness.png` for average fitness over time
   * `best_fitness.png` for best chromosome fitness progression

---

## 📊 Output

After the algorithm runs:

* It prints the **best-performing chromosome** (i.e., the optimal trading strategy found).
* Generates fitness graphs:

  * `fitness.png` - Tracks average fitness per generation
  * `best_fitness.png` - Tracks best chromosome fitness over generations

---

## 🔧 Parameters

| Parameter            | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `population_size`    | Number of candidate strategies in the population          |
| `gene_cnt`           | Number of genes (features or decision rules) per strategy |
| `gene_sample`        | Possible values for each gene (e.g., \[0, 1] for binary)  |
| `selection_pressure` | Controls elitism in selection; higher = more aggressive   |
| `mutation_rate`      | Probability of mutating each gene                         |
| `patience`           | Early stopping after this many rounds of no improvement   |
| `warmup`             | Minimum number of iterations before early stopping        |
| `init_population`    | Optional custom starting population                       |

---

## 📌 To-Do

* [ ] Add **roulette wheel selection** for probabilistic sampling
* [ ] Introduce advanced indicators: **RSI**, **MACD**, **Bollinger Bands**, etc.
* [ ] Improve **risk-adjusted fitness function**
* [ ] Implement **dynamic hyperparameter tuning**

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).