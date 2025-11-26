# Monte Carlo AI Model Benchmark 📊

This framework provides a statistically robust method for benchmarking the latency and consistency of Hugging Face transformer models, particularly within transient, cloud-based environments like Google Colab.

## What is This Benchmark?

This is a **Monte Carlo Simulation** for AI model inference.

Instead of measuring a model's speed with a single, quick run—which is highly susceptible to temporary system noise (e.g., Colab's background processes, network latency, resource allocation spikes)—this benchmark uses repetitive sampling to achieve a statistically significant measure of performance.

For each model, the benchmark performs **$N$ number of independent runs** (e.g., $N=50$ or $N=100$), calculates the latency for each run, and then uses these samples to generate robust performance metrics.

-----

## Why Use the Monte Carlo Approach?

Benchmarking in cloud environments (like Google Colab or shared GPU servers) is challenging because **variance is high**. The Monte Carlo simulation addresses this with three key statistical advantages:

1.  **Robustness (True Average Speed):** By averaging many runs, the $\text{Mean Latency}$ is a much more reliable indicator of the model's typical speed than a single measurement.
2.  **Confidence Intervals (Statistical Certainty):** The **95% Confidence Interval (CI)** defines a range where the model's "true" mean speed is expected to fall $95\%$ of the time. This allows you to claim with statistical confidence that one model is faster than another if their CIs do not overlap.
3.  **Consistency (Stability Measurement):** The **Coefficient of Variation ($\text{CV}$)** quantifies stability by dividing the Standard Deviation ($\text{Std Dev}$) by the Mean.
      * **Low $\text{CV}$ (e.g., $0.01$):** Indicates the model's speed is highly stable and consistent.
      * **High $\text{CV}$ (e.g., $0.35$):** Indicates the model's speed fluctuates significantly between calls, suggesting poor stability for real-time applications.

-----

## How to Use the Benchmark

### 1\. Environment Setup

This benchmark is optimized for **Google Colab** with GPU acceleration.

  * **Setup:** Open the provided notebook file in Google Colab.
  * **Runtime:** Go to **Runtime \> Change runtime type** and select **GPU** as the hardware accelerator.

Execute the following cell in your Colab notebook to install dependencies:

```python
# Cell 1: Setup and Installations
!pip install transformers numpy scipy torch datasets
```

### 2\. Define Your Models

Modify the `model_configs` list in the execution cell (Cell 3 of the original code) to specify the Hugging Face models you wish to test.

| Parameter | Description |
| :--- | :--- |
| `name` | Your chosen name for the benchmark results. |
| `task` | The Hugging Face pipeline task (e.g., `"sentiment-analysis"`, `"question-answering"`). |
| `model` | The specific model ID from the Hugging Face Hub (e.g., `"bert-base-uncased"`). |
| `num_runs` | The number of Monte Carlo iterations (e.g., `100` is recommended). |
| `max_samples` | The number of samples to process in **each** run. |

**Example Configuration:**

```python
model_configs = [
    ModelConfig(
        name="DistilBERT Fast",
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        num_runs=50,
        max_samples=50
    ),
    ModelConfig(
        name="BART Large QA",
        task="question-answering",
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        num_runs=10
    )
]
```

### 3\. Run the Benchmark

Execute the main code block (Cell 2 and Cell 3) containing the `ModelBenchmark` class definition and the execution loop. The framework will automatically:

1.  Load each model onto the available device (GPU or CPU).
2.  Generate synthetic test data appropriate for the task.
3.  Run the inference loop for `num_runs` iterations.
4.  Apply statistical analysis.

### 4\. Analyze the Results

The final output provides a detailed performance summary.

| Metric | Interpretation | Example Result |
| :--- | :--- | :--- |
| **Latency (Mean)** | The most reliable average time required to process a single input sample (in milliseconds). **Lower is better.** | `60.1179 ms/sample` |
| **Latency (Std Dev)** | The degree of variation in speed across all runs. **Lower is better.** | `21.0208 ms/sample` |
| **95% CI** | The range where the *true* mean latency is $95\%$ likely to be found. | `[54.1439 ms, 66.0920 ms]` |
| **CV** | The **Coefficient of Variation** ($\text{Std Dev} / \text{Mean}$). A measure of **consistency**. **Lower is better.** | `0.35` |

#### Interpretation Example

From the provided results:

| Model | Latency (Mean) | Std Dev | CV |
| :--- | :--- | :--- | :--- |
| **DistilBERT Sentiment** | $60.1$ ms | $21.0$ ms | $0.35$ |
| **BERT QA** | $479.2$ ms | $5.6$ ms | $0.01$ |

  * **Speed:** **DistilBERT** is significantly faster (approx. $8 \times$ faster than $\text{BERT QA}$).
  * **Consistency:** **BERT QA** is extremely consistent ($\text{CV}=0.01$), while **DistilBERT** is highly volatile ($\text{CV}=0.35$), meaning its speed can vary dramatically from one call to the next. For real-time services, you might prefer the stability of BERT, even if it is slower.

---

> Made by Mosi-sol from BLUE LOTUS
