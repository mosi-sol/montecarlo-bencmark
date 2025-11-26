- Install the necessary libraries (transformers, torch, numpy, scipy).
- Explicitly select the execution device (GPU is highly recommended for model benchmarking).
- Define the complete Python classes and functions described by the notebook's structure.

## Cell 1: Setup and Installations
This cell ensures all required libraries are installed and the GPU is correctly configured for use with the Hugging Face pipeline.
```python
# Install required libraries
!pip install transformers numpy scipy torch datasets

# Import necessary libraries
import time
import numpy as np
from scipy import stats
from transformers import pipeline
from dataclasses import dataclass
import torch
import warnings

# Suppress Hugging Face warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration for GPU/Device ---
# Check for GPU and assign device (0 for GPU, -1 for CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
```

## Cell 2: Model Configuration and Benchmark Class
This cell defines the structure for holding model information and the core ModelBenchmark class, which handles data generation, running the simulation, and statistical analysis.
```python
@dataclass
class ModelConfig:
    """Configuration for a single model to benchmark."""
    name: str
    task: str
    model: str
    num_runs: int = 100
    max_samples: int = 50  # Max samples to pass to a single pipeline call

class ModelBenchmark:
    """
    Monte Carlo simulation framework for benchmarking Hugging Face AI models.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the benchmark framework."""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.pipeline = None

        # Define generators for sample test data based on task
        self.test_generators = {
            "sentiment-analysis": self._generate_sentiment_data,
            "question-answering": self._generate_qa_data,
            # Add other tasks here (e.g., "text-generation")
        }

        # Example texts for test data generation
        self.base_texts = [
            "The French Revolution began in 1789 and led to the overthrow of the monarchy.",
            "Mount Everest is the highest mountain in the world, standing at 8,848 meters tall.",
            "The human brain contains approximately 86 billion neurons.",
            "Artificial intelligence is rapidly advancing and changing industries.",
            "The project was an overwhelming success and everyone was thrilled."
        ]
        self.base_questions = [
            "What is the highest mountain?",
            "When did the French Revolution start?",
            "What is the brain composed of?"
        ]

    def _generate_sentiment_data(self, num_samples: int):
        """Generates simple text samples for sentiment analysis."""
        data = np.random.choice(self.base_texts, size=num_samples, replace=True)
        return data.tolist()

    def _generate_qa_data(self, num_samples: int):
        """Generates simple question-context pairs for Q&A."""
        # Simple generation: pair a random question with a random context
        data = []
        for _ in range(num_samples):
            question = np.random.choice(self.base_questions)
            context = np.random.choice(self.base_texts)
            data.append({"question": question, "context": context})
        return data

    def _analyze_metrics(self, metrics: dict, model_config: ModelConfig):
        """
        Calculates statistical analysis including mean, std, and confidence intervals.
        """
        analysis = {
            'model_name': model_config.name,
            'task': model_config.task,
            'num_runs': model_config.num_runs,
            'sample_size': len(metrics['latency']),
            'statistics': {}
        }

        for metric_name, values in metrics.items():
            data = np.array(values)
            analysis['statistics'][metric_name] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'cv': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else float('inf')
            }

            # Add confidence intervals (95% CI using t-distribution)
            confidence_level = 0.95
            degrees_freedom = len(data) - 1
            if degrees_freedom > 0:
                t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                margin_error = t_critical * (np.std(data) / np.sqrt(len(data)))
                analysis['statistics'][metric_name]['ci_lower'] = float(np.mean(data) - margin_error)
                analysis['statistics'][metric_name]['ci_upper'] = float(np.mean(data) + margin_error)
            else:
                # Handle case with insufficient data points for CI calculation
                analysis['statistics'][metric_name]['ci_lower'] = analysis['statistics'][metric_name]['mean']
                analysis['statistics'][metric_name]['ci_upper'] = analysis['statistics'][metric_name]['mean']

        return analysis

    def run_monte_carlo(self, model_config: ModelConfig):
        """
        Runs the Monte Carlo simulation for a given model configuration.
        """
        print(f"\n--- Starting benchmark for: {model_config.name} ({model_config.model}) ---")
        
        if model_config.task not in self.test_generators:
             print(f"Error: No data generator defined for task: {model_config.task}")
             return None

        test_data = self.test_generators[model_config.task](num_samples=model_config.max_samples)
        
        metrics = {'latency': []}
        
        try:
            # Load the model with device specification
            self.pipeline = pipeline(
                model_config.task, 
                model=model_config.model, 
                tokenizer=model_config.model,
                device=device  # FIX: Pass the device here to use GPU/CPU
            )
        except Exception as e:
            print(f"Failed to load model {model_config.model}. This might be a memory or access issue in Colab.")
            print(f"Error: {e}")
            return None

        # Monte Carlo Simulation Loop
        for i in range(model_config.num_runs):
            start_time = time.time()
            self.pipeline(test_data, truncation=True) # Run prediction
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000 / len(test_data)
            metrics['latency'].append(latency_ms)

            if (i + 1) % 10 == 0:
                print(f"Run {i+1}/{model_config.num_runs} complete.")

        # Calculate and return the analysis
        return self._analyze_metrics(metrics, model_config)
```

## Cell 3: Execution and Results
This cell sets up the models you want to test and executes the benchmark. (Note: I've chosen small, fast models to ensure they run successfully on a free Colab instance).
```python
# Initialize the benchmark framework
benchmark = ModelBenchmark(random_seed=123)

# Define models to test
model_configs = [
    ModelConfig(
        name="DistilBERT Sentiment",
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        num_runs=50, # Reduced runs for faster execution
    ),
    ModelConfig(
        name="BERT QA",
        task="question-answering",
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        num_runs=10
    )
]

# Run the benchmark for all models
results = []
for config in model_configs:
    result = benchmark.run_monte_carlo(config)
    if result:
        results.append(result)

# Print results
if results:
    print("\n\n=============== FINAL BENCHMARK RESULTS ===============")
    for result in results:
        print(f"\nModel: {result['model_name']} (Task: {result['task']})")
        latency_stats = result['statistics']['latency']
        print(f"  Latency (Mean): {latency_stats['mean']:.4f} ms/sample")
        print(f"  Latency (Std Dev): {latency_stats['std']:.4f} ms/sample")
        print(f"  95% CI: [{latency_stats['ci_lower']:.4f} ms, {latency_stats['ci_upper']:.4f} ms]")
        print(f"  CV: {latency_stats['cv']:.2f}")
```

## Example output by CPU
```
--- Starting benchmark for: DistilBERT Sentiment (distilbert-base-uncased-finetuned-sst-2-english) ---
config.json: 100%
 629/629 [00:00<00:00, 46.9kB/s]
model.safetensors: 100%
 268M/268M [00:01<00:00, 195MB/s]
tokenizer_config.json: 100%
 48.0/48.0 [00:00<00:00, 2.87kB/s]
vocab.txt: 100%
 232k/232k [00:00<00:00, 13.5MB/s]
Device set to use cpu
Run 10/50 complete.
Run 20/50 complete.
Run 30/50 complete.
Run 40/50 complete.
Run 50/50 complete.

--- Starting benchmark for: BERT QA (bert-large-uncased-whole-word-masking-finetuned-squad) ---
config.json: 100%
 443/443 [00:00<00:00, 28.9kB/s]
model.safetensors: 100%
 1.34G/1.34G [00:08<00:00, 268MB/s]
Some weights of the model checkpoint at bert-large-uncased-whole-word-masking-finetuned-squad were not used when initializing BertForQuestionAnswering: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
- This IS expected if you are initializing BertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
tokenizer_config.json: 100%
 48.0/48.0 [00:00<00:00, 2.23kB/s]
vocab.txt: 100%
 232k/232k [00:00<00:00, 12.4MB/s]
tokenizer.json: 100%
 466k/466k [00:00<00:00, 3.00MB/s]
Device set to use cpu
Run 10/10 complete.


=============== FINAL BENCHMARK RESULTS ===============

Model: DistilBERT Sentiment (Task: sentiment-analysis)
  Latency (Mean): 60.1179 ms/sample
  Latency (Std Dev): 21.0208 ms/sample
  95% CI: [54.1439 ms, 66.0920 ms]
  CV: 0.35

Model: BERT QA (Task: question-answering)
  Latency (Mean): 479.1641 ms/sample
  Latency (Std Dev): 5.6322 ms/sample
  95% CI: [475.1350 ms, 483.1931 ms]
  CV: 0.01
```
