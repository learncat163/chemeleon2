from monty.serialization import loadfn
from src.utils.metrics import Metrics

# Load generated structures from previous step
gen_structures = loadfn("outputs/alex-mp-20/generated_structures.json.gz")

# Initialize metrics calculator
metrics = Metrics(
    metrics=["unique", "novel", "e_above_hull"],
    reference_dataset="mp-20",
    phase_diagram="mp-all",
    metastable_threshold=0.1,
)

# Compute metrics
results = metrics.compute(gen_structures=gen_structures)

# Calculate mSUN score (percentage that are unique, novel, AND metastable)
msun_score = (
    results["unique"] & results["novel"] & results["is_metastable"]
).mean() * 100
print(f"mSUN Score: {msun_score:.2f}%")
print(f"Uniqueness: {results['unique'].mean():.2%}")
print(f"Novelty: {results['novel'].mean():.2%}")
print(f"Metastable: {results['is_metastable'].mean():.2%}")

# Convert to DataFrame for analysis
df = metrics.to_dataframe()
df.head()

# Save results to CSV
df.to_csv("outputs/alex-mp-20/metrics_results.csv")