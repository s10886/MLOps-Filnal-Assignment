import kagglehub
import pandas as pd
from pathlib import Path

dataset_path = kagglehub.dataset_download("tawfikelmetwally/employee-dataset")
print(f"Dane pobrane do: {dataset_path}")

df = pd.read_csv(Path(dataset_path) / "Employee.csv")

#Save to raw folder 
output_path = Path("data/raw/Employee.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

#print(f"Saved locally to: {output_path}")