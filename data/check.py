import pandas as pd
import os
import json
if __name__=="__main__":
    file_path = "data/DAPO-Math-17k-verl/train.parquet"
    df = pd.read_parquet(file_path)
    data = df.head(5)
    print(data)

    debug_path = "data/debug.json"
    data.to_json(debug_path, orient='records', indent=2, force_ascii=False)
    print(f"Debug data saved to: {debug_path}")