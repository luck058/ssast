import pandas as pd

# Load the official CSV
df = pd.read_csv("balanced_train_segments.csv", comment='#', skip_blank_lines=True)

# Take just the first 20 rows
df_small = df.head(20) 

# Save it as your new test file
df_small.to_csv("my_test_file.csv", index=False)
