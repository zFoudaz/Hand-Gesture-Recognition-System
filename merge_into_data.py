import pandas as pd
# new files to be merged into data.csv -> new_files
new_files = [

]

main_file = "data.csv"

print("🔍 Loading main file...")
df_main = pd.read_csv(main_file)
print("✔ Main file loaded:", len(df_main), "rows")

all_new = []

for file in new_files:
    print(f"\n🔍 Loading {file} ...")
    df_new = pd.read_csv(file)
    print(f"   ✔ Loaded {len(df_new)} rows")

    all_new.append(df_new)

print("\n🔄 Merging all files together...")

df_merged = pd.concat([df_main] + all_new, ignore_index=True)

df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

df_merged.to_csv("data.csv", index=False)

print("\n✅ DONE! Merged successfully.")
print("📦 New data.csv size:", len(df_merged), "rows")
