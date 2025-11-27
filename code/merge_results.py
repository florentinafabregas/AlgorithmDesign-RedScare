import pandas as pd
from functools import reduce

# base df with Instance name + n
many = pd.read_csv("../results/many_results.csv")
base_df = many.rename(columns={"instance": "Instance name"})[["Instance name", "n"]]

# Load and process the individual results
alt = pd.read_csv("../results/alternate_results.csv")
A_df = alt.rename(columns={"instance": "Instance name",
                           "answer": "A"})[["Instance name", "A"]]

few = pd.read_csv("../results/few_results.csv")
few["Instance name"] = few["filename"].str.replace(r"\.txt$", "", regex=True)
F_df = few.rename(columns={"result": "F"})[["Instance name", "F"]]

M_df = many.rename(columns={"instance": "Instance name"})[["Instance name", "M"]]

none = pd.read_csv("../results/none_results.csv")
none["Instance name"] = none["file"]
N_df = none.rename(columns={"answer": "N"})[["Instance name", "N"]]

some = pd.read_csv("../results/some_results.csv")
some["Instance name"] = some["file"].str.replace(r"\.txt$", "", regex=True)
S_df = some.rename(columns={"answer": "S"})[["Instance name", "S"]]


dfs = [base_df, A_df, F_df, M_df, N_df, S_df]

# merge the results into a single data frame
merged = reduce(
    lambda left, right: pd.merge(left, right, on="Instance name", how="outer"),
    dfs
)

df = merged.copy()

# thousands separator in n
df["n"] = df["n"].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "")

df = df[["Instance name", "n", "A", "F", "M", "N", "S"]]

# build fixed-width text table
headers = list(df.columns)

widths = []
for col in headers:
    max_len = max(len(str(val)) for val in df[col])
    widths.append(max(max_len, len(col)))

def fmt_row(row_vals):
    cells = []
    for i, (val, w) in enumerate(zip(row_vals, widths)):
        if i == 0:
            cells.append(f"{str(val):<{w}}")
        else:
            cells.append(f"{str(val):>{w}}")
    return "  ".join(cells)

header_line = fmt_row(headers)
rule = "-" * len(header_line)

lines = []
lines.append(rule)
lines.append(header_line)
lines.append(rule)

# data rows
for _, row in df.iterrows():
    lines.append(fmt_row(row.tolist()))

lines.append(rule)

# write to a text file
with open("../results/results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
