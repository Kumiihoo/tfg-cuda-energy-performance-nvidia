import pandas as pd

def read_power(path):
    df = pd.read_csv(path, header=None,
                     names=["ts","pstate","power","sm_clock","mem_clock","temp"])
    df["power"] = df["power"].str.replace(" W","", regex=False).astype(float)
    df["sm_clock"] = df["sm_clock"].str.replace(" MHz","", regex=False).astype(float)
    df["mem_clock"] = df["mem_clock"].str.replace(" MHz","", regex=False).astype(float)
    df["temp"] = df["temp"].astype(float)
    return df

def avg_power(df): return df["power"].mean()
def mode_pstate(df): return df["pstate"].mode().iloc[0] if not df["pstate"].mode().empty else ""
def avg_sm(df): return df["sm_clock"].mean()

bw = pd.read_csv("results/bw.csv")
bw_plateau = bw["GBs"].iloc[-1]

comp = pd.read_csv("results/compute.csv")
fp32_peak = comp[comp["dtype"]=="fp32"]["GFLOPs"].max()
fp64_peak = comp[comp["dtype"]=="fp64"]["GFLOPs"].max()

gemm = pd.read_csv("results/gemm.csv")
g0 = gemm[gemm["tf32"]==0]["GFLOPs"].max()
g1 = gemm[gemm["tf32"]==1]["GFLOPs"].max()

p_gemm = read_power("results/energy/power_gemm.csv")
pg = avg_power(p_gemm)

out = pd.DataFrame([
    ["GEMM TF32=0 (max)", f"{g0:.2f} GFLOP/s", f"{pg:.2f} W", f"{(g0/pg):.4f} GFLOP/s/W", mode_pstate(p_gemm), f"{avg_sm(p_gemm):.0f} MHz"],
    ["GEMM TF32=1 (max)", f"{g1:.2f} GFLOP/s", f"{pg:.2f} W", f"{(g1/pg):.4f} GFLOP/s/W", mode_pstate(p_gemm), f"{avg_sm(p_gemm):.0f} MHz"],
], columns=["Test","Performance","Avg Power","Efficiency","Pstate(mode)","Avg SM clock"])

print(out.to_string(index=False))
out.to_csv("results/energy/efficiency_summary_gemm.csv", index=False)
print("\nWrote results/energy/efficiency_summary_gemm.csv")
