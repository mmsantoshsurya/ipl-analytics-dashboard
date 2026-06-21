
from data_loader import *

matches = load_matches()
deliveries = load_deliveries()
merged = load_merged()

print("Matches shape:", matches.shape)
print("Deliveries shape:", deliveries.shape)
print("Merged shape:", merged.shape)
print("Seasons:", sorted(matches["season"].unique()))
print("Teams:", get_all_teams(matches))
print("Nail biters:", matches["nail_biter"].sum())
print("Era counts:", matches["era"].value_counts())
print("Phase counts:", deliveries["phase"].value_counts())
print("Bowler wickets:", deliveries["bowler_wicket"].sum())

# Sanity checks
print("\nTop 5 scorers:", 
    deliveries.groupby("batter")["batsman_runs"]
    .sum().sort_values(ascending=False).head())
venues = sorted(matches["venue"].unique())
for v in venues:
    print(v)
    