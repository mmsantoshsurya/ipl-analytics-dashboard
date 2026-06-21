import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# CONSTANTS
# ============================================================

TEAM_CLEANING = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors",
}

HOME_CITY_MAP = {
    "Mumbai Indians": ["Mumbai", "Wankhede"],
    "Kolkata Knight Riders": ["Kolkata", "Eden Gardens"],
    "Chennai Super Kings": ["Chennai", "Chepauk", "Chidambaram"],
    "Rajasthan Royals": ["Jaipur", "Sawai"],
    "Royal Challengers Bengaluru": ["Bengaluru", "Bangalore", "Chinnaswamy"],
    "Sunrisers Hyderabad": ["Hyderabad", "Uppal", "Rajiv Gandhi"],
    "Delhi Capitals": ["Delhi", "Kotla", "Feroz Shah"],
    "Punjab Kings": ["Mohali", "Punjab", "PCA"],
    "Gujarat Titans": ["Ahmedabad", "Narendra Modi"],
    "Lucknow Super Giants": ["Lucknow", "Ekana"],
}
VENUE_CLEANING = {
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi":"Arun Jaitley Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
    "Saurashtra Cricket Association Stadium": "Niranjan Shah Stadium", 
    "OUTsurance Oval": "Mangaung Oval",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Ekana Cricket Stadium",
    "Vidarbha Cricket Association Stadium, Jamtha": "Vidarbha Cricket Association Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium"
}
BOWLER_WICKET_TYPES = [
    "caught", "bowled", "lbw", "stumped",
    "caught and bowled", "hit wicket"
]

# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

@st.cache_data
def load_matches():
    """Load and clean matches.csv"""
    df = pd.read_csv("data/matches.csv")
    
    # Clean team names
    for col in ["team1", "team2", "toss_winner", "winner"]:
        df[col] = df[col].replace(TEAM_CLEANING)
    
    # Parse dates and extract season
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["season"].astype(str)

    # Normalize season format — convert 2007/08 to 2008, 2009/10 to 2010 etc
    def normalize_season(s):
        s = str(s)
        if "/" in s:
            return "20" + s.split("/")[1]
        return s

    df["season"] = df["season"].apply(normalize_season)
    
    # Clean city names
    df["city"] = df["city"].replace({"Bangalore": "Bengaluru"})


    #Venue Cleaning
    df["venue"] = df["venue"].replace(VENUE_CLEANING)
    
    # Remove no-result matches for analysis
    # Keep them in a separate variable if needed
    df_valid = df[~df["winner"].isin(["No result", ""])].copy()
    df_valid = df_valid.dropna(subset=["winner"])
    
    # Add useful derived columns
    df_valid["toss_winner_won"] = (
        df_valid["toss_winner"] == df_valid["winner"]).astype(int)
    df_valid["batting_first_won"] = (
        df_valid["toss_decision"].apply(
            lambda x: 1 if x == "bat" else 0) == 
        df_valid["toss_winner_won"]).astype(int)
    
    # Era classification
    df_valid["era"] = df_valid["season"].apply(
        lambda x: "Impact Player Era (2023+)" 
        if x in ["2023", "2024"] else "Traditional Era (Pre-2023)")
    
    # Nail biter flag
    close_runs = (df_valid["result"] == "runs") & (df_valid["result_margin"] <= 6)
    close_wickets = (df_valid["result"] == "wickets") & (df_valid["result_margin"] <= 3)
    super_over = (df_valid["super_over"] == "Y")
    df_valid["nail_biter"] = (close_runs | close_wickets | super_over).astype(int)

    df_valid["team1_won"] = (df_valid["winner"] == df_valid["team1"]).astype(int)

    
    
    return df_valid

@st.cache_data
def load_deliveries():
    """Load and clean deliveries.csv"""
    df = pd.read_csv("data/deliveries.csv")
    
    # Clean team names in deliveries
    for col in ["batting_team", "bowling_team"]:
        df[col] = df[col].replace(TEAM_CLEANING)
    
    # Phase classification (0-indexed overs)
    df["phase"] = pd.cut(
        df["over"],
        bins=[-1, 5, 14, 19],
        labels=["Powerplay", "Middle", "Death"]
    )
    
    # Bowler wicket flag (excludes run outs, retired hurt)
    df["bowler_wicket"] = (
        (df["is_wicket"] == 1) & 
        (df["dismissal_kind"].isin(BOWLER_WICKET_TYPES))
    ).astype(int)
    
    # Boundary flag
    df["is_boundary"] = df["batsman_runs"].isin([4, 6]).astype(int)
    df["is_six"] = (df["batsman_runs"] == 6).astype(int)
    df["is_four"] = (df["batsman_runs"] == 4).astype(int)
    df["bowler_runs"] = df["total_runs"] - df["extra_runs"].where(df["extras_type"].isin(["byes","legbyes"]), 0)
    df["any_wicket"] = df["is_wicket"]
    
    return df

@st.cache_data
def load_merged():
    """Merge matches and deliveries on match_id"""
    matches = load_matches()
    deliveries = load_deliveries()
    
    # matches uses 'id', deliveries uses 'match_id'
    merged = deliveries.merge(
        matches[["id", "season", "venue", "city", "era",
                 "team1", "team2", "winner", "method"]],
        left_on="match_id",
        right_on="id",
        how="left"
    )
    return merged

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_team_matches(matches_df, team):
    """Get all matches involving a specific team"""
    return matches_df[
        (matches_df["team1"] == team) | 
        (matches_df["team2"] == team)
    ].copy()

def get_h2h_matches(matches_df, team1, team2):
    """Get all matches between two specific teams"""
    return matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ].copy()

def get_venue_matches(matches_df, venue):
    """Get all matches at a specific venue"""
    return matches_df[matches_df["venue"] == venue].copy()

def get_team_deliveries(deliveries_df, team, role="batting"):
    """Get deliveries where team was batting or bowling"""
    col = "batting_team" if role == "batting" else "bowling_team"
    return deliveries_df[deliveries_df[col] == team].copy()

def is_home_game(row):
    """Check if team1 is playing at home"""
    team = row["team1"]
    venue = str(row["venue"])
    if team in HOME_CITY_MAP:
        return any(city.lower() in venue.lower() 
                   for city in HOME_CITY_MAP[team])
    return False

def get_all_teams(matches_df):
    """Get sorted list of all unique teams"""
    teams = pd.concat([matches_df["team1"], 
                       matches_df["team2"]]).unique()
    return sorted(teams)

def get_all_venues(matches_df):
    """Get sorted list of all unique venues"""
    return sorted(matches_df["venue"].dropna().unique())