import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from data_loader import (load_matches,load_deliveries,load_merged,get_all_teams,get_all_venues)

#PAGE CONFIG

st.set_page_config(page_title="IPL Analytics Dashboard",page_icon="🏏",layout="wide",initial_sidebar_state="collapsed")

#LOAD DATA

matches=load_matches()
deliveries=load_deliveries()
merged=load_merged()

#TABS
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 IPL Overview","🏏 Team Intelligence","⚔️ Rivalry Dossier","🏟️ Venue Intelligence","🤖 Match Predictor","🚀 Impact Player Era"])


#TAB 1:IPL OVERVIEW

with tab1:
    st.title("📊 IPL Macro-Overview")
    st.markdown("*16 seasons of data • 2008-2024 • 1090 macthes*")

    #SECTION1: KPI CARDS:

    st.subheader("Tournament at a Glance")
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Total Matches",f"{len(matches):,}")
    c2.metric("Seasons",matches["season"].nunique())
    c3.metric("Teams",len(get_all_teams(matches)))
    c4.metric("Total Runs",f"{deliveries['total_runs'].sum():,}")
    c5.metric("Total Wickets",f"{deliveries['bowler_wicket'].sum():,}")

    #SECTION2:MATCHES PER SEASON + SIXES AND FOURS

    st.subheader("Seasonal Growth and Boundary Explosion")

    matches_per_season=matches.groupby("season").size()

    boundaries_per_season=merged.groupby("season").agg(sixes=("is_six","sum"),fours=("is_four","sum")).reset_index()
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4))

    #Matches per season
    seasons=matches_per_season.index.to_list()
    ax1.bar(seasons,matches_per_season.values,color="steelblue",alpha=0.8)
    ax1.set_title("Matches_per_Season")
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Number of Matches")
    ax1.tick_params(axis="x",rotation=45)


    #Sixes Per Season
    
    ax2.plot(boundaries_per_season["season"],boundaries_per_season["sixes"],marker="o",color="crimson",label="Sixes",linewidth=2)
    ax2.plot(boundaries_per_season["season"],boundaries_per_season["fours"],marker="s",color="orange",label="Fours",linewidth=2)
    ax2.set_title("Boundary Explosions Over Seasons")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x",rotation=45)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


    st.divider()

    #Section 3:Toss Decision Trends:

    st.subheader("Toss Stratergy Evolution")
    st.markdown("*How has the bat-vs-field decision changed over 16 seasons?*")

    toss_trends=matches.groupby(["season","toss_decision"]).size().unstack(fill_value=0)

    #Converting to percentage
    toss_pct=toss_trends.div(toss_trends.sum(axis=1),axis=0)*100

    fig2,ax=plt.subplots(figsize=(12,4))
    toss_pct.plot(kind="bar",stacked=True,ax=ax,color=["#2196F3","#FF9800"],alpha=0.85)
    ax.set_title("Toss Decision Trend by Season (%)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Percentage(%)")
    ax.tick_params(axis="x",rotation=45)
    ax.legend(["Bat First","Field First"],loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.divider()

    #SECTION 4: Batting First Win Rate:

    st.subheader("Batting First vs Chasing - Win Rate Over Seasons")
    dls_toggle=st.checkbox("Exclude DLS (Rain-Affected) Macthes", value=False)

    if dls_toggle:
        analysis_df=matches[matches["method"].isna()|(matches["method"]=="")].copy()
        st.caption("Showing standard matches only")
    else:
        analysis_df=matches.copy()
        st.caption("Showing All matches including DLS")

    batting_first_by_season=analysis_df.groupby("season").agg(batting_first_wins=("batting_first_won","sum"),total_matches=("batting_first_won","count")).reset_index()
    batting_first_by_season["batting_first_rate"]=(batting_first_by_season["batting_first_wins"]/batting_first_by_season["total_matches"]*100)
    batting_first_by_season["chasing_rate"]=(100-batting_first_by_season["batting_first_rate"])

    fig3,ax=plt.subplots(figsize=(12,4))
    ax.plot(batting_first_by_season["season"],batting_first_by_season["batting_first_rate"],marker="o",color="steelblue",label="Batting First Win %",lw=2)
    ax.plot(batting_first_by_season["season"],batting_first_by_season["chasing_rate"],marker="s",color="crimson",label="Chasing Win %",lw=2)
    ax.axhline(50,color="gray",linestyle="--",alpha=0.5,label="50% line")
    ax.fill_between(batting_first_by_season["season"],batting_first_by_season["batting_first_rate"],50,alpha=0.1,color="steelblue")
    ax.set_title("Batting First vs Chasing Win Rate by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Win Rate")
    ax.tick_params(axis="x",rotation=45)
    ax.legend()
    ax.set_ylim(20,80)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.divider()

    #SECTION 5: NAIL BITERS

    st.subheader("Nail-Biter Frequency")
    st.markdown("*Matches won by ≤6 runs, ≤3 wickets, or decided by Super Over*")
    nail_biters_by_season=matches.groupby("season").agg(nail_biters=("nail_biter","sum"),total=("nail_biter","count")).reset_index()
    nail_biters_by_season["nail_biter_pct"]=(nail_biters_by_season["nail_biters"]/nail_biters_by_season["total"]*100)
    fig4,ax=plt.subplots(figsize=(12,4))
    ax.bar(nail_biters_by_season["season"],nail_biters_by_season["nail_biter_pct"],color="purple",alpha=0.7)
    ax.set_title("% of Nail Biter Matches Per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Percentage of Matches (%)")
    ax.tick_params(axis="x",rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    #Expandable nail-biter details

    with st.expander("🔍 View All Nail-Biter Matches"):
        nail_biter_df=matches[matches["nail_biter"]==1][["season","date","team1","team2","winner","result","result_margin","venue"]].sort_values(by="date",ascending=False)
        st.dataframe(nail_biter_df,use_container_width=True)

    st.divider()

    # SECTION 6:Orange Cap and Purple Cap Trackers

    st.subheader("🧡 Orange Cap & 🟣 Purple Cap Season Leaders")

    #Merge season info to deliveries

    orange_cap=merged.groupby(["season","batter"])["batsman_runs"].sum()
    orange_cap=orange_cap.reset_index()
    orange_cap=orange_cap.loc[orange_cap.groupby("season")["batsman_runs"].idxmax()].sort_values("season")
    orange_cap.columns=["Season","Orange Cap Winner","Runs"]

    purple_cap=merged[merged["bowler_wicket"]==1].groupby(["season","bowler"])["bowler_wicket"].sum()
    purple_cap=purple_cap.reset_index()
    purple_cap=purple_cap.loc[purple_cap.groupby("season")["bowler_wicket"].idxmax()].sort_values("season")
    purple_cap.columns=["Season","Purple Cap Winner","Wickets"]

    cap_col1,cap_col2=st.columns(2)
    with cap_col1:
        st.markdown("**🧡 Orange Cap — Top Run Scorer Each Season**")
        st.dataframe(orange_cap.set_index("Season"),use_container_width=True)
    with cap_col2:
        st.markdown("**🟣 Purple Cap — Top Wicket Taker Each Season**")
        st.dataframe(purple_cap.set_index("Season"),use_container_width=True)

    st.divider()

    #SECTION 7: Consistency Index:
    st.subheader("🏆 Franchise Consistency Index")

    st.markdown("*Number of seasons each team reached the playoffs (post-2011 format)*") 

    modern=matches[matches["season"].astype(int)>=2011]
    playoffs=modern[modern["match_type"].isin(["Qualifier 1","Qualifier 2","Eliminator","Final"])]

    playoff_teams=[]
    for season,group in playoffs.groupby("season"):
        teams=pd.concat([group["team1"],group["team2"]]).unique()
        for team in teams:
            playoff_teams.append({"Team":team,"Season":season})

    playoff_df=pd.DataFrame(playoff_teams)
    consistency=playoff_df.groupby("Team")["Season"].nunique().sort_values()

    fig5,ax=plt.subplots(figsize=(10,6))
    colors=["#1a237e" if v >= 8 else "#1565c0" if v >= 6 else "#42a5f5" if v >= 4 else "#90caf9" for v in consistency.values]
    ax.barh(consistency.index,consistency.values,color=colors)
    ax.set_title("Most Consistent Teams in Modern IPL Era (2011–2024)")
    ax.set_xlabel("PlayOff Appearences")
    ax.axvline(x=consistency.mean(),color="red",linestyle="--",alpha=0.7,label="Average")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    #TAB2: TEAM INTELLIGENCE:

    with tab2:
        st.title("🏏 Team Intelligence")
        st.markdown("*Select a franchise to explore its complete IPL History*")

        #Team Selector
        all_teams=get_all_teams(matches)
        selected_team=st.selectbox("Select a Team",all_teams,index=all_teams.index("Chennai Super Kings"))

        #Filter data for selected team

        team_matches=matches[(matches["team1"]==selected_team)|(matches["team2"]==selected_team)].copy()
        team_wins=team_matches[team_matches["winner"]==selected_team]

        #Section 1: Health Check KPI's

        st.subheader(f"📋{selected_team} - Franchise Health Check")

        total_matches=len(team_matches)
        total_wins=len(team_wins)
        win_rate=total_wins/total_matches *100

        toss_won=team_matches[team_matches["toss_winner"]==selected_team]
        toss_win_rate=len(toss_won)/total_matches *100

        toss_and_match_won=toss_won[toss_won["winner"]==selected_team]
        toss_advantage=len(toss_and_match_won)/len(toss_won) *100 if len(toss_won)>0 else 0

        k1,k2,k3,k4,k5=st.columns(5)
        k1.metric("Total Matches",total_matches)
        k2.metric("Total Wins",total_wins)
        k3.metric("Win Rate",f"{win_rate:.1f}%",delta=f"{win_rate-50:.1f}% vs 50%")
        k4.metric("Toss Win Rate",f"{toss_win_rate:.1f}%")
        k5.metric("Win Rate when won Toss",f"{toss_advantage:.1f}%")

        st.divider()

        #SECTION 2:SEASONAL WIN TRAJECTORY:

        st.subheader("📈 Seasonal Win Trajectory")

        season_stats=team_matches.groupby("season").apply(lambda x:pd.Series({"matches":len(x),"wins":(x["winner"]==selected_team).sum(),})).reset_index()
        season_stats["win_rate"]=season_stats["wins"]/season_stats["matches"] *100
        
        fig,ax=plt.subplots(figsize=(12,4))

        ax.plot(season_stats["season"],season_stats["win_rate"],marker="o",color="green",linewidth=2.5,markersize=7)
        ax.fill_between(season_stats["season"],season_stats["win_rate"],50,where=(season_stats["win_rate"]>=50),alpha=0.15,color="green",label="Above 50%")
        ax.fill_between(season_stats["season"],season_stats["win_rate"],50,where=(season_stats["win_rate"]<50),alpha=0.15,color="red",label="Below 50%")
        ax.axhline(50,color="gray",linestyle="--",alpha=0.7,label="50% benchmark")

        #Highlight Peak Season
        peak_idx=season_stats["win_rate"].idxmax()
        peak_season=season_stats.loc[peak_idx,"season"]
        peak_rate=season_stats.loc[peak_idx,"win_rate"]
        ax.annotate(f"Peak: {peak_season}\n({peak_rate:.0f}%)",xy=(peak_season,peak_rate),xytext=(peak_season,peak_rate + 8),arrowprops=dict(arrowstyle="->",color="darkgreen"),fontsize=9,color="darkgreen",ha="center")
        ax.set_xlabel("Season")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title(f"{selected_team} — Win Rate by Season")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 110)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        #Peak Era Banner
        st.success(f"🏆 Peak Performance Era: **{peak_season}** season with **{peak_rate:.1f}%** win rate")

        st.divider()

        #Section 3:HOME VS AWAY

        st.subheader("🏠 Home vs Away Performance")

        from data_loader import HOME_CITY_MAP
        def check_home(row,team):
            if team not in HOME_CITY_MAP:
                return "Neutral"

            keywords=HOME_CITY_MAP[team]
            venue=str(row["venue"])
            if any(k.lower() in venue.lower() for k in keywords):
                return "Home"
            return "Away"
        
        team_matches["venue_type"]=team_matches.apply(lambda row:check_home(row,selected_team), axis=1)

        venue_stats=team_matches.groupby("venue_type").apply(lambda x:pd.Series({"matches":len(x),"wins":(x["winner"]==selected_team).sum(),"win_rate":(x["winner"]==selected_team).mean()*100})).reset_index()

        col1,col2=st.columns(2)

        with col1:
            fig,ax=plt.subplots(figsize=(6,4))
            colors = {"Home": "green", "Away": "crimson", "Neutral": "gray"}
            bar_colors = [colors.get(v, "gray") for v in venue_stats["venue_type"]]
            ax.bar(venue_stats["venue_type"],venue_stats["win_rate"],color=bar_colors,alpha=0.8)
            ax.axhline(50,color="black",linestyle="--",alpha=0.5)
            ax.set_title("Win Rate: Home vs Away")
            ax.set_ylabel("Win Rate (%)")
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.dataframe(venue_stats.set_index("venue_type"),use_container_width=True)

        
        st.divider()

        #Section 4: Nemesis and Bunnies

        st.subheader("⚔️ Nemesis & Bunny — Win Rate vs Every Opponent")
        opponent_stats=[]

        for opp in get_all_teams(matches):
            if opp==selected_team:
                continue
            h2h=matches[((matches["team1"]==selected_team)&(matches["team2"]==opp))|((matches["team1"]==opp)&(matches["team2"]==selected_team))]
            if len(h2h)<3:
                continue
            wins=(h2h["winner"]==selected_team).sum()
            opponent_stats.append({"Opponent":opp,"Matches":len(h2h),"Wins":wins,"Win Rate":wins/len(h2h)*100})

        opp_df=pd.DataFrame(opponent_stats).sort_values("Win Rate")

        fig,ax=plt.subplots(figsize=(10,5))
        bar_colors=["crimson" if w < 40 else "orange" if w < 55 else "green" for w in opp_df["Win Rate"]]
        ax.barh(opp_df["Opponent"],opp_df["Win Rate"],color=bar_colors,alpha=0.8)
        ax.axvline(50,color="black",linestyle="--",alpha=0.5)
        ax.set_title(f"{selected_team} — Win Rate vs Each Opponent")
        ax.set_xlabel("Win Rate (%)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        #Label Nemesis and Bunny

        nemesis=opp_df.iloc[0]
        bunny = opp_df.iloc[-1]
        

        col1,col2=st.columns(2)

        col1.error(f"😈 **Nemesis:** {nemesis['Opponent']} — {nemesis['Win Rate']:.1f}% win rate")
        col2.success(f"🐰 **Bunny:** {bunny['Opponent']} — {bunny['Win Rate']:.1f}% win rate")

        st.divider()


        #section 5:Hall Of Fame

        st.subheader("🌟 Franchise Hall of Fame")

        #Get deliveries for this team
        team_batting=merged[merged["batting_team"]==selected_team]
        team_bowling=merged[merged["bowling_team"]==selected_team]

        top_batters=team_batting.groupby("batter")["batsman_runs"].sum()\
                                .sort_values(ascending=False).head(5)\
                                .reset_index()
        top_batters.columns=["Player","Runs"]

        top_bowlers=team_bowling[team_bowling["bowler_wicket"]==1]\
                                .groupby("bowler")["bowler_wicket"].sum()\
                                .sort_values(ascending=False).head(5)\
                                .reset_index()
        
        top_bowlers.columns=["Player","Wickets"]

        hof_col1,hof_col2=st.columns(2)

        with hof_col1:
            st.markdown(f"**🏏 Top 5 Run Scorers for {selected_team}**")
            st.dataframe(top_batters.set_index("Player"),use_container_width=True)
        with hof_col2:
            st.markdown(f"**🎳 Top 5 Wicket Takers for {selected_team}**")
            st.dataframe(top_bowlers.set_index("Player"), use_container_width=True)

        st.divider()

        #SECTION-6 RECORD VICTORIES AND DEFEATS

        st.subheader("🏆 Record Victories & Historic Defeats")

        wins_by_runs=team_wins[team_wins["result"]=="runs"].sort_values("result_margin",ascending=False).head(3)
        wins_by_wickets=team_wins[team_wins["result"]=="wickets"].sort_values("result_margin",ascending=False).head(3)

        defeats=team_matches[(team_matches["winner"]!=selected_team)&(team_matches["winner"].notna())]
        defeats_by_runs=defeats[defeats["result"]=="runs"].sort_values("result_margin",ascending=False).head(3)
        defeats_by_wickets = defeats[defeats["result"] == "wickets"].sort_values("result_margin",ascending=False).head(3)


        cols=st.columns(4)

        with cols[0]:
            st.markdown("**🏆 Biggest Wins (by runs)**")
            st.dataframe(wins_by_runs[["season","winner","team1","team2","result_margin"]].set_index("season"),use_container_width=True)

        with cols[1]:
            st.markdown("**🏆 Biggest Wins (by wickets)**")
            st.dataframe(wins_by_wickets[["season","winner","team1","team2","result_margin"]].set_index("season"),use_container_width=True)
        
        with cols[2]:
            st.markdown("**💔 Heaviest Defeats (by runs)**")
            st.dataframe(defeats_by_runs[["season","winner","team1","team2","result_margin"]].set_index("season"),use_container_width=True)

        with cols[3]:
            st.markdown("**💔 Heaviest Defeats (by wickets)**")
            st.dataframe(defeats_by_wickets[["season","winner","team1","team2","result_margin"]].set_index("season"),use_container_width=True)

        # TAB 3: RIVALRY DOSSIER (HEAD TO HEAD)
        # =======================================
    
    with tab3:
        st.title("⚔️ Rivalry Dossier")
        st.markdown("*Select two teams to uncover their complete head-to-head history*")
        
        all_teams=get_all_teams(matches)

        col_a,col_b=st.columns(2)

        with col_a:
            
            team1_select=st.selectbox("Team1",all_teams,index=all_teams.index("Mumbai Indians"),key="t1")
        
        with col_b:

            team2_options=[t for t in all_teams if t!=team1_select]
            team2_select=st.selectbox("Team2",team2_options,index=team2_options.index("Chennai Super Kings") if "Chennai Super Kings" in team2_options else 0,key="t2")

        #Filter H2H Matches

        h2h=matches[((matches["team1"]==team1_select)&(matches["team2"]==team2_select))|((matches["team1"]==team2_select)&(matches["team2"]==team1_select))].copy().sort_values("date")
        if(len(h2h)==0):
            st.warning(f"{team1_select} and {team2_select} have never played each other.")
        else:
            #Section1:SUPREMACY SPLIT
            st.subheader("👑 Head-to-Head Supremacy")

            team1_wins=(h2h["winner"]==team1_select).sum()
            team2_wins=(h2h["winner"]==team2_select).sum()
            total_played=len(h2h)

            c1,c2,c3=st.columns(3)
            c1.metric(f"{team1_select} Wins", team1_wins)
            c2.metric("Total Matches", total_played)
            c3.metric(f"{team2_select} Wins", team2_wins)

            #Visual Split bar
            fig,ax=plt.subplots(figsize=(10,1.5))
            ax.barh([0],[team1_wins],color="steelblue",label=team1_select)
            ax.barh([0],[team2_wins],left=[team1_wins],color="crimson",label=team2_select)
            ax.set_xlim(0, total_played)
            ax.set_yticks([])
            ax.set_xlabel("Matches Won")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            nail_biters_h2h=h2h["nail_biter"].sum()
            st.info(f"🔥 **{nail_biters_h2h}** of their **{total_played}** encounters "
                     f"were nail-biters ({nail_biters_h2h/total_played*100:.0f}%)")
            
            st.divider()


            #Section2:Win Rate Trends Over Seasons

            st.subheader("📈 Pendulum of Dominance")

            h2h_season_stats=h2h.groupby("season").apply(lambda x:pd.Series({"matches":len(x),f"{team1_select}_wins":(x["winner"]==team1_select).sum()})).reset_index()

            h2h_season_stats["team1_win_rate"]=(h2h_season_stats[f"{team1_select}_wins"]/h2h_season_stats["matches"]*100)

            fig,ax=plt.subplots(figsize=(12,4))
            ax.plot(h2h_season_stats["season"],h2h_season_stats["team1_win_rate"],marker="o",color="steelblue",linewidth=2,label=f"{team1_select} win rate")
            ax.axhline(50,color="gray",linestyle="--",alpha=0.6,label="50% line")
            ax.fill_between(h2h_season_stats["season"],h2h_season_stats["team1_win_rate"],50,where=(h2h_season_stats["team1_win_rate"]>=50),alpha=0.15,color="steelblue")
            ax.fill_between(h2h_season_stats["season"],h2h_season_stats["team1_win_rate"],50,where=(h2h_season_stats["team1_win_rate"]<50),alpha=0.15,color="crimson")
            ax.set_xlabel("Season")
            ax.set_ylabel(f"{team1_select} Win Rate (%)")
            ax.set_title(f"{team1_select} vs {team2_select} — Win Rate Trend")
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(0, 100)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
            st.divider()

            #SECTION3 : VENUE BREAKDOWN:

            st.subheader("🏟️ Ground Advantage Checklist")

            venue_h2h=h2h.groupby("venue").apply(lambda x:pd.Series({"matches":len(x),f"{team1_select}_wins":(x["winner"]==team1_select).sum(),f"{team2_select}_wins":(x["winner"]==team2_select).sum()})).reset_index()
            venue_h2h=venue_h2h[venue_h2h["matches"]>=2].sort_values("matches",ascending=True)

            if len(venue_h2h)>0:
                fig,ax=plt.subplots(figsize=(10,max(3,len(venue_h2h)*0.4)))
                y_pos=np.arange(len(venue_h2h))
                ax.barh(y_pos,venue_h2h[f"{team1_select}_wins"],color="steelblue",label=team1_select)
                ax.barh(y_pos,venue_h2h[f"{team2_select}_wins"],left=venue_h2h[f"{team1_select}_wins"],color="crimson",label=team2_select)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(venue_h2h["venue"], fontsize=8)
                ax.set_xlabel("Matches Won")
                ax.set_title("Venue-wise Head-to-Head Record (min 2 matches)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Not enough venue data (need 2+ matches at same venue)")
            
            st.divider()

            #SECTION 4:Batting First VS  Chasing

            st.subheader("🎯 The Coin Toss Factor")
            bat_first_h2h=h2h.groupby(h2h.apply(lambda r:"Batted First Won" if r["batting_first_won"]==1 else "Chasing Team Won",axis=1)).size()
            fig,ax=plt.subplots(figsize=(8,3.5))
            ax.bar(bat_first_h2h.index,bat_first_h2h.values,color=["steelblue","orange"],alpha=0.8)
            ax.set_ylabel("Number of Matches")
            ax.set_title(f"Batting First vs Chasing — {team1_select} vs {team2_select}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
            st.divider()

            #SECTION 5:MVP MATRIX

            st.subheader("🌟 Rivalry MVP Matrix")
            st.markdown(f"*Top performers when {team1_select} and {team2_select} clash*")

            h2h_match_ids=h2h["id"].tolist()
            h2h_deliveries=merged[merged["match_id"].isin(h2h_match_ids)]


            #Top batters in this rivalry:

            rivalry_batters=h2h_deliveries[h2h_deliveries["extras_type"]!="wides"].groupby("batter").agg(runs=("batsman_runs","sum"),balls=("batsman_runs","count")).reset_index()
            rivalry_batters["strike_rate"]=(rivalry_batters["runs"]/rivalry_batters["balls"]*100)
            rivalry_batters=rivalry_batters[rivalry_batters["balls"]>=20].sort_values("runs",ascending=False).head(5)

            #Top Bowlers in this rivalry:

            h2h_deliveries["bowler_runs"] = h2h_deliveries["total_runs"] - h2h_deliveries["extra_runs"].where(h2h_deliveries["extras_type"].isin(["byes","legbyes"]), 0)
            rivalry_bowlers=h2h_deliveries.groupby("bowler").agg(wickets=("bowler_wicket","sum"),runs_conceded=("bowler_runs","sum"),balls_bowled=("extras_type", lambda x: (~x.isin(["wides", "noballs"])).sum())).reset_index()
            rivalry_bowlers["economy"]=(rivalry_bowlers["runs_conceded"]/(rivalry_bowlers["balls_bowled"]/6))
            rivalry_bowlers=rivalry_bowlers[rivalry_bowlers["balls_bowled"]>=24].sort_values("wickets",ascending=False).head(5)


            mvp_col1,mvp_col2=st.columns(2)

            with mvp_col1:
                st.markdown("**🏏 Top Run Scorers in this Rivalry**")
                display_batters=rivalry_batters[["batter","runs","strike_rate"]].copy()
                display_batters["strike_rate"]=display_batters["strike_rate"].round(1)
                display_batters.columns=["Player","Runs","Strike Rate"]
                st.dataframe(display_batters.set_index("Player"),use_container_width=True)

            with mvp_col2:
                st.markdown("**🎳 Top Wicket Takers in this Rivalry**")
                display_bowlers=rivalry_bowlers[["bowler","wickets","economy"]].copy()
                display_bowlers["economy"]=display_bowlers["economy"].round(2)
                display_bowlers.columns=["Player","Wickets","Economy"]
                st.dataframe(display_bowlers.set_index("Player"),use_container_width=True)

            st.divider()

            #SECTION 6:SCORE DISTRIBUTION

            st.subheader("📊 Historical Innings Score Distribution")

            standard_h2h = h2h[h2h["method"].isna() | (h2h["method"] == "")]
            standard_match_ids = standard_h2h["id"].tolist()
            h2h_deliveries_standard = merged[(merged["match_id"].isin(standard_match_ids)) & (merged["inning"].isin([1, 2]))]




            innings_scores=h2h_deliveries_standard.groupby(["match_id","inning"])["total_runs"].sum().reset_index()

            fig,ax=plt.subplots(figsize=(10,4))
            first_innings = innings_scores[innings_scores["inning"]==1]["total_runs"]
            second_innings = innings_scores[innings_scores["inning"]==2]["total_runs"]
            bp=ax.boxplot([first_innings, second_innings], vert=False, patch_artist=True,labels=["1st Innings", "2nd Innings"])
            colors = ["steelblue", "orange"]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xlabel("Innings Total Score")
            ax.set_title(f"Score Distribution by Innings — {team1_select} vs {team2_select}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(f"Average 1st innings: **{first_innings.mean():.0f}** | "f"Average 2nd innings: **{second_innings.mean():.0f}**")







