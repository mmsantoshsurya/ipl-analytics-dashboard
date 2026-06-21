import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from data_loader import load_matches, HOME_CITY_MAP

def is_home(team,venue):
    if team not in HOME_CITY_MAP:
        return 0
    keywords=HOME_CITY_MAP[team]
    venue=str(venue)
    return int(any(k.lower() in venue.lower() for k in keywords))

def get_h2h_winrate(data):
    h2h=[]
    for idx,row in data.iterrows():
        t1,t2=row["team1"],row["team2"]
        past=data.iloc[:idx]
        past_matches=past[((past["team1"]==t1)&(past["team2"]==t2))|((past["team1"]==t2)&(past["team2"]==t1))]
        if len(past_matches)==0:
            h2h.append(0.5)
        else:
            t1_wins=past_matches[past_matches["winner"]==t1].shape[0]
            h2h.append(t1_wins/len(past_matches))
    
    return h2h
def get_recent_form(data,team_col,n=5):
    form=[]
    for idx,row in data.iterrows():
        team=row[team_col]
        past=data.iloc[:idx]
        past_team=past[(past["team1"]==team)|(past["team2"]==team)].tail(n)
        if len(past_team)==0:
            form.append(0.5)
        else:
            wins=past_team[past_team["winner"]==team].shape[0]
            form.append(wins/len(past_team))
    return form

def train_model():
    data=load_matches()
    data=data.sort_values("date").reset_index(drop=True)

    print("Building features....")
    data["h2h_winrate"]=get_h2h_winrate(data)
    data["team1_form"]=get_recent_form(data,"team1")
    data["team2_form"]=get_recent_form(data,"team2")
    data["team1_is_home"]=data.apply(lambda r:is_home(r["team1"],r["venue"]),axis=1)

    data["toss_winner_is_team1"]=(data["toss_winner"]==data["team1"]).astype(int)
    data["chose_bat"]=(data["toss_decision"]=="bat").astype(int)

    #Consistent team encoding across team1/team2

    le_team=LabelEncoder()
    all_teams=pd.concat([data["team1"],data["team2"]]).unique()
    le_team.fit(all_teams)
    data["team1_enc"]=le_team.transform(data["team1"])   
    data["team2_enc"]=le_team.transform(data["team2"])

    le_venue=LabelEncoder()
    data["venue_enc"]=le_venue.fit_transform(data["venue"])

    le_era=LabelEncoder()
    data["era_enc"]=le_era.fit_transform(data["era"])

    features=["team1_enc","team2_enc","venue_enc","toss_winner_is_team1","chose_bat","h2h_winrate","team1_form","team2_form","team1_is_home","era_enc"]
    x=data[features]
    y=data["team1_won"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    model=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=5,random_state=42)
    model.fit(x_train,y_train)
    print(model.classes_)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")


    importance_df=pd.DataFrame({ "Feature": features,"Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    print(importance_df)

     # Save model + encoders together
    with open("model.pkl","wb") as f:
        pickle.dump({
            "model":model,
            "le_team1":le_team,
            "le_venue": le_venue,
            "le_era": le_era,
            "features": features,
            "accuracy": accuracy,
            "feature_importance": importance_df
        },f)

    print("Model saved to model.pkl")    
    return model,accuracy

if __name__=="__main__":
    train_model()




