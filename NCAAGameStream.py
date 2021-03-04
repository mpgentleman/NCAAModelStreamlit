import NCAA_Functions as NF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
import requests
import time
from bs4 import BeautifulSoup
import streamlit as st

def GetTwoChartsTogether_EMA(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,SecondStat,PomStatHome,PomStatAway,VegasStat):
    HomeTeamInfo["EM3"]=HomeTeamInfo['AdjO3ExpMA']-HomeTeamInfo['AdjD3ExpMA']
    HomeTeamInfo["EM5"]=HomeTeamInfo['AdjO5ExpMA']-HomeTeamInfo['AdjD5ExpMA']
    HomeTeamInfo["EM10"]=HomeTeamInfo['AdjO10ExpMA']-HomeTeamInfo['AdjD10ExpMA']

    AwayTeamInfo["EM3"]=AwayTeamInfo['AdjO3ExpMA']-AwayTeamInfo['AdjD3ExpMA']
    AwayTeamInfo["EM5"]=AwayTeamInfo['AdjO5ExpMA']-AwayTeamInfo['AdjD5ExpMA']
    AwayTeamInfo["EM10"]=AwayTeamInfo['AdjO10ExpMA']-AwayTeamInfo['AdjD10ExpMA']

    
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+SecondStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.scatter(AwayTeamInfo.index,AwayTeamInfo[SecondStat],color='black')
    ax1.plot(AwayTeamInfo[PomStatAway],color='green')
    ax1.plot(AwayTeamInfo["EM3"],color='red')
    ax1.plot(AwayTeamInfo["EM5"],color='black')
    ax1.plot(AwayTeamInfo["EM10"],color='purple')
    
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='dodgerblue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.scatter(HomeTeamInfo.index,HomeTeamInfo[FirstStat],color='black')
    ax2.plot(HomeTeamInfo[PomStatHome],color='green')
    ax2.plot(HomeTeamInfo["EM3"],color='red')
    ax2.plot(HomeTeamInfo["EM5"],color='black')
    ax2.plot(HomeTeamInfo["EM10"],color='purple')

    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat],color='dodgerblue')
    st.pyplot(f)
    #plt.show()
    
def GetTwoTeamChartsTogetherDec6(pp,AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,PomStat,VegasStat):
    HomeTeamInfo["First 3 Game"]=HomeTeamInfo[FirstStat].rolling(3).mean()
    HomeTeamInfo["First 5 Game"]=HomeTeamInfo[FirstStat].rolling(5).mean()
    HomeTeamInfo["First 10 Game"]=HomeTeamInfo[FirstStat].rolling(10).mean()
    
    AwayTeamInfo["Second 3 Game"]=AwayTeamInfo[FirstStat].rolling(3).mean()
    AwayTeamInfo["Second 5 Game"]=AwayTeamInfo[FirstStat].rolling(5).mean()
    AwayTeamInfo["Second 10 Game"]=AwayTeamInfo[FirstStat].rolling(10).mean()
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+FirstStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.scatter(AwayTeamInfo.index,AwayTeamInfo[FirstStat])
    ax1.plot(AwayTeamInfo[PomStat])
    ax1.plot(AwayTeamInfo["Second 10 Game"])
    ax1.plot(AwayTeamInfo["Second 3 Game"],color='red')
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='blue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.scatter(HomeTeamInfo.index,HomeTeamInfo[FirstStat])
    ax2.plot(HomeTeamInfo[PomStat])
    ax2.plot(HomeTeamInfo["First 10 Game"])
    ax2.plot(HomeTeamInfo["First 3 Game"],color='red')
    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat],color='blue')
    st.pyplot(f)
    #plt.show()
    #f.savefig(pp,format='pdf')
def getOverplayingChartBothTeamsDec4(pp,HomeTeamData,AwayTeamData,HomeTeam,AwayTeam):
    
    

   
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    #ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ChartTitleName=HomeTeam+" Overplaying and ATS"
    ax1.set_title(ChartTitleName)
    ax1.plot(HomeTeamData["DifCumSum"])
    ax1.plot(HomeTeamData["DifCumSumEMA"])
    ax1.bar(HomeTeamData.index,HomeTeamData["ATS"])
    ChartTitleName=AwayTeam+" Overplaying and ATS"
    
    ax2.set_title(ChartTitleName)
    #ax1.scatter(AwayTeamData.index,AwayTeamInfo[FirstStat])
    ax2.plot(AwayTeamData["DifCumSum"])
    ax2.plot(AwayTeamData["DifCumSumEMA"])
    ax2.bar(AwayTeamData.index,AwayTeamData["ATS"])
    st.pyplot(f)
    #plt.show()
    #f.savefig(pp,format='pdf')    
def GetTwoTeamChartsTogether(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,VegasStat):
    HomeTeamInfo["First 3 Game"]=HomeTeamInfo[FirstStat].rolling(3).mean()
    HomeTeamInfo["First 5 Game"]=HomeTeamInfo[FirstStat].rolling(5).mean()
    HomeTeamInfo["First 10 Game"]=HomeTeamInfo[FirstStat].rolling(10).mean()
    
    AwayTeamInfo["Second 3 Game"]=AwayTeamInfo[FirstStat].rolling(3).mean()
    AwayTeamInfo["Second 5 Game"]=AwayTeamInfo[FirstStat].rolling(5).mean()
    AwayTeamInfo["Second 10 Game"]=AwayTeamInfo[FirstStat].rolling(10).mean()
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+FirstStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.scatter(AwayTeamInfo.index,AwayTeamInfo[FirstStat])
    ax1.plot(AwayTeamInfo["PomAdjEMCurrent"])
    ax1.plot(AwayTeamInfo["Second 10 Game"],color='green')
    ax1.plot(AwayTeamInfo["Second 3 Game"],color='red')
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='blue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.scatter(HomeTeamInfo.index,HomeTeamInfo[FirstStat])
    ax2.plot(HomeTeamInfo["PomAdjEMCurrent"])
    ax2.plot(HomeTeamInfo["First 10 Game"],color='green')
    ax2.plot(HomeTeamInfo["First 3 Game"],color='red')
    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat],color='blue')
    st.pyplot(f)
    #plt.show() 
    
def get_team_reg_dif(teamname):
    test1=NF.GetThisTeamInfoFromCsv(teamname,"TeamDataFiles2021")
    test1
    test1['New_ID'] = range(0, 0+len(test1))
    p = sns.regplot(x="New_ID",y="EMRating5GameExpMA",order=2,data=test1)
    p2=sns.regplot(x='New_ID', y='PomAdjEMCurrent',order=2, data=test1)
    plt.show()

    a=p.get_lines()[0].get_ydata()
    a1=p2.get_lines()[1].get_ydata()
    a3=np.subtract(a,a1)
    plt.plot(a3)
    plt.show()
    return(a3[-1])
import numpy as np
from datetime import datetime,date,time
add_selectbox = st.sidebar.header("Date Range Picker")
add_selectbox_start =st.sidebar.date_input('start date')
#add_selectbox_finish =st.sidebar.date_input('end_date')
st.header(add_selectbox_start)
dateString=str(add_selectbox_start)
#s1.replace('-', '')
dateToday=dateString.replace('-', '')
d2=dateString.split('-')[1]+'_'+dateString.split('-')[2]+'_'+dateString.split('-')[0]
st.header(d2)
st.header(dateToday)
#dateToday='20210227'
dateforRankings=dateToday
dateforRankings5=d2

#dateToday=dateToGetNowWrite[0]
#dateforRankings=dateToGetNowWrite[0]
#dateforRankings5=datesUnderlineAdd[0]




TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
TeamDatabase2.set_index("OldTRankName", inplace=True)
MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings/tm_seasons_stats_ranks"+dateforRankings5+".csv")
MG_DF1["updated"]=NF.update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
MG_DF1.set_index("updated", inplace=True)
from matplotlib.backends.backend_pdf import PdfPages
WhichFile='TeamDataFiles2021'
pp= PdfPages("Daily_Team_Charts_"+dateToday+".pdf")
Dailyschedule=pd.read_csv("C:/Users/mpgen/DailySchedules2021/"+dateToday+"Schedule.csv")
Dailyschedule=Dailyschedule.sort_values(by=['Reg_dif'])


import plotly.graph_objects as go
import pandas as pd
lengthrows=int(len(Dailyschedule)/2)
rowEvenColor = 'lightgrey'
rowOddColor = 'white'
fig = go.Figure(data=[go.Table(
    header=dict(values=list(Dailyschedule.columns),
                fill_color=rowEvenColor,
                align='left'),
    cells=dict(values=[Dailyschedule.AWAY, Dailyschedule.HOME, Dailyschedule.VegasSpread, Dailyschedule.VegasTotal, Dailyschedule.Court, Dailyschedule.Time,Dailyschedule.Reg_dif],
    fill_color = [[rowOddColor,rowEvenColor]*lengthrows],
               align='left',
    font_size=12,
    height=30))
])


fig.update_layout(width=1200, height=800)
#fig.show()
st.plotly_chart(fig)
AwayList=list(Dailyschedule['AWAY'])
HomeList=list(Dailyschedule['HOME'])

AwayTeam = st.sidebar.selectbox('Away Team',AwayList)
HomeTeam = st.sidebar.selectbox('Home Team',HomeList)
whereIsGame = st.sidebar.selectbox('Neutral Site',['Yes', 'No'])


TeamDatabase=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
#TeamDatabase.set_index("University", inplace=True)
#TeamList=list(TeamDatabase1["OldTRankName"])
TeamDatabase.set_index("OldTRankName", inplace=True)
Dailyschedule['VegasSpread'] = Dailyschedule.VegasSpread.apply(NF.calculate_to_numeric)
Dailyschedule['Total'] = Dailyschedule.VegasTotal.apply(NF.calculate_to_numeric)   
#PomeroyDF1=GetPomeroyData()
PomeroyDF1=pd.read_csv("C:/Users/mpgen/PomeroyDailyRankings2021/PomeroyRankings"+dateforRankings+".csv")
#PomeroyDF1=sanitizeEntireColumn(PomeroyDF1,"Team")
PomeroyDF1["updated"]=NF.update_type(PomeroyDF1.Team,TeamDatabase.set_index('PomeroyName').UpdatedTRankName)
PomeroyDF1["updated"]=PomeroyDF1["updated"].str.rstrip()
   
PomeroyDF1.set_index("updated", inplace=True)

BartDF1=pd.read_csv("C:/Users/mpgen/TRankDailyRankings2021/"+dateforRankings+".csv")
#getBartDataTest()
#print(BartDF1)  
BartDF1["updated"]=NF.update_type(BartDF1.Team,TeamDatabase.set_index('TRankName').UpdatedTRankName)

BartDF1.set_index("updated", inplace=True)


import seaborn
MonteCarloNumberofGames=5
SecondMonteCarloNumberofGames=10
LeagueTempo=70.2
LeagueOE=102.2
LeagueOE=102.9
OverplayingList=[]
OverplayingList2=[]
appendTeamList=[]
appended_data1=[]
appended_dataExtra=[]
appended_dataTime=[]
appendTeam_OU_List=[]

appended_data1MG=[]
appendTeamListMG=[]



st.title('NCAA Game Matchup')

#### Sidebar Creation #######



test1=NF.GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2021")

test2=NF.GetThisTeamInfoFromCsv(HomeTeam,"TeamDataFiles2021")
AwayTeamB=TeamDatabase.loc[AwayTeam,"UpdatedTRankName"]
HomeTeamB=TeamDatabase.loc[HomeTeam,"UpdatedTRankName"]

AwayTeamP=TeamDatabase.loc[AwayTeam,"UpdatedTRankName"]
HomeTeamP=TeamDatabase.loc[HomeTeam,"UpdatedTRankName"]
    
AwayTeamM=TeamDatabase2.loc[AwayTeam,"SportsReference"]
HomeTeamM=TeamDatabase2.loc[HomeTeam,"SportsReference"]
test1['AdjEM_MG']=test1['AdjOE_MG']-test1['AdjDE_MG']
test2['AdjEM_MG']=test2['AdjOE_MG']-test2['AdjDE_MG']

#NF.GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","PomAdjEMCurrent","PomAdjEMCurrent","ATS")
plt.style.use('seaborn')
    
test1['New_ID'] = range(0, 0+len(test1))
test2['New_ID'] = range(0, 0+len(test2))
#p=sns.regplot(x="New_ID", y="EMRating", data=DftoChange,order=2);
fig_dims = (15,10)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=fig_dims)
    #fig, axs = plt.subplots(ncols=2,figsize=fig_dims)
plt.figure(figsize=(20, 12))
ax1.set_title(AwayTeam)
ax2.set_title(HomeTeam)
fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=2, ax=ax1)
fig2=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test1,order=2, ax=ax1)
fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2)
fig4=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test2,order=2, ax=ax2)   
st.pyplot(fig)

GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","PomAdjEMCurrent","PomAdjEMCurrent","ATS")
#GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating",'AdjEM_MG','AdjEM_MG',"ATS")
GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"EMRating","PomAdjEMCurrent","ATS")
getOverplayingChartBothTeamsDec4(pp,test1,test2,AwayTeam,HomeTeam)
GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"AdjO","PomAdjOECurrent","ATS")
GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"AdjD","PomAdjDECurrent","OverUnder")
GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Pace","PomTempo","OverUnder")
  
GetTwoTeamChartsTogether(test1,test2,AwayTeam,HomeTeam,"PPP","ATS")
GetTwoTeamChartsTogether(test1,test2,AwayTeam,HomeTeam,"PPP1","OverUnder")

#st.pyplot(fig2)


#st.pyplot(fig3)
#st.pyplot(fig4)

team2Signals=list(test2["SignalSum"])
team1Signals=list(test1["SignalSum"])
    #print(test1)
    #print(test2)
if len(test2)>0:
        
    HomeTeamSignalScore=team2Signals[-1]
    #AwayTeamSignalScore=team1Signals[-1]   
    team2Play=list(test2["PlayingOverRating"])
    HomeTeamSignalPlay=team2Play[-1]
    team2PlaySum=list(test2["DifCumSum"])
    HomeTeamSignalPlayOver=team2PlaySum[-1]

    OverplayingList.append([HomeTeamB,HomeTeamSignalPlay,HomeTeamSignalPlayOver])
else:
    HomeTeamSignalPlay=0
    HomeTeamSignalPlayOver=0
    test2=test1
    #AwayTeamSignalPlay=0
    print("No games for Home Team")
if len(test1)>0:

    AwayTeamSignalScore=team1Signals[-1]   

    team1Play=list(test1["PlayingOverRating"])
    AwayTeamSignalPlay=team1Play[-1]
    team1PlayOver=list(test1["DifCumSum"])
    AwayTeamSignalPlayOver=team1PlayOver[-1]
    OverplayingList.append([AwayTeamB,AwayTeamSignalPlay,AwayTeamSignalPlayOver])

else:
    #HomeTeamSignalPlay=
    AwayTeamSignalPlayOver=0
    AwayTeamSignalPlay=0  
    test1=test2
    print("No games for Away Team")

if whereisGame =="N":

    PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NF.SetPomeroyDataNeutral2020(PomeroyDF1,AwayTeamP,HomeTeamP)
    #print(thePGameTempo)
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NF.SetBartDataNeutral(BartDF1,AwayTeamB,HomeTeamB,thePGameTempo)
    MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NF.Set_MG_Data_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)

    #print(BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo)
    B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NF.NewgetGamePredictionBartNeutralCourt(test1.iloc[len(test1.index)-1]["AdjO3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjD3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjO3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjD3ExpMAS"],thePGameTempo,LeagueOE)
    #CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBartNeutralCourtCurrent(test1.iloc[len(test1.index)-1]["AdjTCurrent"],test1.iloc[len(test1.index)-1]["AdjOECurrent"],test1.iloc[len(test1.index)-1]["AdjDECurrent"],test2.iloc[len(test2.index)-1]["AdjTCurrent"],test2.iloc[len(test2.index)-1]["AdjOECurrent"],test2.iloc[len(test2.index)-1]["AdjDECurrent"],LeagueOE,LeagueTempo)
    #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
    r,r1,theEstimatedTotal,theEstimatedSpread=NF.getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
    Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=NF.getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
    MG_Rank_Score_Dif=NF.get_MG_Margin_Dif_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
else:
    if whereisGame =="A":
        PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NF.SetPomeroyData2020(PomeroyDF1,HomeTeamP,AwayTeamP)
        BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NF.SetBartData(BartDF1,HomeTeamB,AwayTeamB,thePGameTempo)
        
        MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NF.Set_MG_Data(MG_DF1,HomeTeamM,AwayTeamM,thePGameTempo)


        #print(BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo)
        B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NF.NewgetGamePredictionBart(test1.iloc[len(test1.index)-1]["AdjO3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjD3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjO3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjD3ExpMAS"],thePGameTempo,LeagueOE)
        #CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBartCurrent(test1.iloc[len(test1.index)-1]["AdjTCurrent"],test1.iloc[len(test1.index)-1]["AdjOECurrent"],test1.iloc[len(test1.index)-1]["AdjDECurrent"],test2.iloc[len(test2.index)-1]["AdjTCurrent"],test2.iloc[len(test2.index)-1]["AdjOECurrent"],test2.iloc[len(test2.index)-1]["AdjDECurrent"],LeagueOE,LeagueTempo)
        #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
        r,r1,theEstimatedTotal,theEstimatedSpread=NF.getMonteCarloGameScore(test2,test1,MonteCarloNumberofGames,10000,thePGameTempo)
        Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=NF.getMonteCarloGameScore(test2,test1,SecondMonteCarloNumberofGames,10000,thePGameTempo)
        MG_Rank_Scoree_Dif=NF.get_MG_Margin_Dif_Ratio(MG_DF1,HomeTeamM,AwayTeamM,thePGameTempo)
    else:
        PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NF.SetPomeroyData2020(PomeroyDF1,AwayTeamP,HomeTeamP)
        BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NF.SetBartData(BartDF1,AwayTeamB,HomeTeamB,thePGameTempo)
        
        MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NF.Set_MG_Data(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)

        B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NF.NewgetGamePredictionBart(test2.iloc[len(test2.index)-1]["AdjO3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjD3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjO3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjD3ExpMAS"],thePGameTempo,LeagueOE)
        #CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBartCurrent(test1.iloc[len(test1.index)-1]["AdjTCurrent"],test1.iloc[len(test1.index)-1]["AdjOECurrent"],test1.iloc[len(test1.index)-1]["AdjDECurrent"],test2.iloc[len(test2.index)-1]["AdjTCurrent"],test2.iloc[len(test2.index)-1]["AdjOECurrent"],test2.iloc[len(test2.index)-1]["AdjDECurrent"],LeagueOE,LeagueTempo)
        #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
        r,r1,theEstimatedTotal,theEstimatedSpread=NF.getMonteCarloGameScore(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
        Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=NF.getMonteCarloGameScore(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
        MG_Rank_Score_Dif=NF.get_MG_Margin_Dif_Ratio(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
if whereisGame != "H":
    B3GHomeTeamSpread=B3GHomeTeamSpread*-1
    PHomeTeamSpread=PHomeTeamSpread*-1
    BHomeTeamSpread=BHomeTeamSpread*-1
    MHomeTeamSpread=MHomeTeamSpread*-1
    theEstimatedSpread=theEstimatedSpread*-1
    theEstimatedSpread10G=theEstimatedSpread10G*-1  
MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
B3GHomeTeamSpread=B3GHomeTeamSpread*-1   