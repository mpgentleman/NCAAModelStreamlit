#import NCAA_Functions as NF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
import requests
import time
from bs4 import BeautifulSoup
import streamlit as st
from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import numpy as np
#import selenium
#from selenium import webdriver
#from webdriver_manager.chrome import ChromeDriverManager



def GetThisTeamInfoFromCsv(ThisTeam,WhichFile):


    TeamInfo=pd.read_csv("Data/"+WhichFile+"/"+ThisTeam+"Data.csv")
    return(TeamInfo)


def calculate_to_numeric(price):
    taxes = pd.to_numeric(price,errors='coerce')
    return taxes

def update_type(t1, t2, dropna=False):
    return t1.map(t2).dropna() if dropna else t1.map(t2).fillna(t1)

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

import plotly.graph_objects as go
import pandas as pd

TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
 
AwayTeamAll=list(TeamDatabase2['OldTRankName'])
HomeTeamAll=list(TeamDatabase2['OldTRankName'])


st.title('NCAA Head to Head Matchup')
add_selectbox = st.sidebar.header("Select Todays Date")
add_selectbox_start =st.sidebar.date_input('Pick date')

dateString=str(add_selectbox_start)

dateToday=dateString.replace('-', '')
Dailyschedule=pd.read_csv("Data/DailySchedules2021/"+dateToday+"Schedule.csv")

d2=dateString.split('-')[1]+'_'+dateString.split('-')[2]+'_'+dateString.split('-')[0]
themonth=int(dateString.split('-')[1])
theday=int(dateString.split('-')[2])
theyear=dateString.split('-')[0]

Tables_Selection=st.sidebar.selectbox('Any or Scheduled',['Any', 'Todays Games'])
if 'Any' in  Tables_Selection:
    AwayTeam = st.sidebar.selectbox('Away Team',AwayTeamAll)
    HomeTeam = st.sidebar.selectbox('Home Team',HomeTeamAll)
if 'Todays Games' in  Tables_Selection:

    Tables_Choice=st.sidebar.selectbox('Sort Games By',['Alphabetical', 'Time','Regression_Difference'])
    


    

    if 'Alphabetical'in  Tables_Choice:
        Dailyschedule=Dailyschedule.sort_values(by=['AWAY'])
    if 'Time' in Tables_Choice:
        Dailyschedule=Dailyschedule.sort_values(by=['Time'])   
    if 'Regression_Difference' in Tables_Choice: 
        Dailyschedule=Dailyschedule.sort_values(by=['Reg_dif'])
    AwayList=list(Dailyschedule['AWAY'])
    HomeList=list(Dailyschedule['HOME'])

    AwayTeam = st.sidebar.selectbox('Away Team',AwayList)
    HomeTeam = st.sidebar.selectbox('Home Team',HomeList)



    


if st.button('Run'):
    dateforRankings=dateToday
    dateforRankings5=d2

    #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
    TeamDatabase2.set_index("OldTRankName", inplace=True)
    MG_DF1=pd.read_csv("Data/MGRankings/tm_seasons_stats_ranks"+dateforRankings5+".csv")
    MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
    MG_DF1.set_index("updated", inplace=True)
    from matplotlib.backends.backend_pdf import PdfPages
    WhichFile='TeamDataFiles2021'
    pp= PdfPages("Daily_Team_Charts_"+dateToday+".pdf")
    if 'Todays Games' in  Tables_Selection:
        st.header('Games Today')
    

    


        lengthrows=int(len(Dailyschedule)/2)
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(Dailyschedule.columns),
                    fill_color='grey',
                    align='left'),
            cells=dict(values=[Dailyschedule.AWAY, Dailyschedule.HOME, Dailyschedule.VegasSpread, Dailyschedule.VegasTotal, Dailyschedule.Court, Dailyschedule.Time,Dailyschedule.Reg_dif],
            fill_color = [[rowOddColor,rowEvenColor]*lengthrows],
                align='left',
            font_size=12,
            height=30))
        ])


        fig.update_layout(width=1200, height=800)

        st.plotly_chart(fig)
    

    

    





    TeamDatabase=pd.read_csv("Data/TeamDatabase.csv")
    #TeamDatabase.set_index("University", inplace=True)
    #TeamList=list(TeamDatabase1["OldTRankName"])
    TeamDatabase.set_index("OldTRankName", inplace=True)
    Dailyschedule['VegasSpread'] = Dailyschedule.VegasSpread.apply(calculate_to_numeric)
    Dailyschedule['Total'] = Dailyschedule.VegasTotal.apply(calculate_to_numeric)   
    #PomeroyDF1=GetPomeroyData()
    PomeroyDF1=pd.read_csv("Data/PomeroyDailyRankings2021/PomeroyRankings"+dateforRankings+".csv")
    #PomeroyDF1=sanitizeEntireColumn(PomeroyDF1,"Team")
    PomeroyDF1["updated"]=update_type(PomeroyDF1.Team,TeamDatabase.set_index('PomeroyName').UpdatedTRankName)
    PomeroyDF1["updated"]=PomeroyDF1["updated"].str.rstrip()
   
    PomeroyDF1.set_index("updated", inplace=True)

    BartDF1=pd.read_csv("Data/TRankDailyRankings2021/"+dateforRankings+".csv")
    #getBartDataTest()
    #print(BartDF1)  
    BartDF1["updated"]=update_type(BartDF1.Team,TeamDatabase.set_index('TRankName').UpdatedTRankName)

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



    st.header('Team Matchup')

    #### Sidebar Creation #######

    

    test1=GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2021")

    test2=GetThisTeamInfoFromCsv(HomeTeam,"TeamDataFiles2021")
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
    
    st.subheader('Polynomial Regression Charts')
    st.text('Daily Pomeroy Rankings line in green for each game')
    st.text('Polynomial Regression of actual game performance in blue for each game ')
    st.text('If the blue line is above the green then the team is playing better than its ranking ')
    st.pyplot(fig)
    
    st.subheader('Pomeroy Ranking and ATS Record')
    st.text('Pomeroy Rankings by game Line in Green')
    st.text('Blue bars are positive if the team won against the spread')
    GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","PomAdjEMCurrent","PomAdjEMCurrent","ATS")
    
    st.subheader('MG Rankings and ATS spread')
    st.text('MG Rankings by game Line in Green')
    #GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating",'AdjEM_MG','AdjEM_MG',"ATS")
    GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"EMRating","PomAdjEMCurrent","ATS")
    
    st.subheader('Team Playing Over its Ranking')
    st.text('Blue bars are positive if the team played over its rating')
    st.text('The green and blue lines are cumulative moving averages')
    getOverplayingChartBothTeamsDec4(pp,test1,test2,AwayTeam,HomeTeam)
    
    st.subheader('Adjusted Offense and the ATS spread')
    GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"AdjO","PomAdjOECurrent","ATS")
    
    st.subheader('Adjusted Defense against the Over/Under')
    GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"AdjD","PomAdjDECurrent","OverUnder")
    
    st.subheader('Estimated Pace against the Over/Under')
    GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Pace","PomTempo","OverUnder")
    
    st.subheader('Points per Possesion against the ATS')
    GetTwoTeamChartsTogether(test1,test2,AwayTeam,HomeTeam,"PPP","ATS")
    
    st.subheader('Defensive Points per Possesion against the Over/Under')
    GetTwoTeamChartsTogether(test1,test2,AwayTeam,HomeTeam,"PPP1","OverUnder")

#st.pyplot(fig2)


#st.pyplot(fig3)
#st.pyplot(fig4)
