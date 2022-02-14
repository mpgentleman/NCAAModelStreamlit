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
#import st_aggrid
#import selenium
#from selenium import webdriver
#from webdriver_manager.chrome import ChromeDriverManager
from st_aggrid import AgGrid,JsCode,DataReturnMode,GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import random
from joypy import joyplot
import numpy as np
from datetime import datetime,date,time

import plotly.graph_objects as go
import pandas as pd

def cellStyleDynamic(data: pd.Series):

    datNeg = data[data < 0]
    datPos = data[data > 0]

    if len(datNeg) > 0 and len(datPos) > 0:
        _, binsN = pd.cut(datNeg, bins=4, retbins=True, precision=0)
        _, binsP = pd.cut(datPos, bins=4, retbins=True, precision=0)
        code = """
            function(params) {
              if (isNaN(params.value) || params.value === 0) return {'color': 'black', 'backgroundColor': '#e9edf5'};
              if (params.value < %d) return {'color': 'white','backgroundColor':  '#ff0000'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff4c4c'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff9999'};
              if (params.value < 0) return {'color': 'white', 'backgroundColor': '#ffb2b2'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#a8bad9'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#7d97c6'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#3c64aa'};
              
              return {'color': 'white', 'backgroundColor': '#2753a1'};
            };
            """ % (binsN[1], binsN[2], binsN[3], binsP[1], binsP[2], binsP[3])

    elif len(datNeg) > 0:
        dat, bins = pd.cut(datNeg, bins=4, retbins=True, precision=0)
        code = """
            function(params) {
              if (isNaN(params.value) || params.value === 0) return {'color': 'black', 'backgroundColor': '#e9edf5'};
              if (params.value < %d) return {'color': 'white','backgroundColor':  '#ff0000'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff4c4c'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff9999'};
              if (params.value < 0) return {'color': 'white', 'backgroundColor': '#ffb2b2'};
              
              return {'color': 'white', 'backgroundColor': '#2753a1'};
            };
            """ % (bins[1], bins[2], bins[3])

    elif len(datPos) > 0:
        dat, bins = pd.cut(datPos, bins=4, retbins=True, precision=0)
        code = """
            function(params) {
              if (isNaN(params.value) || params.value === 0) return {'color': 'black', 'backgroundColor': '#e9edf5'};
              if (params.value < 0) return {'color': 'white', 'backgroundColor': '#ffb2b2'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#a8bad9'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#7d97c6'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#3c64aa'};
              
              return {'color': 'white', 'backgroundColor': '#2753a1'};
            };
            """ % (bins[1], bins[2], bins[3])
    else:
        code = """
            function(params) {
              return {'color': 'white', 'backgroundColor': '#a8bad9'};
            };
            """

    return JsCode(code)
cellStyle = JsCode("""
            function(params) {
                return {'color': 'black','backgroundColor':  '#e8f4f8'};
            };
            """)
def numberFormat(precision: int, comma: bool = True):
    if comma:
        jscript = """
        function(params) {
          if (isNaN(params.value)) {
            return "";
          }
          return params.value.toFixed(%d).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        };
        """ % precision
    else:
        jscript = """
        function(params) {
          if (isNaN(params.value)) {
            return "";
          }
          return params.value.toFixed(%d);
        };
        """ % precision

    return JsCode(jscript)

def cellStyleGrey():
    j = """
    function(params) {
      return {'color': 'black', 'backgroundColor': '#ececec'};
    };
    """
    return JsCode(j)


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

def getDistributionMatchupCharts(AwayTeam,HomeTeam):
    
    teamname1=AwayTeam
    test1=GetThisTeamInfoFromCsv(teamname1,"TeamDataFilesStarter2022")
    teamname2=HomeTeam
    test2=GetThisTeamInfoFromCsv(teamname2,"TeamDataFilesStarter2022")

    test2EFG=test2['EFG%']
    test2TO=test2['TO%']
    test2OR=test2['OR%']
    test2FTR=test2['FTR%']
    test2ADJO=test2['AdjO']
    test2ADJD=test2['AdjD']
    test2ADJEM=test2['EMRating']


    test1EFG=random.choices(list(test1['EFG%']), k=len(test2['EFG%']))
    test1TO=random.choices(list(test1['TO%']), k=len(test2['TO%']))
    test1OR=random.choices(list(test1['OR%']), k=len(test2['OR%']))
    test1FTR=random.choices(list(test1['FTR%']), k=len(test2['FTR%']))
    test1ADJO=random.choices(list(test1['AdjO']), k=len(test2['AdjO']))
    test1ADJD=random.choices(list(test1['AdjD']), k=len(test2['AdjD']))
    test1ADJEM=random.choices(list(test1['EMRating']), k=len(test2['EMRating']))

    data = pd.DataFrame({teamname1:test1EFG,teamname2:test2EFG})

    data['Stat']='EFG%'

    data2 = pd.DataFrame({teamname1:test1TO,teamname2:test2TO})
    data2['Stat']='TO%'
    #data2['Base']=1
    data3 = pd.DataFrame({teamname1:test1OR,teamname2:test2OR})
    data3['Stat']='OR%'
    #data2['Base']=1
    data5 = pd.DataFrame({teamname1:test1FTR,teamname2:test2FTR})
    data5['Stat']='FTR%'
    data6 = pd.DataFrame({teamname1:test1ADJO,teamname2:test2ADJO})
    data6['Stat']='AdjO'
    data7 = pd.DataFrame({teamname1:test1ADJD,teamname2:test2ADJD})
    data7['Stat']='AdjD'
    data8 = pd.DataFrame({teamname1:test1ADJEM,teamname2:test2ADJEM})
    data8['Stat']='AdjEM'


    dataOne=pd.concat([data6,data7])
    data4=pd.concat([data,data2,data3,data5])
    #st.dataframe(dataOne)
    #st.dataframe(data4)
    plt.figure(figsize=(5,5), dpi= 80)

    #plt.figure(dpi= 380)
    #ax1, fig =joyplot(data8[[teamname1,teamname2,'Stat']],
    #   by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True )
    #st.pyplot(ax1)
    ax2, fig2 =joyplot(dataOne[[teamname1,teamname2,'Stat']],
       by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True ,figsize=(5, 8))
    
    plt.title('Efficiency Matchup Stats', fontsize=10)
    plt.show()
    st.pyplot(ax2)
    ax3, fig3 =joyplot(data4[[teamname1,teamname2,'Stat']],
    by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True ,figsize=(15, 15))
    
    plt.title('Four Factors Matchup Stats', fontsize=20)
    plt.show()
    st.pyplot(ax3)
#data4
def getDistributionMatchupChartsNew(AwayTeam,HomeTeam):
    
    teamname1=AwayTeam
    test1=GetThisTeamInfoFromCsv(teamname1,"TeamDataFiles2022")
    teamname2=HomeTeam
    test2=GetThisTeamInfoFromCsv(teamname2,"TeamDataFiles2022")

    test2EFG=test2['EFG%']
    test2TO=test2['TO%']
    test2OR=test2['OR%']
    test2FTR=test2['FTR%']
    test2ADJO=test2['AdjO']
    test2ADJD=test2['AdjD']
    test2ADJEM=test2['EMRating']


    test1EFG=random.choices(list(test1['EFG%']), k=len(test2['EFG%']))
    test1TO=random.choices(list(test1['TO%']), k=len(test2['TO%']))
    test1OR=random.choices(list(test1['OR%']), k=len(test2['OR%']))
    test1FTR=random.choices(list(test1['FTR%']), k=len(test2['FTR%']))
    test1ADJO=random.choices(list(test1['AdjO']), k=len(test2['AdjO']))
    test1ADJD=random.choices(list(test1['AdjD']), k=len(test2['AdjD']))
    test1ADJEM=random.choices(list(test1['EMRating']), k=len(test2['EMRating']))

    data = pd.DataFrame({teamname1:test1EFG,teamname2:test2EFG})

    data['Stat']='EFG%'

    data2 = pd.DataFrame({teamname1:test1TO,teamname2:test2TO})
    data2['Stat']='TO%'
    #data2['Base']=1
    data3 = pd.DataFrame({teamname1:test1OR,teamname2:test2OR})
    data3['Stat']='OR%'
    #data2['Base']=1
    data5 = pd.DataFrame({teamname1:test1FTR,teamname2:test2FTR})
    data5['Stat']='FTR%'
    data6 = pd.DataFrame({teamname1:test1ADJO,teamname2:test2ADJO})
    data6['Stat']='AdjO'
    data7 = pd.DataFrame({teamname1:test1ADJD,teamname2:test2ADJD})
    data7['Stat']='AdjD'
    data8 = pd.DataFrame({teamname1:test1ADJEM,teamname2:test2ADJEM})
    data8['Stat']='AdjEM'


    dataOne=pd.concat([data6,data7,data8])
    data4=pd.concat([data,data2,data3,data5])
    c1,c2= st.columns(2)
    #with c1:
    #    c1.subheader('Four Factor Matchup') 
    #    ax1, fig =joyplot(data4[[teamname1,teamname2,'Stat']],
    #       by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True,range_style='own', 
    #                      grid="y", linewidth=1,figsize=(4, 4) )
    #    plt.title('Four Factor Matchup Stats', fontsize=10)
    #    st.pyplot(ax1)

    #    plt.show()
    #with c2:
    #    c2.subheader('Efficiency Matchup Stats') 
    #    ax2, fig2 =joyplot(dataOne[[teamname1,teamname2,'Stat']],
    #       by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True ,figsize=(4, 4))
    
    #    plt.title('Efficiency Matchup Stats', fontsize=10)
    #    plt.show()
    #    st.pyplot(ax2)
    #st.dataframe(data8)
    st.subheader('Four Factor Matchup') 
    ax4, fig =joyplot(data4[[teamname1,teamname2,'Stat']],
           by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True, 
                          grid="y", linewidth=1,figsize=(15, 6) )
    plt.title('Four Factor Matchup Stats', fontsize=10)
    st.pyplot(ax4)
    st.subheader('Efficiency Matchup Stats') 
    ax3, fig2 =joyplot(dataOne[[teamname1,teamname2,'Stat']],
           by='Stat',column=[teamname1, teamname2],alpha=.70,legend=True ,figsize=(15, 6))
    
    plt.title('Efficiency Matchup Stats', fontsize=10)
    #plt.show()
    st.pyplot(ax3)

def getTeamDFTable(team1,teamname):
    colsM=['DateNew','Op Rank','Opponent','Result','Pace','ATSVegas','OverUnderVegas','ATS','EMRating','PlayingOverRating']
    numeric=['numericColumn','numberColumnFilter']
    team1=team1[colsM]
    allcols=team1.columns
    header1=teamname+' Game History'
    st.subheader(header1)
    csTotal=cellStyleDynamic(team1.PlayingOverRating)
    gb = GridOptionsBuilder.from_dataframe(team1)
    #gb.configure_columns('PlayingOverRating', type=numeric, valueFormatter=numberFormat(1))
    gb.configure_columns(allcols, cellStyle=cellStyle)
    gb.configure_column('PlayingOverRating',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()

    AgGrid(team1, gridOptions=gridOptions, enable_enterprise_modules=True,height=1000,allow_unsafe_jscode=True)


st.set_page_config(layout="wide")
TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
 
AwayTeamAll=list(TeamDatabase2['OldTRankName'])
HomeTeamAll=list(TeamDatabase2['OldTRankName'])


st.title('NCAA Head to Head Matchup')
add_selectbox = st.sidebar.header("Select Todays Date")
add_selectbox_start =st.sidebar.date_input('Pick date')



Tables_Selection=st.sidebar.selectbox('Any or Scheduled',['Any', 'Todays Games'])
if 'Any' in  Tables_Selection:
    AwayTeam = st.sidebar.selectbox('Away Team',AwayTeamAll)
    HomeTeam = st.sidebar.selectbox('Home Team',HomeTeamAll)


    



    


if st.button('Run'):
    dateforRankings=dateToday
    dateforRankings5=d2

    #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
    TeamDatabase2.set_index("OldTRankName", inplace=True)
    MG_DF1=pd.read_csv("Data/MGRankings2022/tm_seasons_stats_ranks"+dateforRankings5+" .csv")
    MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
    MG_DF1.set_index("updated", inplace=True)
    from matplotlib.backends.backend_pdf import PdfPages
    WhichFile='TeamDataFiles2021'
    pp= PdfPages("Daily_Team_Charts_"+dateToday+".pdf")
    if 'Todays Games' in  Tables_Selection:
        st.header('Games Today')
        dateString=str(add_selectbox_start)

        dateToday=dateString.replace('-', '')
        Dailyschedule=pd.read_csv("Data/DailySchedules2022/"+dateToday+"Schedule.csv")

        d2=dateString.split('-')[1]+'_'+dateString.split('-')[2]+'_'+dateString.split('-')[0]
        themonth=int(dateString.split('-')[1])
        theday=int(dateString.split('-')[2])
        theyear=dateString.split('-')[0]
        Tables_Choice=st.sidebar.selectbox('Sort Games By',['Alphabetical', 'Time','Regression_Difference','OverPlaying'])
    


    

        if 'Alphabetical'in  Tables_Choice:
            Dailyschedule=Dailyschedule.sort_values(by=['AWAY'])
        if 'Time' in Tables_Choice:
            Dailyschedule=Dailyschedule.sort_values(by=['Time'])   
        if 'Regression_Difference' in Tables_Choice: 
            Dailyschedule=Dailyschedule.sort_values(by=['Reg_dif'])
        if 'OverPlaying' in Tables_Choice: 
            Dailyschedule=Dailyschedule.sort_values(by=['Over_dif'])
        AwayList=list(Dailyschedule['AWAY'])
        HomeList=list(Dailyschedule['HOME'])

        AwayTeam = st.sidebar.selectbox('Away Team',AwayList)
        HomeTeam = st.sidebar.selectbox('Home Team',HomeList)
    


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

        #st.plotly_chart(fig)
        allcols=Dailyschedule.columns
        gb = GridOptionsBuilder.from_dataframe(Dailyschedule)
        gb.configure_columns(allcols, cellStyle=cellStyle)
        csTotal=cellStyleDynamic(Dailyschedule.Reg_dif)
        gb.configure_column('Reg_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
        #gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()

        AgGrid(Dailyschedule, gridOptions=gridOptions, enable_enterprise_modules=True,allow_unsafe_jscode=True,height=800)


    

    





    TeamDatabase=pd.read_csv("Data/TeamDatabase.csv")
    #TeamDatabase.set_index("University", inplace=True)
    #TeamList=list(TeamDatabase1["OldTRankName"])
    TeamDatabase.set_index("OldTRankName", inplace=True)
    Dailyschedule['VegasSpread'] = Dailyschedule.VegasSpread.apply(calculate_to_numeric)
    Dailyschedule['Total'] = Dailyschedule.VegasTotal.apply(calculate_to_numeric)   
    #PomeroyDF1=GetPomeroyData()
    PomeroyDF1=pd.read_csv("Data/PomeroyDailyRankings2022/PomeroyRankings"+dateforRankings+".csv")
    #PomeroyDF1=sanitizeEntireColumn(PomeroyDF1,"Team")
    PomeroyDF1["updated"]=update_type(PomeroyDF1.Team,TeamDatabase.set_index('PomeroyName').UpdatedTRankName)
    PomeroyDF1["updated"]=PomeroyDF1["updated"].str.rstrip()
   
    PomeroyDF1.set_index("updated", inplace=True)

    BartDF1=pd.read_csv("Data/TRankDailyRankings2022/"+dateforRankings+".csv")
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

    

    test1=GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2022")

    test2=GetThisTeamInfoFromCsv(HomeTeam,"TeamDataFiles2022")
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
#############################################
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
    fig5=sns.regplot(x='New_ID', y='AdjEM_MG', data=test1,order=2, ax=ax1)
    fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2)
    fig4=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test2,order=2, ax=ax2)
    fig6=sns.regplot(x='New_ID', y='AdjEM_MG', data=test2,order=2, ax=ax2)
    st.subheader('Polynomial Regression Charts')
    st.text('Daily Pomeroy Rankings line in green for each game')
    st.text('Polynomial Regression of actual game performance in blue for each game ')
    st.text('If the blue line is above the green line(Pomeroy) then the team is playing better than its ranking ')
    st.text('The red line is MG rankings ')
    st.pyplot(fig)






    
    st.subheader('Pomeroy Ranking and ATS Record')
    st.text('Pomeroy Rankings by game Line in Green')
    st.text('Blue bars are positive if the team won against the spread')
    GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","PomAdjEMCurrent","PomAdjEMCurrent","ATS")
    
    st.subheader('MG Rankings and ATS spread')
    st.text('MG Rankings by game Line in Green')
    #GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating",'AdjEM_MG','AdjEM_MG',"ATS")
    GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"EMRating","Adj_Margin_EM_MG","MG_SpreadWinATS")
    
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
    getDistributionMatchupChartsNew(AwayTeam,HomeTeam)

    #getDistributionMatchupCharts(AwayTeam,HomeTeam)
    getTeamDFTable(test1,AwayTeam)
    getTeamDFTable(test2,HomeTeam)
#st.pyplot(fig2)


#st.pyplot(fig3)
#st.pyplot(fig4)
