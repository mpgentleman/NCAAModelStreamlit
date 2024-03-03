# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:54:29 2021

@author: mpgen
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

#from urllib import urlopen
import scipy
from bs4 import BeautifulSoup
from scipy import stats
import math
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import random
import pandas as pd
import requests
#import tabulate
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import scipy as scipy
#import plotly as plotly
#import plotly.plotly as py
import scipy.stats as ss
import numpy as np

import seaborn as sns
import pandas as pd
#import talib as ta
import numpy as np

import argparse


import datetime as dt
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

key_="qQZveKizwAAUyk4gzMmB";

import mechanicalsoup
#import kenpompy
#from kenpompy.utils import login
import requests
#from selenium import webdriver
#import webdriver_manager
#from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import datetime
import scipy
from sportsreference.ncaab.teams import Teams
import datetime
from datetime import datetime
from sportsreference.ncaab.boxscore import Boxscores
from sportsreference.ncaab.boxscore import Boxscore
import scipy.stats

# Import linear model and pre-processing modules from scikit-learn
#from sklearn import linear_model, preprocessing

# Import warnings package, set to suppress warnings throughout this Colab
import warnings
warnings.simplefilter("ignore")

# Import and setup for Plotly in Colab, including function to be called later
#import plotly.plotly as py

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
sns.set_theme()
import pandas as pd

def calculate_to_numeric(price):
    taxes = pd.to_numeric(price,errors='coerce')
    return taxes

#from tabulate import tabulate
def sample_wr(population, k):

	"Chooses k random elements (with replacement) from a population"

	n = len(population)

	_random, _int = random.random,int  # speed hack

	result = [None] * k
	for i in xrange(k):

		j = _int(_random() * n)
		result[i] = population[j]

	return result
import random

import mechanicalsoup
import kenpompy
from kenpompy.utils import login

LeagueTempo=70.3
LeagueOE=100.5
MonteCarloNumberofGames=5
SecondMonteCarloNumberofGames=10







def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    if len(population)>0:
        n = len(population)
        _random, _int = random.random, int  # speed hack 
        result = [None] * k
        for i in range(k):
            j = _int(_random() * n)
            result[i] = population[j]
    else:
        result=[65] * k
    return result



def NewgetGamePrediction(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE):
    #GameTempo=math.trunc((Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo)
    #GameTempo=math.ceil((Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo)
    GameTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo
    #GameTempo=math.trunc((Team1AdjTempo*Team2AdjTempo)/LeagueTempo)
    Team1Score=(.986*Team1AdjOff/LeagueOE*.986*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*1.014*Team1AdjDef*1.014)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return Team1Score,Team2Score,OverUnder,PointDiff,GameTempo

def NewgetGamePredictionNeutralCourt(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE):
  
    GameTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo

    Team1Score=(Team1AdjOff/LeagueOE*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*Team1AdjDef)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return Team1Score,Team2Score,OverUnder,PointDiff,GameTempo





def NewgetGamePredictionBart(Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,TheTempo,LeagueOE):

     
    GameTempo=TheTempo+2
    #GameTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo

    Team1Score=(.986*Team1AdjOff/LeagueOE*.986*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*1.014*Team1AdjDef*1.014)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return Team1Score,Team2Score,OverUnder,PointDiff,GameTempo

def NewgetGamePredictionBartNeutralCourt(Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,TheTempo,LeagueOE):

     
    GameTempo=TheTempo+2
    #GameTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo

    Team1Score=(Team1AdjOff/LeagueOE*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*Team1AdjDef)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return Team1Score,Team2Score,OverUnder,PointDiff,GameTempo


def getGameDistributionNew(Team1Off,Team1Def,Team2Off,Team2Def,GameTempo,LeagueOE):
    k=len(Team1Off)
    Spreadresult = [None] * k
    Pointresult= [None] * k
    for i in range(k):
        Team1Score=(.986*Team1Off[i]/LeagueOE*.986*Team2Def[i])*GameTempo/100
        Team2Score=(Team2Off[i]/LeagueOE*1.014*Team1Def[i]*1.014)*GameTempo/100
        Pointresult[i]=Team1Score+Team2Score
        Spreadresult[i]=Team1Score-Team2Score
    return Pointresult,Spreadresult

def getGameDistributionNewNeutral(Team1Off,Team1Def,Team2Off,Team2Def,GameTempoN,LeagueOE):
    k=len(Team1Off)
    Spreadresult = [None] * k
    Pointresult= [None] * k
    for i in range(k):
        Team1Score=(Team1Off[i]/LeagueOE*Team2Def[i])*GameTempoN/100
        Team2Score=(Team2Off[i]/LeagueOE*Team1Def[i])*GameTempoN/100
        Pointresult[i]=Team1Score+Team2Score
        Spreadresult[i]=Team1Score-Team2Score
    return Pointresult,Spreadresult




def getGameDistribution(Team1Off,Team1Def,Team2Off,Team2Def,GameTempo):
    k=len(Team1Off)
    Spreadresult = [None] * k
    Pointresult= [None] * k
    for i in range(k):
        Team1Score=(Team1Off[i]*Team2Def[i])*GameTempo/10000
        Team2Score=(Team2Off[i]*Team1Def[i])*GameTempo/10000
        Pointresult[i]=Team1Score+Team2Score
        Spreadresult[i]=Team1Score-Team2Score
    return Pointresult,Spreadresult


def GetVegasProjectedScore(ProjectedTotal,ProjectedSpread):
    HomeVScore=(ProjectedTotal-ProjectedSpread)/2
    AwayVScore=(HomeVScore+ProjectedSpread)
    return AwayVScore,HomeVScore

def getMonteCarloGameScore(AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    w=AwayTeamData
    TexasO=list(w["AdjO"])
    TexasD=list(w["AdjD"])
    w1=HomeTeamData
    StO=list(w1["AdjO"])
    StD=list(w1["AdjD"])
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNew(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread


def getMonteCarloGameScoreNeutralCourt(AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    w=AwayTeamData
    TexasO=list(w["AdjO"])
    TexasD=list(w["AdjD"])
    w1=HomeTeamData
    StO=list(w1["AdjO"])
    StD=list(w1["AdjD"])
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNewNeutral(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread


def GetVegasProjectedScore(ProjectedTotal,ProjectedSpread):
    HomeVScore=(ProjectedTotal-ProjectedSpread)/2
    AwayVScore=(HomeVScore+ProjectedSpread)
    return AwayVScore,HomeVScore

def SetBartData(BartDF,AwayTeamB1,HomeTeamB1,TheTempo):
    


 
    Team1AdjOff=BartDF.loc[AwayTeamB1,"AdjOE"]
    Team2AdjOff=BartDF.loc[HomeTeamB1,"AdjOE"]
    
    #Team1AdjT=BartDF.loc[AwayTeamB1,"ADJT"]
    #Team2AdjT=BartDF.loc[HomeTeamB1,"ADJT"]
    
    Team1AdjDef=BartDF.loc[AwayTeamB1,"AdjDE"]
    Team2AdjDef=BartDF.loc[HomeTeamB1,"AdjDE"]
    
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,TheTempo,LeagueOE)
    return BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo

def SetBartDataNeutral(BartDF,AwayTeamB1,HomeTeamB1,TheTempo):
    


 
    Team1AdjOff=BartDF.loc[AwayTeamB1,"AdjOE"]
    Team2AdjOff=BartDF.loc[HomeTeamB1,"AdjOE"]
    
        
    #Team1AdjT=BartDF.loc[AwayTeamB1,"ADJT"]
    #Team2AdjT=BartDF.loc[HomeTeamB1,"ADJT"]

    Team1AdjDef=BartDF.loc[AwayTeamB1,"AdjDE"]
    Team2AdjDef=BartDF.loc[HomeTeamB1,"AdjDE"]
    
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,TheTempo,LeagueOE)
    return BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo

def SetPomeroyData(PomeroyDF,AwayTeamP1,HomeTeamP1):
    
    PomeroyDF['AdjEM'] = PomeroyDF.AdjEM.apply(calculate_to_numeric)
    PomeroyDF['AdjO'] = PomeroyDF.AdjO.apply(calculate_to_numeric)
    PomeroyDF['AdjORank'] = PomeroyDF.AdjORank.apply(calculate_to_numeric)
    PomeroyDF['AdjD'] = PomeroyDF.AdjD.apply(calculate_to_numeric)
    PomeroyDF['AdjDRank'] = PomeroyDF.AdjDRank.apply(calculate_to_numeric)
    PomeroyDF['AdjT'] = PomeroyDF.AdjT.apply(calculate_to_numeric)
    PomeroyDF['AdjTRank'] = PomeroyDF.AdjTRank.apply(calculate_to_numeric)


    Team1AdjTempo=PomeroyDF.loc[AwayTeamP1,"AdjT"]
    Team2AdjTempo=PomeroyDF.loc[HomeTeamP1,"AdjT"]
    Team1AdjOff=PomeroyDF.loc[AwayTeamP1,"AdjO"]
    Team2AdjOff=PomeroyDF.loc[HomeTeamP1,"AdjO"]

    Team1AdjDef=PomeroyDF.loc[AwayTeamP1,"AdjD"]
    Team2AdjDef=PomeroyDF.loc[HomeTeamP1,"AdjD"]
    PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE)
    return PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo

def SetPomeroyDataNeutral(PomeroyDF,AwayTeamP1,HomeTeamP1):
    
    PomeroyDF['AdjEM'] = PomeroyDF.AdjEM.apply(calculate_to_numeric)
    PomeroyDF['AdjO'] = PomeroyDF.AdjO.apply(calculate_to_numeric)
    PomeroyDF['AdjORank'] = PomeroyDF.AdjORank.apply(calculate_to_numeric)
    PomeroyDF['AdjD'] = PomeroyDF.AdjD.apply(calculate_to_numeric)
    PomeroyDF['AdjDRank'] = PomeroyDF.AdjDRank.apply(calculate_to_numeric)
    PomeroyDF['AdjT'] = PomeroyDF.AdjT.apply(calculate_to_numeric)
    PomeroyDF['AdjTRank'] = PomeroyDF.AdjTRank.apply(calculate_to_numeric)


    Team1AdjTempo=PomeroyDF.loc[AwayTeamP1,"AdjT"]
    Team2AdjTempo=PomeroyDF.loc[HomeTeamP1,"AdjT"]
    Team1AdjOff=PomeroyDF.loc[AwayTeamP1,"AdjO"]
    Team2AdjOff=PomeroyDF.loc[HomeTeamP1,"AdjO"]

    Team1AdjDef=PomeroyDF.loc[AwayTeamP1,"AdjD"]
    Team2AdjDef=PomeroyDF.loc[HomeTeamP1,"AdjD"]
    PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE)
    return PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo



def SetPomeroyData2020(PomeroyDF,AwayTeamP1,HomeTeamP1):
    
    PomeroyDF['AdjEM'] = PomeroyDF.AdjEM.apply(calculate_to_numeric)
    PomeroyDF['AdjO'] = PomeroyDF.AdjO.apply(calculate_to_numeric)
    PomeroyDF['AdjORank'] = PomeroyDF.AdjO_Rank.apply(calculate_to_numeric)
    PomeroyDF['AdjD'] = PomeroyDF.AdjD.apply(calculate_to_numeric)
    PomeroyDF['AdjDRank'] = PomeroyDF.AdjD_Rank.apply(calculate_to_numeric)
    PomeroyDF['AdjT'] = PomeroyDF.AdjT.apply(calculate_to_numeric)
    PomeroyDF['AdjTRank'] = PomeroyDF.AdjT_Rank.apply(calculate_to_numeric)


    Team1AdjTempo=PomeroyDF.loc[AwayTeamP1,"AdjT"]
    Team2AdjTempo=PomeroyDF.loc[HomeTeamP1,"AdjT"]
    Team1AdjOff=PomeroyDF.loc[AwayTeamP1,"AdjO"]
    Team2AdjOff=PomeroyDF.loc[HomeTeamP1,"AdjO"]

    Team1AdjDef=PomeroyDF.loc[AwayTeamP1,"AdjD"]
    Team2AdjDef=PomeroyDF.loc[HomeTeamP1,"AdjD"]
    PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE)
    return PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo

def SetPomeroyDataNeutral2020(PomeroyDF,AwayTeamP1,HomeTeamP1):
    
    PomeroyDF['AdjEM'] = PomeroyDF.AdjEM.apply(calculate_to_numeric)
    PomeroyDF['AdjO'] = PomeroyDF.AdjO.apply(calculate_to_numeric)
    PomeroyDF['AdjORank'] = PomeroyDF.AdjO_Rank.apply(calculate_to_numeric)
    PomeroyDF['AdjD'] = PomeroyDF.AdjD.apply(calculate_to_numeric)
    PomeroyDF['AdjDRank'] = PomeroyDF.AdjD_Rank.apply(calculate_to_numeric)
    PomeroyDF['AdjT'] = PomeroyDF.AdjT.apply(calculate_to_numeric)
    PomeroyDF['AdjTRank'] = PomeroyDF.AdjT_Rank.apply(calculate_to_numeric)


    Team1AdjTempo=PomeroyDF.loc[AwayTeamP1,"AdjT"]
    Team2AdjTempo=PomeroyDF.loc[HomeTeamP1,"AdjT"]
    Team1AdjOff=PomeroyDF.loc[AwayTeamP1,"AdjO"]
    Team2AdjOff=PomeroyDF.loc[HomeTeamP1,"AdjO"]

    Team1AdjDef=PomeroyDF.loc[AwayTeamP1,"AdjD"]
    Team2AdjDef=PomeroyDF.loc[HomeTeamP1,"AdjD"]
    PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE)
    return PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo
def update_type(t1, t2, dropna=False):
    return t1.map(t2).dropna() if dropna else t1.map(t2).fillna(t1)

def getBartDataTest():
    # need to check year
    res = requests.get("http://www.barttorvik.com/trank.php?year=2019&sort=&conlimit=")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table') [0]
    df = pd.read_html(str(table))[0]
    newCName=["Rank","Team","Conference","Record","AdjOE","OERank","AdjDE","DERank","BARTH","PRecord","CRecord","EFG%","EGO Rank","EFG%D","EGD Rank","TO%","TO Rank","TO%D","TOD Rank","OR%","OR Rank","OR%D%","ORD Rank","FT%","FT Rank","FT%D","FTD Rank","FTR%1","2P1","ADJT","WAB"]

    df.columns=newCName
    BartRankings=df[["Rank","Team","Record","AdjOE","OERank","AdjDE","DERank","BARTH","EFG%","EGO Rank","EFG%D","EGD Rank","TO%","TO Rank","TO%D","TOD Rank","OR%","OR Rank","OR%D%","ORD Rank","FT%","FT Rank","FT%D","ADJT"]].copy()

    #BartRankings.set_index("Team", inplace=True)

    BartRankings['AdjOE'] = BartRankings.AdjOE.apply(calculate_to_numeric)
    BartRankings['OERank'] = BartRankings.OERank.apply(calculate_to_numeric)
    BartRankings['AdjDE'] = BartRankings.AdjDE.apply(calculate_to_numeric)
    BartRankings['DERank'] = BartRankings.DERank.apply(calculate_to_numeric)
    #BartRankings['ADJT'] = BartRankings.ADJT.apply(calculate_to_numeric)
    BartRankings['BARTH'] = BartRankings.BARTH.apply(calculate_to_numeric)

    return BartRankings


def sanitize_teamname(name):
    """
    At the end of the year, teams are listed as 'Florida 3' for
    example, to denote the fact that Florida is a 3 seed. We obviously
    don't want the 3.
    """
    endings = [' %s' % (i+1) for i in range(16)]
    endings.reverse()
    for ending in endings:
        if name.endswith(ending):
            name = name[:-len(ending)]
    return name


def NewgetGamePredictionBartCurrent(Team1AdjTempo,Team1AdjOff,Team1AdjDef,Team2AdjTempo,Team2AdjOff,Team2AdjDef,LeagueOE,LeagueTempo):

    TheTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo

    GameTempo=TheTempo+2
    
    Team1Score=(.986*Team1AdjOff/LeagueOE*.986*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*1.014*Team1AdjDef*1.014)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return Team1Score,Team2Score,OverUnder,PointDiff,GameTempo

def NewgetGamePredictionBartNeutralCourtCurrent(Team1AdjTempo,Team1AdjOff,Team1AdjDef,Team2AdjTempo,Team2AdjOff,Team2AdjDef,LeagueOE,LeagueTempo):

    TheTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo 
    GameTempo=TheTempo+2
    
    Team1Score=(Team1AdjOff/LeagueOE*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*Team1AdjDef)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return Team1Score,Team2Score,OverUnder,PointDiff,GameTempo

def SetBartDataCurrent(BartDF,AwayTeamB1,HomeTeamB1,TheTempo):
    


 
    Team1AdjOff=BartDF.loc[AwayTeamB1,"AdjOE"]
    Team2AdjOff=BartDF.loc[HomeTeamB1,"AdjOE"]

    Team1AdjDef=BartDF.loc[AwayTeamB1,"AdjDE"]
    Team2AdjDef=BartDF.loc[HomeTeamB1,"AdjDE"]
    
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(TheTempo,Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,LeagueOE)
    return BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo

def SetBartDataCurrentNeutral(BartDF,AwayTeamB1,HomeTeamB1,TheTempo):
    


 
    Team1AdjOff=BartDF.loc[AwayTeamB1,"AdjOE"]
    Team2AdjOff=BartDF.loc[HomeTeamB1,"AdjOE"]

    Team1AdjDef=BartDF.loc[AwayTeamB1,"AdjDE"]
    Team2AdjDef=BartDF.loc[HomeTeamB1,"AdjDE"]
    
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(TheTempo,Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,LeagueOE)
    return BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo


def AddSystemWinLoss(hh):
    hh["MaxSpread"]=hh[['TCurSpread','T3GSpread','MC5Spread','MC10Spread']].max(axis=1)
    hh["MinSpread"]=hh[['TCurSpread','T3GSpread','MC5Spread','MC10Spread']].min(axis=1)
    hh["MaxTotal"]=hh[['TCurTotal','T3GTotal','MC5Total','MC10Total']].max(axis=1)
    hh["MinTotal"]=hh[['TCurTotal','T3GTotal','MC5Total','MC10Total']].min(axis=1)
    hh["SystemSpread"]=0
    hh["SystemTotal"]=0
    for i in range(len(hh.index)):
        if hh.loc[i,"ATSVegas"] < hh.loc[i,"MinSpread"]:
            if hh.loc[i,"ATS"]>0:
                hh.loc[i,"SystemSpread"]=-1
            else:
                hh.loc[i,"SystemSpread"]=1
        if hh.loc[i,"ATSVegas"] > hh.loc[i,"MaxSpread"]:
            if hh.loc[i,"ATS"]>=0:
                hh.loc[i,"SystemSpread"]=1
            else:
                hh.loc[i,"SystemSpread"]=-1
                
                
        if hh.loc[i,"OverUnderVegas"] < hh.loc[i,"MinTotal"]:
            if hh.loc[i,"OverUnder"]>0:
                hh.loc[i,"SystemTotal"]=1
            else:
                hh.loc[i,"SystemTotal"]=-1
        if hh.loc[i,"OverUnderVegas"] > hh.loc[i,"MaxTotal"]:
            if hh.loc[i,"OverUnder"]>=0:
                hh.loc[i,"SystemTotal"]=-1
            else:
                hh.loc[i,"SystemTotal"]=1
    return(hh)

def AddSystemWinLossDifferentColumns(hh):
    hh["MaxSpread"]=hh[['TCurSpread','T3GSpread','MC5Spread','MC10Spread']].max(axis=1)
    hh["MinSpread"]=hh[['TCurSpread','T3GSpread','MC5Spread','MC10Spread']].min(axis=1)
    hh["MaxTotal"]=hh[['TCurTotal','T3GTotal','MC5Total','MC10Total']].max(axis=1)
    hh["MinTotal"]=hh[['TCurTotal','T3GTotal','MC5Total','MC10Total']].min(axis=1)
    hh["SystemSpreadWin"]=0
    hh["SystemSpreadLoss"]=0
    hh["SystemTotalWin"]=0
    hh["SystemTotalLoss"]=0
    for i in range(len(hh.index)):
        if hh.loc[i,"ATSVegas"] < hh.loc[i,"MinSpread"]:
            if hh.loc[i,"ATS"]>0:
                hh.loc[i,"SystemSpreadLoss"]=1
            else:
                hh.loc[i,"SystemSpreadWin"]=1
        if hh.loc[i,"ATSVegas"] > hh.loc[i,"MaxSpread"]:
            if hh.loc[i,"ATS"]>=0:
                hh.loc[i,"SystemSpreadWin"]=1
            else:
                hh.loc[i,"SystemSpreadLoss"]=1
                
                
        if hh.loc[i,"OverUnderVegas"] < hh.loc[i,"MinTotal"]:
            if hh.loc[i,"OverUnder"]>=0:
                hh.loc[i,"SystemTotalWin"]=1
            else:
                hh.loc[i,"SystemTotalLoss"]=1
        if hh.loc[i,"OverUnderVegas"] > hh.loc[i,"MaxTotal"]:
            if hh.loc[i,"OverUnder"]>0:
                hh.loc[i,"SystemTotalLoss"]=1
            else:
                hh.loc[i,"SystemTotalWin"]=1
    return(hh)


def GetThisTeamInfoFromCsv(ThisTeam,WhichFile):


    TeamInfo=pd.read_csv("C:/Users/mpgen/"+WhichFile+"/"+ThisTeam+"Data.csv")
    return(TeamInfo)


def AddWinLossAgainstSpread(hh,ColumntoTestSpread,ColumntoTestTotal):

    SpreadName=ColumntoTestSpread+"ATS"
    
    TotalName=ColumntoTestTotal+"Total"
    
    hh[SpreadName]=0
    hh[TotalName]=0
    for i in range(len(hh.index)):
        if hh.loc[i,"ATSVegas"] < hh.loc[i,ColumntoTestSpread]:
            if hh.loc[i,"ATS"]>0:
                hh.loc[i,SpreadName]=-1
            else:
                hh.loc[i,SpreadName]=1
        else: 
            if hh.loc[i,"ATS"]>=0:
                hh.loc[i,SpreadName]=1
            else:
                hh.loc[i,SpreadName]=-1
                
        if hh.loc[i,"OverUnderVegas"] > hh.loc[i,ColumntoTestTotal]:
            if hh.loc[i,"OverUnder"]>=0:
                hh.loc[i,TotalName]=-1
            else:
                hh.loc[i,TotalName]=1
        else:
            if hh.loc[i,"OverUnder"]>0:
                hh.loc[i,TotalName]=1
            else:
                hh.loc[i,TotalName]=-1
                
                
    return(hh)


def AddWinLossAgainstSpreadDifferentColumns(hh,ColumntoTestSpread,ColumntoTestTotal):

    SpreadNameWin=ColumntoTestSpread+"WinATS"
    SpreadNameLoss=ColumntoTestSpread+"LossATS"
    TotalNameWin=ColumntoTestTotal+"WinTotal"
    TotalNameLoss=ColumntoTestTotal+"LossTotal"
    
    hh[SpreadNameWin]=0
    hh[SpreadNameLoss]=0
    hh[TotalNameWin]=0
    hh[TotalNameLoss]=0
    
    for i in range(len(hh.index)):
        if hh.loc[i,"ATSVegas"] < hh.loc[i,ColumntoTestSpread]:
            if hh.loc[i,"ATS"]>0:
                hh.loc[i,SpreadNameLoss]=1
            else:
                hh.loc[i,SpreadNameWin]=1
        else: 
            if hh.loc[i,"ATS"]>=0:
                hh.loc[i,SpreadNameWin]=1
            else:
                hh.loc[i,SpreadNameLoss]=1
                
        if hh.loc[i,"OverUnderVegas"] > hh.loc[i,ColumntoTestTotal]:
            if hh.loc[i,"OverUnder"]>=0:
                hh.loc[i,TotalNameLoss]=1
            else:
                hh.loc[i,TotalNameWin]=1
        else:
            if hh.loc[i,"OverUnder"]>0:
                hh.loc[i,TotalNameWin]=1
            else:
                hh.loc[i,TotalNameLoss]=1
                

    
    
    return(hh)


def AddSimpleWinLossATS(hh,ColumntoTest,ColumntoTestAgainst):

    SpreadNameWin=ColumntoTest+"ATSWin"
    SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    hh[SpreadNameWin]=0
    hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(hh.index)):
        if hh.loc[i,ColumntoTest] > hh.loc[i,ColumntoTestAgainst]:
            if hh.loc[i,"ATS"]>=0:
                hh.loc[i,SpreadNameWin]=1
            else:
                hh.loc[i,SpreadNameLoss]=1
        else: 
            if hh.loc[i,"ATS"]>0:
                hh.loc[i,SpreadNameLoss]=1
            else:
                hh.loc[i,SpreadNameWin]=1
                
       
    numberofWins= hh[SpreadNameWin].sum() 
    numberofLosses= hh[SpreadNameLoss].sum()
    return(hh,numberofWins,numberofLosses)



def GetOneTeamChartwithColumns(df5,WhatTeam):
    df5["EMRating 2 Game"]=df5["EMRating"].ewm(span=2,adjust=False).mean()
    df5["EMRating 3 Game"]=df5["EMRating"].ewm(span=3,adjust=False).mean()
    df5["EMRating 5 Game"]=df5["EMRating"].ewm(span=5,adjust=False).mean()
    df5["EMRating 5 GameShift"]=df5["EMRating 5 Game"].shift(1).fillna(0)
    ChartTitleName=WhatTeam+" EMRating and ATS"
    plt.title(ChartTitleName)
    plt.scatter(df5.index,df5["EMRating"])
    df5["AdjEMCurrent"].plot()
    df5["EMRating 5 Game"].plot()
    df5["EMRating 5 GameShift"].plot()
    #df5["EMRating 2 Game"].plot()
    df5.index,df5["ATS"].plot(kind='bar')
  
    plt.show()

    
def AddWinLossEMAOver(hh,ColumntoTestSpread,ColumnHurdle):
    SpreadName=ColumntoTestSpread+"ATS"
    hh[SpreadName]=0
    for i in range(len(hh.index)):
        if hh.loc[i,ColumntoTestSpread] > hh.loc[i,ColumnHurdle]:
            if hh.loc[i,"ATS"]>=0:
                hh.loc[i,SpreadName]=1
            else:
                hh.loc[i,SpreadName]=-1
        else:
            if hh.loc[i,"ATS"]>0:
                hh.loc[i,SpreadName]=-1
            else:
                hh.loc[i,SpreadName]=1
    return(hh)


def sanitizeEntireColumn(df,ColumntoSanitize):

    for i in range(len(df.index)):
        df.loc[i,ColumntoSanitize]=sanitize_teamname(df.loc[i,ColumntoSanitize])
    return(df)

def GetBracketMatrix():
    BracketLookup="http://bracketmatrix.com"
    res = requests.get(BracketLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))[0]
    df1=df.iloc[8:, 0:4]
    return(df1)

def GetBracketMatrixRegionals():
    BracketLookup="http://www.gadepool.com/bracketology.html"
    res = requests.get(BracketLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))[0]
    #df1=df.iloc[8:, 0:4]
    return(df)

#doesnt work
def GetVegasOdds():
    BracketLookup="http://www.vegasinsider.com/college-basketball/odds/futures/"
    res = requests.get(BracketLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))[0]
    #df1=df.iloc[8:, 0:4]
    return(df)

def getScorefromString(testString):
    highScore=testString.split('-')[0]
    if "\xf0" in testString.split('-')[1][:3]:
        lowScore=testString.split('-')[1][:2]
    else:
        lowScore=testString.split('-')[1]
    #lowScore=testString.split('-')[1]
    totalScore=int(highScore)+int(lowScore)
    scoreDiff=int(highScore)-int(lowScore)
    return (int(highScore),int(lowScore),totalScore,scoreDiff)


dateCreator = {"Nov": '11',"Dec":'12',"Jan":'01',"Feb":'02',"Mar":'03',"Apr":'04'}


def getGameInfoFromScheduleFixedMay7(dftobreak):
    if int(dftobreak[0].split()[1])>9:
        theDayNumber=dftobreak[0].split()[1]
    else:
        theDayNumber='0'+dftobreak[0].split()[1]
    if int(dateCreator[dftobreak[0].split()[0]])<5:
        thisDate='2018'+dateCreator[dftobreak[0].split()[0]]+theDayNumber
        #thisDate='2018'+dateCreator[dftobreak[0].split()[0]]+dftobreak[0].split()[1]
    else:
        thisDate='2017'+dateCreator[dftobreak[0].split()[0]]+theDayNumber
        #thisDate='2017'+dateCreator[dftobreak[0].split()[0]]+dftobreak[0].split()[1]
    print(thisDate)
    
    if " at " in dftobreak[1]:
        whereisGame='H'
        if "-" in dftobreak[1]:
            #takeout=" "+dftobreak[1].split(' at ')[1].split()[0]+" "
            #newchar=dftobreak[1].split('at')[1].replace(takeout,"")

            #newtakeout=" "+newchar.split(' ')[len(newchar.split())-1]
            #newhometeam=newchar.replace(newtakeout,"")
            #newawaytakeout=dftobreak[1].split('at')[0].split()[0]+" "
            #newawayteam=dftobreak[1].split('at')[0].replace(newawaytakeout,"")[:-1]
            
            rightside=dftobreak[1].split(' at ')[1]
            leftside=dftobreak[1].split(' at ')[0]
            ranking=leftside.split()[0]+" "
            newawayteam=leftside.replace(ranking,"")
            #ranking
            ranking=rightside.split()[0]+" "
            newrightside=rightside.replace(ranking,"")
            #newhometeam=newrightside.split(' ')[0]
            newh=" "+newrightside.split(' ')[len(newrightside.split())-1]
            newhometeam=newrightside.replace(newh,"")
            #takeout=" "+dftobreak1[1].split(' at ')[1].split()[0]+" "
            #newchar=dftobreak1[1].split('at')[1].replace(takeout,"")

            #newtakeout=" "+newchar.split(' ')[len(newchar.split())-1]
            #newhometeam=newchar.replace(newtakeout,"")
            #newawaytakeout=dftobreak1[1].split('at')[0].split()[0]+" "
            #newawayteam=dftobreak1[1].split('at')[0].replace(newawaytakeout,"")[:-1]
            
            
            
        else:
            takeout=dftobreak[1].split(' at ')[1].split()[0]+" "
            newhometeam=dftobreak[1].split(' at ')[1].replace(takeout,"")
            newtakeout=dftobreak[1].split(' at ')[0].split()[0]+" "
            newawayteam=dftobreak[1].split(' at ')[0].replace(newtakeout,"")
    else:
        whereisGame='N'
        if "-" in dftobreak[1]:
            takeout=" "+dftobreak[1].split('vs')[1].split()[0]+" "
            newchar=dftobreak[1].split('vs')[1].replace(takeout,"")

            newtakeout=" "+newchar.split(' ')[len(newchar.split())-1]
            newhometeam=newchar.replace(newtakeout,"")
            newawaytakeout=dftobreak[1].split('vs')[0].split()[0]+" "
            newawayteam=dftobreak[1].split('vs')[0].replace(newawaytakeout,"")[:-1]
        else:
            takeout=dftobreak[1].split(' vs ')[1].split()[0]+" "
            newhometeam=dftobreak[1].split(' vs ')[1].replace(takeout,"")
            newtakeout=dftobreak[1].split(' vs ')[0].split()[0]+" "
            newawayteam=dftobreak[1].split(' vs ')[0].replace(newtakeout,"")


    actualwinner=dftobreak[4].split(',')[0]
    actualscore=dftobreak[4].split(', ')[1]
    #theVegasSpreadData,theVegasOverData=getVegasFromCsv(newhometeam,thisDate)
    
    projectedspread=dftobreak[2].split(', ')[0].split()[(len(dftobreak[2].split(', ')[0].split())-1)]

    favorite=dftobreak[2].split(', ')[0].replace(projectedspread,"")[:-1]
    projectedscore=dftobreak[2].split(', ')[1].split()[0]
    projectedwinpercent=dftobreak[2].split(', ')[1].split()[1][1:-1]
    projectedlosspercent=str(100-int(dftobreak[2].split(', ')[1].split()[1][1:-2]))+"%"
    theVegasSpreadData,theVegasOverData,theTG3Spread,theTG3Over,theMC5Spread,theMC5Over,theMC10Spread,theMC10Over,HomeTeamHot,AwayTeamHot=getDataFromCsvforGameday(newhometeam,newawayteam,thisDate)
    
    #else:
        #theVegasSpreadData,theVegasOverData,theTG3Spread,theTG3Over,theMC5Spread,theMC5Over,theMC10Spread,theMC10Over,HomeTeamHot,AwayTeamHot=0
    #    projectedsprerad,favorite,projectedwinpwercent,projectedlosspercent=0
    if actualwinner==newhometeam:
        hometeamscore,awayteamscore,actualTotal,actualDiff=getScorefromString(actualscore)
        actualDiff=actualDiff*-1
    else:
        awayteamscore,hometeamscore,actualTotal,actualDiff=getScorefromString(actualscore)
        
    if favorite==newhometeam:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff=getScorefromString(projectedscore)
        projectedDiff=projectedDiff*-1
        hometeampercent=projectedwinpercent
        awayteampercent=projectedlosspercent
    else:
        projectedawayteamscore,projectedhometeamscore,projectedTotal,projectedDiff=getScorefromString(projectedscore)
        hometeampercent=projectedlosspercent
        awayteampercent=projectedwinpercent
    #gameinfo=(thisDate,whereisGame,newawayteam,newhometeam,favorite,projectedspread,actualTotal,actualDiff,projectedscore,projectedwinpercent,projectedlosspercent,actualwinner,actualscore)
    j=pd.DataFrame.from_items([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('Trend', [AwayTeamHot,HomeTeamHot]),('Vegas', [theVegasOverData,theVegasSpreadData]),('Actual', [actualTotal, actualDiff]),('TRank', [projectedTotal, projectedDiff]),('TG3', [theTG3Over, theTG3Spread]),('MC5', [theMC5Over, theMC5Spread]),('MC10', [theMC10Over, theMC10Spread]),('Final', [awayteamscore,hometeamscore]),('Win%', [awayteampercent, hometeampercent])])
    #j1=pd.DataFrame.from_items([('Teams', [AwayTeam, HomeTeam]), ('VScore', [AwayVScore, HomeVScore]),('PomScore', [round(PAwayTeamScore,2), round(PHomeTeamScore,2)]),('TRScore', [round(BAwayTeamScore,2), round(BHomeTeamScore,2)]),('3GScore', [round(B3GAwayTeamScore,2), round(B3GHomeTeamScore,2)]),('Edge5', [edgeAgainstVegasTotal,edgeAgainstVegasSpread ]),('Edge10', [edgeAgainstVegasTotal10G,edgeAgainstVegasSpread10G ])])

    gameinfo=(thisDate,whereisGame,newawayteam,newhometeam,favorite,projectedspread,hometeamscore,awayteamscore,actualTotal,actualDiff,projectedscore,projectedTotal,projectedDiff,projectedwinpercent,projectedlosspercent,actualwinner,actualscore)
    return(j)

def getGameInfoFromSchedule(dftobreak):
    
    
    newtakeout=dftobreak[1].split('vs')[0].split()[0]+" "
    takeout=" "+dftobreak[1].split('vs')[1].replace(" NCAA-T", "").split()[0]+" "
    newawayteam=dftobreak[1].split('vs')[0].replace(newtakeout,"")[:-1]
    newhometeam=dftobreak[1].split('vs')[1].replace(" NCAA-T", "").replace(takeout,"")
    projectedspread=dftobreak[2].split(', ')[0].split()[(len(dftobreak[2].split(', ')[0].split())-1)]

    favorite=dftobreak[2].split(', ')[0].replace(projectedspread,"")[:-1]
    projectedscore=dftobreak[2].split(', ')[1].split()[0]
    projectedwinpercent=dftobreak[2].split(', ')[1].split()[1]
    actualwinner=dftobreak[4].split(',')[0]
    actualscore=dftobreak[4].split(',')[1]
    gameinfo=(newawayteam,newhometeam,favorite,projectedspread,projectedscore,projectedwinpercent,actualwinner,actualscore)
    return(gameinfo)

def convertFullDatetoShortDate(theDate):
    #if int(theDate[6:8])>9:
    #    newDate=theDate[4:6]+"-"+theDate[6:8]
    #else:
    #    newDate=theDate[4:6]+"-"+theDate[7:8]
    newDate=theDate[4:6]+"-"+theDate[6:8]    
    return(newDate)


#WhichFile='TeamDataFilesCompleteApr17'
def getVegasFromCsv(WhichTeam,WhichDate):
    df=GetThisTeamInfoFromCsv(WhichTeam,WhichFile)
    newDate=convertFullDatetoShortDate(WhichDate)
    ATSVegasis=df.loc[df['Date'] == newDate,"ATSVegas"].values[0]
    OUVegasis=df.loc[df['Date'] == newDate,"OverUnderVegas"].values[0]
    return(ATSVegasis,OUVegasis)
#WhichFile='TeamDataFilesEverything'
def getDataFromCsvforGameday(WhichTeam,OtherTeam,WhichDate):
    df=GetThisTeamInfoFromCsv(WhichTeam,WhichFile)
    dfOther=GetThisTeamInfoFromCsv(OtherTeam,WhichFile)
    newDate=convertFullDatetoShortDate(WhichDate)
    ATSVegasis=df.loc[df['Date'] == newDate,"ATSVegas"].values[0]
    OUVegasis=df.loc[df['Date'] == newDate,"OverUnderVegas"].values[0]
    TG3ATSis=df.loc[df['Date'] == newDate,"T3GSpread"].values[0]
    TG3OUis=df.loc[df['Date'] == newDate,"T3GTotal"].values[0]
    MC5ATSis=df.loc[df['Date'] == newDate,"MC5Spread"].values[0]
    MC5OUis=df.loc[df['Date'] == newDate,"MC5Total"].values[0]
    MC10ATSis=df.loc[df['Date'] == newDate,"MC10Spread"].values[0]
    MC10OUis=df.loc[df['Date'] == newDate,"MC10Total"].values[0]
    HomeTeamTrending=df.loc[df['Date'] == newDate,"DifCumSumFastEMA"].values[0]-df.loc[df['Date'] == newDate,"DifCumSumEMA"].values[0]
    AwayTeamTrending=dfOther.loc[dfOther['Date'] == newDate,"DifCumSumFastEMA"].values[0]-dfOther.loc[dfOther['Date'] == newDate,"DifCumSumEMA"].values[0]

    #if HomeTeamTrending > 0:
    #    HomeTeamisHot=True
    #else:
    #    HomeTeamisHot=False
        
    return(ATSVegasis,OUVegasis,TG3ATSis,TG3OUis,MC5ATSis,MC5OUis,MC10ATSis,MC10OUis,HomeTeamTrending,AwayTeamTrending)

def CalculateBrierScore(probArray,OutcomeArray):
    DiffArray=probArray-OutcomeArray # subtract prob from outcome
    SquaredArray=np.power(DiffArray,2)  # square the array
    theBrierScore=SquaredArray.sum()/len(probArray)  # add up results and divide by length of array
    return(theBrierScore)
import math
def kenpom_prob(point_spread, std=10.5):
    """Calculate team win probability using KenPom predicted point spread."""
    return 0.5 * (1 + math.erf((point_spread) / (std*math.sqrt(2))))

def GetWinLossPercentGameWinner(whichList,ColumntoTest,ColumntoTestAgainst):
    
    #SpreadNameWin=ColumntoTest+"ATSWin"
    #SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    winLossList=[]
    PercentList=[]
    #hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(whichList)):
        PercentList.append(kenpom_prob(abs(whichList[i][ColumntoTest][1]), std=10.5))
        if whichList[i][ColumntoTestAgainst][1] < 0:
            
            if whichList[i][ColumntoTest][1]>=0:
                winLossList.append(0)
            else:
                winLossList.append(1)
        else: 
            if whichList[i][ColumntoTest][1]>0:
                winLossList.append(1)
            else:
                winLossList.append(0)
                
    #print(winLossList)
    #print(PercentList)
    BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,BrierScoreToday)

def createSeabornChartPandas(DfNew):
    gg1=[DfNew["TRank"][1],DfNew["TG3"][1],DfNew["MC5"][1],DfNew["MC10"][1]]
    thenewDF=pd.DataFrame({"Spreads":gg1})

    thenewDF["Teams"]=DfNew["Teams"][1]
    thenewDF["Vegas"]=DfNew["Vegas"][1]
    thenewDF["Actual"]=DfNew["Actual"][1]
    return(thenewDF)

def createSeabornChartPandasOverUnder(DfNew):
    gg1=[DfNew["TRank"][0],DfNew["TG3"][0],DfNew["MC5"][0],DfNew["MC10"][0]]
    thenewDF=pd.DataFrame({"Totals":gg1})

    thenewDF["Teams"]=DfNew["Teams"][1]
    thenewDF["Vegas"]=DfNew["Vegas"][0]
    thenewDF["Actual"]=DfNew["Actual"][0]
    return(thenewDF)





def AddDateandPaceTestingJan2(DftoChange):
    DftoChange["DateNew"]=""
    #DftoChange["Pace"]=0
    for i in range(len(DftoChange.index)):
        #DftoChange.loc[i,"Pace"]=int(DftoChange.loc[i,"Result"][-3:-2]+DftoChange.loc[i,"Result"][-2:-1])
        
        if int(DftoChange["Date"][i][-5:][:2])>5:
            DftoChange.loc[i,"DateNew"]="2018"+DftoChange.loc[i,"Date"][-5:][:2]+DftoChange.loc[i,"Date"][-5:][3:5]
        else:
            DftoChange.loc[i,"DateNew"]="2019"+"0"+DftoChange.loc[i,"Date"][-4:][:1]+DftoChange.loc[i,"Date"][-4:][2:4]
    return DftoChange
def AddDateandPaceTestingJan2_2020(DftoChange):
    DftoChange["DateNew"]=""
    #DftoChange["Pace"]=0
    for i in range(len(DftoChange.index)):
        #DftoChange.loc[i,"Pace"]=int(DftoChange.loc[i,"Result"][-3:-2]+DftoChange.loc[i,"Result"][-2:-1])
        print(DftoChange["Date"][i][-5:][:2])
        if int(DftoChange["Date"][i][-5:][:2])>5:
            DftoChange.loc[i,"DateNew"]="2019"+DftoChange.loc[i,"Date"][-5:][:2]+DftoChange.loc[i,"Date"][-5:][3:5]
        else:
            DftoChange.loc[i,"DateNew"]="2020"+"0"+DftoChange.loc[i,"Date"][-4:][:1]+DftoChange.loc[i,"Date"][-4:][2:4]
        print(DftoChange.loc[i,"DateNew"])
    return DftoChange

def AddDateandPaceTestingJan2_2021(DftoChange):
    DftoChange["DateNew"]=""
    #DftoChange["Pace"]=0
    for i in range(len(DftoChange.index)):
        #DftoChange.loc[i,"Pace"]=int(DftoChange.loc[i,"Result"][-3:-2]+DftoChange.loc[i,"Result"][-2:-1])
        print(DftoChange["Date"][i][-5:][:2])
        if int(DftoChange["Date"][i][-5:][:2])>5:
            DftoChange.loc[i,"DateNew"]="2020"+DftoChange.loc[i,"Date"][-5:][:2]+DftoChange.loc[i,"Date"][-5:][3:5]
        else:
            DftoChange.loc[i,"DateNew"]="2021"+"0"+DftoChange.loc[i,"Date"][-4:][:1]+DftoChange.loc[i,"Date"][-4:][2:4]
        print(DftoChange.loc[i,"DateNew"])
    return DftoChange
def AddDateandPaceTestingJan2_2022(DftoChange):
    DftoChange["DateNew"]=""
    #DftoChange["Pace"]=0
    for i in range(len(DftoChange.index)):
        #DftoChange.loc[i,"Pace"]=int(DftoChange.loc[i,"Result"][-3:-2]+DftoChange.loc[i,"Result"][-2:-1])
        print(DftoChange["Date"][i][-5:][:2])
        if int(DftoChange["Date"][i][-5:][:2])>5:
            DftoChange.loc[i,"DateNew"]="2021"+DftoChange.loc[i,"Date"][-5:][:2]+DftoChange.loc[i,"Date"][-5:][3:5]
        else:
            DftoChange.loc[i,"DateNew"]="2022"+"0"+DftoChange.loc[i,"Date"][-4:][:1]+DftoChange.loc[i,"Date"][-4:][2:4]
        print(DftoChange.loc[i,"DateNew"])
    return DftoChange



def getTheDayoftheMonth(df1):
    if int(df1)>9:
        theDayNumber=df1
    else:
        theDayNumber='0'+df1
    return theDayNumber


def findHomeTeamandSpreadfromBartSchedule(df1):
#df1=df[0].iloc[0][1]
#u Pass in his string returns split of Away and home team '38 Oklahoma vs 17 Wisconsin'
    if " at " in df1:
        whereisGame='H'
        if "-" in df1:

            
            rightside=df1.split(' at ')[1]
            leftside=df1.split(' at ')[0]
            ranking=leftside.split()[0]+" "
            newawayteam=leftside.replace(ranking,"")

            ranking=rightside.split()[0]+" "
            newrightside=rightside.replace(ranking,"")

            newh=" "+newrightside.split(' ')[len(newrightside.split())-1]
            newhometeam=newrightside.replace(newh,"")

            
            
        else:
            takeout=df1.split(' at ')[1].split()[0]+" "
            newhometeam=df1.split(' at ')[1].replace(takeout,"")
            newtakeout=df1.split(' at ')[0].split()[0]+" "
            newawayteam=df1.split(' at ')[0].replace(newtakeout,"")
    else:
        whereisGame='N'
        if "-" in df1:
            takeout=" "+df1.split('vs')[1].split()[0]+" "
            newchar=df1.split('vs')[1].replace(takeout,"")

            newtakeout=" "+newchar.split(' ')[len(newchar.split())-1]
            newhometeam=newchar.replace(newtakeout,"")
            #newawaytakeout=dftobreak[1].split('vs')[0].split()[0]+" "
            #newawayteam=dftobreak[1].split('vs')[0].replace(newawaytakeout,"")[:-1]
            #newawaytakeout=df1[1].split('vs')[0].split()[0]+" "
            #newawayteam=df1[1].split('vs')[0].replace(newawaytakeout,"")[:-1]
            newawaytakeout=df1.split('vs')[0].split()[0]+" "
            #print(newawaytakeout)
            newawayteam=df1.split('vs')[0].replace(newawaytakeout,"")[:-1]
            
        else:
            takeout=df1.split(' vs ')[1].split()[0]+" "
            newhometeam=df1.split(' vs ')[1].replace(takeout,"")
            newtakeout=df1.split(' vs ')[0].split()[0]+" "
            newawayteam=df1.split(' vs ')[0].replace(newtakeout,"")
    return(newhometeam,newawayteam,whereisGame)
  
#df[0].iloc[0][2]   
def getProjectedSpreadandTotalFromBartSchedule(df1,newhometeam):
    projectedspread=df1.split(', ')[0].split()[(len(df1.split(', ')[0].split())-1)]

    favorite=df1.split(', ')[0].replace(projectedspread,"")[:-1]
    projectedscore=df1.split(', ')[1].split()[0]
    projectedwinpercent=df1.split(', ')[1].split()[1][1:-1]
    projectedlosspercent=str(100-int(df1.split(', ')[1].split()[1][1:-2]))+"%"
    
    if favorite==newhometeam:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff=getScorefromString(projectedscore)
        projectedDiff=projectedDiff
        hometeampercent=projectedwinpercent
        awayteampercent=projectedlosspercent
    else:
        projectedawayteamscore,projectedhometeamscore,projectedTotal,projectedDiff=getScorefromString(projectedscore)
        hometeampercent=projectedlosspercent
        awayteampercent=projectedwinpercent
        projectedDiff=projectedDiff*-1
    return(projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff,hometeampercent,awayteampercent)

def getTheCorrectDateFromStringBartSchedule(df1,theDayNumberToday):
    if int(dateCreator[df1])<5:
        thisDate='2019'+dateCreator[df1]+theDayNumberToday
        #thisDate='2018'+dateCreator[dftobreak[0].split()[0]]+dftobreak[0].split()[1]
    else:
        thisDate='2018'+dateCreator[df1]+theDayNumberToday
    return thisDate
def getTheCorrectDateFromStringBartSchedule2020(df1,theDayNumberToday):
    if int(dateCreator[df1])<5:
        thisDate='2020'+dateCreator[df1]+theDayNumberToday
        #thisDate='2018'+dateCreator[dftobreak[0].split()[0]]+dftobreak[0].split()[1]
    else:
        thisDate='2019'+dateCreator[df1]+theDayNumberToday
    return thisDate

def getTheCorrectDateFromStringBartSchedule2021(df1,theDayNumberToday):
    if int(dateCreator[df1])<5:
        thisDate='2021'+dateCreator[df1]+theDayNumberToday
        #thisDate='2018'+dateCreator[dftobreak[0].split()[0]]+dftobreak[0].split()[1]
    else:
        thisDate='2020'+dateCreator[df1]+theDayNumberToday
    return thisDate


def getInfoForScheduleForCsvDump(theGameDayInfo):
    team1=theGameDayInfo.iloc[0]["Teams"]
    team2=theGameDayInfo.iloc[1]["Teams"]
    theSpread=theGameDayInfo.iloc[1]["TRank"]
    TotalScore=theGameDayInfo.iloc[0]["TRank"]
    Court=theGameDayInfo.iloc[1]["Date"]
    return(team1,team2,theSpread,TotalScore,Court)

def getGameInfoFromBartDailyScheduleNov23(df1,thisDate,gameCounter):
#df[0].iloc[0][0]   
#u'Nov 22 12:30 PM CT'
    #theDateStringFromBart=df1[0].iloc[0][0]
    #theDayNumber=getTheDayoftheMonth(theDateStringFromBart.split()[1])
    #thisDate=getTheCorrectDateFromStringBartSchedule(theDateStringFromBart.split()[0],theDayNumber)
#df[0].iloc[0][1] 
#'38 Oklahoma vs 17 Wisconsin'
    theTeamStringFromBart=df1[0].iloc[0][gameCounter]
    print(theTeamStringFromBart)
    newhometeam,newawayteam,whereisGame=findHomeTeamandSpreadfromBartSchedule(theTeamStringFromBart)
#df[0].iloc[0][2] 
#'Wisconsin -3.4, 74-70 (63%)'
    print(newhometeam,newawayteam,whereisGame)
    thePointSpreadStringFromBart=df1[0].iloc[0][gameCounter+1]
    print(thePointSpreadStringFromBart)
    print(len(thePointSpreadStringFromBart.split()))
    if len(thePointSpreadStringFromBart.split()[len(thePointSpreadStringFromBart.split())-1])<6:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff,hometeampercent,awayteampercent=getProjectedSpreadandTotalFromBartSchedule(thePointSpreadStringFromBart,newhometeam)
    
        j=pd.DataFrame.from_records([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [projectedTotal, projectedDiff]),('Win%', [awayteampercent, hometeampercent])])
    else:
        j=pd.DataFrame.from_records([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [0, 0]),('Win%', [0, 0])])
    return(j)


#    dfNow=df[10:]
def getTeamHistoryForSpreadHistory(thisTeam,thisGameDate):
    dfNow=GetThisTeamInfoFromCsv(thisTeam,WhichFile) 
   
    OpponentIs=dfNow.loc[dfNow['Date'] == thisGameDate,"Opponent"].values[0]
    
    ATSVegasis=dfNow.loc[dfNow['Date'] == thisGameDate,"ATSVegas"].values[0]
    ActualFinalSpread=dfNow.loc[dfNow['Date'] == thisGameDate,"ATSVegas"].values[0]-dfNow.loc[dfNow['Date'] == thisGameDate,"ATS"].values[0]
    TRankATS=dfNow.loc[dfNow['Date'] == thisGameDate,"TCurSpread"].values[0]
    TG3ATSis=dfNow.loc[dfNow['Date'] == thisGameDate,"T3GSpread"].values[0]
    MC5ATSis=dfNow.loc[dfNow['Date'] == thisGameDate,"MC5Spread"].values[0]
    MC10ATSis=dfNow.loc[dfNow['Date'] == thisGameDate,"MC10Spread"].values[0]
    
    gg1=[TRankATS,TG3ATSis,MC5ATSis,MC10ATSis]
    thenewDF=pd.DataFrame({"Spreads":gg1})

    thenewDF["Teams"]=OpponentIs
    thenewDF["Vegas"]=ATSVegasis
    thenewDF["Actual"]=ActualFinalSpread
    return(thenewDF)
    
    #HomeTeamTrending=df.loc[df['Date'] == newDate,"DifCumSumFastEMA"].values[0]-df.loc[df['Date'] == newDate,"DifCumSumEMA"].values[0]
    #AwayTeamTrending=dfOther.loc[dfOther['Date'] == newDate,"DifCumSumFastEMA"].values[0]-dfOther.loc[dfOther['Date'] == newDate,"DifCumSumEMA"].values[0]

def DropUnrankedTeams(CheckingTeam):
    for i in range(len(CheckingTeam.index)):
        if CheckingTeam.loc[i,'Op Rank'].split(' ')[0] == 0:
            CheckingTeam=CheckingTeam.drop(CheckingTeam[CheckingTeam.loc[i,"ADJ O"] ==0].index, inplace=False).reset_index()
    return(CheckingTeam) 

def GetTeamDataRevisedNov18Again(NameofTeam):
    # get correctyear
    NameofTeamUniversity=TeamDatabase.loc[NameofTeam,"University"]
    FullHomeTeamName=TeamDatabase.loc[NameofTeam,"Full Team Name"]
    FullHomeTeamName = FullHomeTeamName.replace(" ","-")
    HomeTeamLookup="http://barttorvik.com/results.php?team="+NameofTeamUniversity+"&year=2019"
    res = requests.get(HomeTeamLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    dfHomeResults = pd.read_html(str(table))[0]
    newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","Pace","jj","jj2","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","y"]
    dfHomeResults.columns=newCName
    dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","Pace","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()

    dfHomeResults['ADJ O'] = dfHomeResults['ADJ O'].astype(str)
    if any(dfHomeResults['ADJ O'] == "-"):
        dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] =="-"].index, inplace=False).reset_index()
        
        
    HomeTeamATSLookup="https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/ats-results"
    res = requests.get(HomeTeamATSLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeATS = pd.read_html(str(table))[0]
    dfHomeATS['Opp Rank'] = dfHomeATS['Opp Rank'].astype(str)
    if any(dfHomeATS['Opp Rank'] == "--"):
        dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    if dfHomeATS.columns.get_values()[0] == 'index':
        dfHomeATS=dfHomeATS[dfHomeATS.columns[1:]]   

    HomeTeamOverUnderLookup = "https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/over-under-results"
    res = requests.get(HomeTeamOverUnderLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeOver = pd.read_html(str(table))[0]
    dfHomeOver['Opp Rank'] = dfHomeOver['Opp Rank'].astype(str)
    if any(dfHomeOver['Opp Rank'] == "--"):
        dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    if dfHomeOver.columns.get_values()[0] == 'index':
        dfHomeOver=dfHomeOver[dfHomeOver.columns[1:]]   
 

    
    #newCName=["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]
    #newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","WinP","jj","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]

    #dfHomeResults.columns=newCName
    #dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()
    dfHomeResults=dfHomeResults.reset_index()

    #print(dfHomeResults.dtypes)
    dfHomeResults=DropUnrankedTeams(dfHomeResults)
    #print(dfHomeResults)
    #dfHomeResults=dfHomeResults.reset_index()
    #dfHomeResults["ADJ O"]=convert_column_remove_period(dfHomeResults["ADJ O"])
    #int(DftoChange.loc[0,"Op Rank"].split(' ')[0])ATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O1"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ADJ O"]=dfHomeResults["ADJ O"].astype(np.float64)
    dfHomeResults["ADJ D"]=dfHomeResults["ADJ D"].astype(np.float64)
    #dfHomeResults["ADJ O"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ D"]=pd.to.numeric(dfHomeResults["ADJ D"],errors='coerce').fillna(0).astype(np.float64)
    g=dfHomeResults["ADJ O"].values-dfHomeResults["ADJ D"].values
    dfHomeResults["EMRating"]=g
    dfHomeResults["EMRating3GameEMA"]=dfHomeResults["EMRating"].rolling(3).mean()
    dfHomeResults["EMRating3GameExpMA"]=dfHomeResults["EMRating"].ewm(span=3,adjust=False).mean()

    #dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.float64)


    dfHomeResults['AdjO']=dfHomeResults["ADJ O"].values
    dfHomeResults["AdjO3GameEMA"]=dfHomeResults["AdjO"].rolling(3).mean()
    
    dfHomeResults['AdjD']=dfHomeResults["ADJ D"].values
    dfHomeResults["AdjD3GameEMA"]=dfHomeResults["AdjD"].rolling(3).mean()
    
    dfHomeResults['PPP']=dfHomeResults["PPP"].values
    dfHomeResults['EFG%']=dfHomeResults["EFG%"].values
    dfHomeResults['TO%']=dfHomeResults["TO%"].values

    dfHomeResults['PPP1']=dfHomeResults["PPP1"].values
    dfHomeResults['EFG%1']=dfHomeResults["EFG%1"].values
    dfHomeResults['TO%1']=dfHomeResults["TO%1"].values
    HomeTeamData=dfHomeResults[["ATS","EMRating","AdjO","AdjO3GameEMA","AdjD","AdjD3GameEMA","Date","HomeAway","Op Rank","Opponent","Result","Pace","PPP","EFG%","TO%","PPP1","EFG%1","TO%1","OverUnder","ATSVegas","OverUnderVegas"]].copy()

    HomeTeamData["EMRating"]=pd.to_numeric(HomeTeamData["EMRating"],errors='coerce').fillna(0).astype(np.int64)
    return HomeTeamData

def getBartDataTest():
    # need to check year
    res = requests.get("http://www.barttorvik.com/trank.php?year=2019&sort=&conlimit=")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table') [0]
    df = pd.read_html(str(table))[0]
    newCName=["Rank","Team","Conference","Games","Record","AdjOE","AdjDE","OERank","DERank","BARTH","PRecord","CRecord","EFG%","EGO Rank","EFG%D","EGD Rank","TO%","TO Rank","TO%D","TOD Rank","ADJT","OR Rank","OR%D%","ORD Rank","FT%","FT Rank","FT%D","FTD Rank","FTR%1","2P1","3P1","WAB","ADJ","WAB@"]

    df.columns=newCName
    BartRankings=df[["Rank","Team","Record","AdjOE","AdjDE","OERank","DERank","BARTH","EFG%","EGO Rank","EFG%D","EGD Rank","TO%","TO Rank","TO%D","TOD Rank","ADJ","OR Rank","OR%D%","ORD Rank","FT%","FT Rank","FT%D","ADJT"]].copy()

    #BartRankings.set_index("Team", inplace=True)

    BartRankings['AdjOE'] = BartRankings.AdjOE.apply(calculate_to_numeric)
    BartRankings['OERank'] = BartRankings.OERank.apply(calculate_to_numeric)
    BartRankings['AdjDE'] = BartRankings.AdjDE.apply(calculate_to_numeric)
    BartRankings['DERank'] = BartRankings.DERank.apply(calculate_to_numeric)
    BartRankings['ADJT'] = BartRankings.ADJT.apply(calculate_to_numeric)
    BartRankings['BARTH'] = BartRankings.BARTH.apply(calculate_to_numeric)

    return BartRankings


def GetTeamDataRevisedDec24Again(NameofTeam):
    # get correctyear
    NameofTeamUniversity=TeamDatabase.loc[NameofTeam,"University"]
    FullHomeTeamName=TeamDatabase.loc[NameofTeam,"Full Team Name"]
    FullHomeTeamName = FullHomeTeamName.replace(" ","-")
    HomeTeamLookup="http://barttorvik.com/results.php?team="+NameofTeamUniversity+"&year=2019"
    res = requests.get(HomeTeamLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    dfHomeResults = pd.read_html(str(table))[0]
    newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","Pace","jj","jj2","jj3","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","y"]
    dfHomeResults.columns=newCName
    dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","Pace","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()

    dfHomeResults['ADJ O'] = dfHomeResults['ADJ O'].astype(str)
    if any(dfHomeResults['ADJ O'] == "-"):
        dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] =="-"].index, inplace=False).reset_index()
        
        
    HomeTeamATSLookup="https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/ats-results"
    res = requests.get(HomeTeamATSLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeATS = pd.read_html(str(table))[0]
    dfHomeATS['Opp Rank'] = dfHomeATS['Opp Rank'].astype(str)
    if any(dfHomeATS['Opp Rank'] == "--"):
        dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    if dfHomeATS.columns.get_values()[0] == 'index':
        dfHomeATS=dfHomeATS[dfHomeATS.columns[1:]]   

    HomeTeamOverUnderLookup = "https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/over-under-results"
    res = requests.get(HomeTeamOverUnderLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeOver = pd.read_html(str(table))[0]
    dfHomeOver['Opp Rank'] = dfHomeOver['Opp Rank'].astype(str)
    if any(dfHomeOver['Opp Rank'] == "--"):
        dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    if dfHomeOver.columns.get_values()[0] == 'index':
        dfHomeOver=dfHomeOver[dfHomeOver.columns[1:]]   
 

    
    #newCName=["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]
    #newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","WinP","jj","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]

    #dfHomeResults.columns=newCName
    #dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()
    dfHomeResults=dfHomeResults.reset_index()

    #print(dfHomeResults.dtypes)
    dfHomeResults=DropUnrankedTeams(dfHomeResults)
    #print(dfHomeResults)
    #dfHomeResults=dfHomeResults.reset_index()
    #dfHomeResults["ADJ O"]=convert_column_remove_period(dfHomeResults["ADJ O"])
    #int(DftoChange.loc[0,"Op Rank"].split(' ')[0])ATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O1"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ADJ O"]=dfHomeResults["ADJ O"].astype(np.float64)
    dfHomeResults["ADJ D"]=dfHomeResults["ADJ D"].astype(np.float64)
    #dfHomeResults["ADJ O"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ D"]=pd.to.numeric(dfHomeResults["ADJ D"],errors='coerce').fillna(0).astype(np.float64)
    g=dfHomeResults["ADJ O"].values-dfHomeResults["ADJ D"].values
    dfHomeResults["EMRating"]=g
    dfHomeResults["EMRating3GameEMA"]=dfHomeResults["EMRating"].rolling(3).mean()
    dfHomeResults["EMRating3GameExpMA"]=dfHomeResults["EMRating"].ewm(span=3,adjust=False).mean()

    #dfHomeResults["EMRating5GameEMA"]=dfHomeResults["EMRating"].rolling(5).mean()
    dfHomeResults["EMRating5GameExpMA"]=dfHomeResults["EMRating"].ewm(span=5,adjust=False).mean()
    dfHomeResults["EMRating5GameExpMA"].fillna(0, inplace=True)
    
    #dfHomeResults["EMRating10GameEMA"]=dfHomeResults["EMRating"].rolling(10).mean()
    dfHomeResults["EMRating10GameExpMA"]=dfHomeResults["EMRating"].ewm(span=10,adjust=False).mean()
    dfHomeResults["EMRating10GameExpMA"].fillna(0, inplace=True)

    
    #dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #esults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.float64)


    dfHomeResults['AdjO']=dfHomeResults["ADJ O"].values
    dfHomeResults["AdjO3GameEMA"]=dfHomeResults["AdjO"].rolling(3).mean()
    dfHomeResults["AdjO3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['AdjD']=dfHomeResults["ADJ D"].values
    dfHomeResults["AdjD3GameEMA"]=dfHomeResults["AdjD"].rolling(3).mean()
    dfHomeResults["AdjD3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['PPP']=dfHomeResults["PPP"].values
    dfHomeResults['EFG%']=dfHomeResults["EFG%"].values
    dfHomeResults['TO%']=dfHomeResults["TO%"].values

    dfHomeResults['PPP1']=dfHomeResults["PPP1"].values
    dfHomeResults['EFG%1']=dfHomeResults["EFG%1"].values
    dfHomeResults['TO%1']=dfHomeResults["TO%1"].values
    HomeTeamData=dfHomeResults[["ATS","EMRating","EMRating3GameEMA","EMRating5GameExpMA","EMRating10GameExpMA","AdjO","AdjO3GameEMA","AdjD","AdjD3GameEMA","Date","HomeAway","Op Rank","Opponent","Result","Pace","PPP","EFG%","TO%","PPP1","EFG%1","TO%1","OverUnder","ATSVegas","OverUnderVegas"]].copy()

    HomeTeamData["EMRating"]=pd.to_numeric(HomeTeamData["EMRating"],errors='coerce').fillna(0).astype(np.int64)
    
    return HomeTeamData

def GetTeamDataRevisedDec24Again2020(NameofTeam):
    # get correctyear
    NameofTeamUniversity=TeamDatabase.loc[NameofTeam,"University"]
    FullHomeTeamName=TeamDatabase.loc[NameofTeam,"Full Team Name"]
    FullHomeTeamName = FullHomeTeamName.replace(" ","-")
    HomeTeamLookup="http://barttorvik.com/results.php?team="+NameofTeamUniversity+"&year=2020"
    res = requests.get(HomeTeamLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    dfHomeResults = pd.read_html(str(table))[0]

    newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Rseult2","Result","Pace","jj","jj2","jj3","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","y"]
    dfHomeResults=dfHomeResults.iloc[:, : 30]
    dfHomeResults.columns=newCName
    
    dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","Pace","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()

    dfHomeResults['ADJ O'] = dfHomeResults['ADJ O'].astype(str)
    if any(dfHomeResults['ADJ O'] == "-"):
        dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] =="-"].index, inplace=False).reset_index()

    #if any(dfHomeResults['ADJ O'] == ''):
    #    dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] ==''].index, inplace=False).reset_index()
    continents=['H','A','N']
    dfHomeResults=dfHomeResults[dfHomeResults.HomeAway.isin(continents)]
        
    HomeTeamATSLookup="https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/ats-results"
    res = requests.get(HomeTeamATSLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeATS = pd.read_html(str(table))[0]
    dfHomeATS['Opp Rank'] = dfHomeATS['Opp Rank'].astype(str)
    if any(dfHomeATS['Opp Rank'] == "--"):
        dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #if dfHomeATS.columns.get_values()[0] == 'index':
     #   dfHomeATS=dfHomeATS[dfHomeATS.columns[1:]]   

    HomeTeamOverUnderLookup = "https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/over-under-results"
    res = requests.get(HomeTeamOverUnderLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeOver = pd.read_html(str(table))[0]
    dfHomeOver['Opp Rank'] = dfHomeOver['Opp Rank'].astype(str)
    if any(dfHomeOver['Opp Rank'] == "--" ):
        dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #if dfHomeOver.columns.get_values()[0] == 'index':
     #   dfHomeOver=dfHomeOver[dfHomeOver.columns[1:]]   
 

    
    #newCName=["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]
    #newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","WinP","jj","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]

    #dfHomeResults.columns=newCName
    #dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()
    dfHomeResults=dfHomeResults.reset_index()

    #print(dfHomeResults.dtypes)
    dfHomeResults=DropUnrankedTeams(dfHomeResults)
    #print(dfHomeResults)
    #dfHomeResults=dfHomeResults.reset_index()
    #dfHomeResults["ADJ O"]=convert_column_remove_period(dfHomeResults["ADJ O"])
    #int(DftoChange.loc[0,"Op Rank"].split(' ')[0])ATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O1"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O"]=dfHomeResults["ADJ O"].astype(np.float64)
    #dfHomeResults["ADJ D"]=dfHomeResults["ADJ D"].astype(np.float64)
    dfHomeResults["ADJ O"]=pd.to_numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ADJ D"]=pd.to_numeric(dfHomeResults["ADJ D"],errors='coerce').fillna(0).astype(np.float64)
    g=dfHomeResults["ADJ O"].values-dfHomeResults["ADJ D"].values
    dfHomeResults["EMRating"]=g
    dfHomeResults["EMRating3GameEMA"]=dfHomeResults["EMRating"].rolling(3).mean()
    dfHomeResults["EMRating3GameExpMA"]=dfHomeResults["EMRating"].ewm(span=3,adjust=False).mean()

    #dfHomeResults["EMRating5GameEMA"]=dfHomeResults["EMRating"].rolling(5).mean()
    dfHomeResults["EMRating5GameExpMA"]=dfHomeResults["EMRating"].ewm(span=5,adjust=False).mean()
    dfHomeResults["EMRating5GameExpMA"].fillna(0, inplace=True)
    
    #dfHomeResults["EMRating10GameEMA"]=dfHomeResults["EMRating"].rolling(10).mean()
    dfHomeResults["EMRating10GameExpMA"]=dfHomeResults["EMRating"].ewm(span=10,adjust=False).mean()
    dfHomeResults["EMRating10GameExpMA"].fillna(0, inplace=True)

    
    #dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #esults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[5]],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[5]],errors='coerce').fillna(0).astype(np.float64)


    dfHomeResults['AdjO']=dfHomeResults["ADJ O"].values
    dfHomeResults["AdjO3GameEMA"]=dfHomeResults["AdjO"].rolling(3).mean()
    dfHomeResults["AdjO3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['AdjD']=dfHomeResults["ADJ D"].values
    dfHomeResults["AdjD3GameEMA"]=dfHomeResults["AdjD"].rolling(3).mean()
    dfHomeResults["AdjD3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['PPP']=dfHomeResults["PPP"].values
    dfHomeResults['EFG%']=dfHomeResults["EFG%"].values
    dfHomeResults['TO%']=dfHomeResults["TO%"].values

    dfHomeResults['PPP1']=dfHomeResults["PPP1"].values
    dfHomeResults['EFG%1']=dfHomeResults["EFG%1"].values
    dfHomeResults['TO%1']=dfHomeResults["TO%1"].values
    HomeTeamData=dfHomeResults[["ATS","EMRating","EMRating3GameEMA","EMRating5GameExpMA","EMRating10GameExpMA","AdjO","AdjO3GameEMA","AdjD","AdjD3GameEMA","Date","HomeAway","Op Rank","Opponent","Result","Pace","PPP","EFG%","TO%","PPP1","EFG%1","TO%1","OverUnder","ATSVegas","OverUnderVegas"]].copy()

    HomeTeamData["EMRating"]=pd.to_numeric(HomeTeamData["EMRating"],errors='coerce').fillna(0).astype(np.int64)
    
    return HomeTeamData

def GetTeamDataRevisedDec24Again2021(NameofTeam):
    TeamDatabase=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
    TeamDatabase.set_index("OldTRankName", inplace=True)
    # get correctyear
    NameofTeamUniversity=TeamDatabase.loc[NameofTeam,"University"]
    FullHomeTeamName=TeamDatabase.loc[NameofTeam,"Full Team Name"]
    FullHomeTeamName = FullHomeTeamName.replace(" ","-")
    HomeTeamLookup="http://barttorvik.com/results.php?team="+NameofTeamUniversity+"&year=2021"
    res = requests.get(HomeTeamLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    dfHomeResults = pd.read_html(str(table))[0]

    newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Rseult2","Result","Pace","jj","jj2","jj3","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","y"]
    dfHomeResults=dfHomeResults.iloc[:, : 30]
    dfHomeResults.columns=newCName
    
    dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","Pace","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()

    dfHomeResults['ADJ O'] = dfHomeResults['ADJ O'].astype(str)
    if any(dfHomeResults['ADJ O'] == "-"):
        dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] =="-"].index, inplace=False).reset_index()

    #if any(dfHomeResults['ADJ O'] == ''):
    #    dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] ==''].index, inplace=False).reset_index()
    continents=['H','A','N']
    dfHomeResults=dfHomeResults[dfHomeResults.HomeAway.isin(continents)]
        
    HomeTeamATSLookup="https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/ats-results"
    res = requests.get(HomeTeamATSLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeATS = pd.read_html(str(table))[0]
    dfHomeATS['Opp Rank'] = dfHomeATS['Opp Rank'].astype(str)
    if any(dfHomeATS['Opp Rank'] == "--"):
        dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #if dfHomeATS.columns.get_values()[0] == 'index':
     #   dfHomeATS=dfHomeATS[dfHomeATS.columns[1:]]   

    HomeTeamOverUnderLookup = "https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/over-under-results"
    res = requests.get(HomeTeamOverUnderLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeOver = pd.read_html(str(table))[0]
    dfHomeOver['Opp Rank'] = dfHomeOver['Opp Rank'].astype(str)
    if any(dfHomeOver['Opp Rank'] == "--" ):
        dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #if dfHomeOver.columns.get_values()[0] == 'index':
     #   dfHomeOver=dfHomeOver[dfHomeOver.columns[1:]]   
 

    
    #newCName=["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]
    #newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","WinP","jj","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]

    #dfHomeResults.columns=newCName
    #dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()
    dfHomeResults=dfHomeResults.reset_index()

    #print(dfHomeResults.dtypes)
    dfHomeResults=DropUnrankedTeams(dfHomeResults)
    #print(dfHomeResults)
    #dfHomeResults=dfHomeResults.reset_index()
    #dfHomeResults["ADJ O"]=convert_column_remove_period(dfHomeResults["ADJ O"])
    #int(DftoChange.loc[0,"Op Rank"].split(' ')[0])ATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O1"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O"]=dfHomeResults["ADJ O"].astype(np.float64)
    #dfHomeResults["ADJ D"]=dfHomeResults["ADJ D"].astype(np.float64)
    dfHomeResults["ADJ O"]=pd.to_numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ADJ D"]=pd.to_numeric(dfHomeResults["ADJ D"],errors='coerce').fillna(0).astype(np.float64)
    g=dfHomeResults["ADJ O"].values-dfHomeResults["ADJ D"].values
    dfHomeResults["EMRating"]=g
    dfHomeResults["EMRating3GameEMA"]=dfHomeResults["EMRating"].rolling(3).mean()
    dfHomeResults["EMRating3GameExpMA"]=dfHomeResults["EMRating"].ewm(span=3,adjust=False).mean()

    #dfHomeResults["EMRating5GameEMA"]=dfHomeResults["EMRating"].rolling(5).mean()
    dfHomeResults["EMRating5GameExpMA"]=dfHomeResults["EMRating"].ewm(span=5,adjust=False).mean()
    dfHomeResults["EMRating5GameExpMA"].fillna(0, inplace=True)
    
    #dfHomeResults["EMRating10GameEMA"]=dfHomeResults["EMRating"].rolling(10).mean()
    dfHomeResults["EMRating10GameExpMA"]=dfHomeResults["EMRating"].ewm(span=10,adjust=False).mean()
    dfHomeResults["EMRating10GameExpMA"].fillna(0, inplace=True)

    
    #dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #esults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #goodColumnATS=len(dfHomeATS.columns)-2
    newNum=len(dfHomeATS.columns)-3
    newNumOver=len(dfHomeOver.columns)-4

    dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[newNum]],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[newNumOver]],errors='coerce').fillna(0).astype(np.float64)

#"OR%","FTR%"
    dfHomeResults['AdjO']=dfHomeResults["ADJ O"].values
    dfHomeResults["AdjO3GameEMA"]=dfHomeResults["AdjO"].rolling(3).mean()
    dfHomeResults["AdjO3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['AdjD']=dfHomeResults["ADJ D"].values
    dfHomeResults["AdjD3GameEMA"]=dfHomeResults["AdjD"].rolling(3).mean()
    dfHomeResults["AdjD3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['PPP']=dfHomeResults["PPP"].values
    dfHomeResults['EFG%']=dfHomeResults["EFG%"].values
    dfHomeResults['TO%']=dfHomeResults["TO%"].values
    dfHomeResults['OR%']=dfHomeResults["OR%"].values
    dfHomeResults['FTR%']=dfHomeResults["FTR%"].values
    dfHomeResults['PPP1']=dfHomeResults["PPP1"].values
    dfHomeResults['EFG%1']=dfHomeResults["EFG%1"].values
    dfHomeResults['TO%1']=dfHomeResults["TO%1"].values
    dfHomeResults['OR%1']=dfHomeResults["OR%1"].values
    dfHomeResults['FTR%1']=dfHomeResults["FTR%1"].values
    dfHomeResults["AdjO3ExpMA"]=dfHomeResults["ADJ O"].ewm(span=3,adjust=False).mean()
    dfHomeResults["AdjO3ExpMAS"]=dfHomeResults["AdjO3ExpMA"]
    dfHomeResults["AdjO3ExpMAS"]=dfHomeResults["AdjO3ExpMAS"].shift(1).fillna(0)
    dfHomeResults["AdjO3ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjO5ExpMA"]=dfHomeResults["ADJ O"].ewm(span=5,adjust=False).mean()
    dfHomeResults["AdjO5ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjO10ExpMA"]=dfHomeResults["ADJ O"].ewm(span=10,adjust=False).mean()
    dfHomeResults["AdjO10ExpMA"].fillna(0, inplace=True)
    
    
    dfHomeResults["AdjD3ExpMA"]=dfHomeResults["ADJ D"].ewm(span=3,adjust=False).mean()
    dfHomeResults["AdjD3ExpMAS"]=dfHomeResults["AdjD3ExpMA"]
    dfHomeResults["AdjD3ExpMAS"]=dfHomeResults["AdjD3ExpMAS"].shift(1).fillna(0)
    dfHomeResults["AdjD3ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjD5ExpMA"]=dfHomeResults["ADJ D"].ewm(span=5,adjust=False).mean()
    dfHomeResults["AdjD5ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjD10ExpMA"]=dfHomeResults["ADJ D"].ewm(span=10,adjust=False).mean()
    dfHomeResults["AdjD10ExpMA"].fillna(0, inplace=True)
    
    
    HomeTeamData=dfHomeResults[["ATS","EMRating","EMRating3GameEMA","EMRating5GameExpMA","EMRating10GameExpMA","AdjO","AdjO3GameEMA","AdjD","AdjD3GameEMA","Date","HomeAway","Op Rank","Opponent","Result","Pace","PPP","EFG%","TO%","OR%","FTR%","PPP1","EFG%1","TO%1","OR%1","FTR%1","OverUnder","ATSVegas","OverUnderVegas","AdjO3ExpMA","AdjO3ExpMAS","AdjO5ExpMA","AdjO10ExpMA","AdjD3ExpMA","AdjD3ExpMAS","AdjD5ExpMA","AdjD10ExpMA"]].copy()

    HomeTeamData["EMRating"]=pd.to_numeric(HomeTeamData["EMRating"],errors='coerce').fillna(0).astype(np.int64)
    
    return HomeTeamData

def GetTeamDataRevisedDec24Again2022(NameofTeam):
    TeamDatabase=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
    TeamDatabase.set_index("OldTRankName", inplace=True)
    # get correctyear
    NameofTeamUniversity=TeamDatabase.loc[NameofTeam,"University"]
    FullHomeTeamName=TeamDatabase.loc[NameofTeam,"Full Team Name"]
    FullHomeTeamName = FullHomeTeamName.replace(" ","-")
    HomeTeamLookup="http://barttorvik.com/results.php?team="+NameofTeamUniversity+"&year=2022"
    res = requests.get(HomeTeamLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    dfHomeResults = pd.read_html(str(table))[0]

    newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Rseult2","Result","Pace","jj","jj2","jj3","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","y"]
    dfHomeResults=dfHomeResults.iloc[:, : 30]
    dfHomeResults.columns=newCName
    
    dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","Pace","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()

    dfHomeResults['ADJ O'] = dfHomeResults['ADJ O'].astype(str)
    if any(dfHomeResults['ADJ O'] == "-"):
        dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] =="-"].index, inplace=False).reset_index()

    #if any(dfHomeResults['ADJ O'] == ''):
    #    dfHomeResults=dfHomeResults.drop(dfHomeResults[dfHomeResults["ADJ O"] ==''].index, inplace=False).reset_index()
    continents=['H','A','N']
    dfHomeResults=dfHomeResults[dfHomeResults.HomeAway.isin(continents)]
        
    HomeTeamATSLookup="https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/ats-results"
    res = requests.get(HomeTeamATSLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeATS = pd.read_html(str(table))[0]
    dfHomeATS['Opp Rank'] = dfHomeATS['Opp Rank'].astype(str)
    if any(dfHomeATS['Opp Rank'] == "--"):
        dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeATS=dfHomeATS.drop(dfHomeATS[dfHomeATS["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #if dfHomeATS.columns.get_values()[0] == 'index':
     #   dfHomeATS=dfHomeATS[dfHomeATS.columns[1:]]   

    HomeTeamOverUnderLookup = "https://www.teamrankings.com/ncaa-basketball/team/"+FullHomeTeamName+"/over-under-results"
    res = requests.get(HomeTeamOverUnderLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    dfHomeOver = pd.read_html(str(table))[0]
    dfHomeOver['Opp Rank'] = dfHomeOver['Opp Rank'].astype(str)
    if any(dfHomeOver['Opp Rank'] == "--" ):
        dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #dfHomeOver=dfHomeOver.drop(dfHomeOver[dfHomeOver["Opp Rank"] =="--"].index, inplace=False).reset_index()
    #if dfHomeOver.columns.get_values()[0] == 'index':
     #   dfHomeOver=dfHomeOver[dfHomeOver.columns[1:]]   
 

    
    #newCName=["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]
    #newCName=["Date","HomeAway","BS","Op Rank","MoreBS","Opponent","Result","WinP","jj","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore","r","y"]

    #dfHomeResults.columns=newCName
    #dfHomeResults=dfHomeResults[["Date","HomeAway","Op Rank","Opponent","Result","ADJ O","ADJ D","PPP","EFG%","TO%","OR%","FTR%","2P","3P","PPP1","EFG%1","TO%1","OR%1","FTR%1","2P1","3P1","GScore"]].dropna()
    dfHomeResults=dfHomeResults.reset_index()

    #print(dfHomeResults.dtypes)
    dfHomeResults=DropUnrankedTeams(dfHomeResults)
    #print(dfHomeResults)
    #dfHomeResults=dfHomeResults.reset_index()
    #dfHomeResults["ADJ O"]=convert_column_remove_period(dfHomeResults["ADJ O"])
    #int(DftoChange.loc[0,"Op Rank"].split(' ')[0])ATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O1"]=pd.to.numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    #dfHomeResults["ADJ O"]=dfHomeResults["ADJ O"].astype(np.float64)
    #dfHomeResults["ADJ D"]=dfHomeResults["ADJ D"].astype(np.float64)
    dfHomeResults["ADJ O"]=pd.to_numeric(dfHomeResults["ADJ O"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["ADJ D"]=pd.to_numeric(dfHomeResults["ADJ D"],errors='coerce').fillna(0).astype(np.float64)
    g=dfHomeResults["ADJ O"].values-dfHomeResults["ADJ D"].values
    dfHomeResults["EMRating"]=g
    dfHomeResults["EMRating3GameEMA"]=dfHomeResults["EMRating"].rolling(3).mean()
    dfHomeResults["EMRating3GameExpMA"]=dfHomeResults["EMRating"].ewm(span=3,adjust=False).mean()

    #dfHomeResults["EMRating5GameEMA"]=dfHomeResults["EMRating"].rolling(5).mean()
    dfHomeResults["EMRating5GameExpMA"]=dfHomeResults["EMRating"].ewm(span=5,adjust=False).mean()
    dfHomeResults["EMRating5GameExpMA"].fillna(0, inplace=True)
    
    #dfHomeResults["EMRating10GameEMA"]=dfHomeResults["EMRating"].rolling(10).mean()
    dfHomeResults["EMRating10GameExpMA"]=dfHomeResults["EMRating"].ewm(span=10,adjust=False).mean()
    dfHomeResults["EMRating10GameExpMA"].fillna(0, inplace=True)

    
    #dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #esults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.int64)
    #dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[4]],errors='coerce').fillna(0).astype(np.int64)
    dfHomeResults["ATS"]=pd.to_numeric(dfHomeATS["Diff"],errors='coerce').fillna(0).astype(np.float64)
    #goodColumnATS=len(dfHomeATS.columns)-2
    newNum=len(dfHomeATS.columns)-3
    newNumOver=len(dfHomeOver.columns)-4

    dfHomeResults["ATSVegas"]=pd.to_numeric(dfHomeATS[dfHomeATS.columns[newNum]],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnder"]=pd.to_numeric(dfHomeOver["Diff"],errors='coerce').fillna(0).astype(np.float64)
    dfHomeResults["OverUnderVegas"]=pd.to_numeric(dfHomeOver[dfHomeOver.columns[newNumOver]],errors='coerce').fillna(0).astype(np.float64)

#"OR%","FTR%"
    dfHomeResults['AdjO']=dfHomeResults["ADJ O"].values
    dfHomeResults["AdjO3GameEMA"]=dfHomeResults["AdjO"].rolling(3).mean()
    dfHomeResults["AdjO3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['AdjD']=dfHomeResults["ADJ D"].values
    dfHomeResults["AdjD3GameEMA"]=dfHomeResults["AdjD"].rolling(3).mean()
    dfHomeResults["AdjD3GameEMA"].fillna(0, inplace=True)
    dfHomeResults['PPP']=dfHomeResults["PPP"].values
    dfHomeResults['EFG%']=dfHomeResults["EFG%"].values
    dfHomeResults['TO%']=dfHomeResults["TO%"].values
    dfHomeResults['OR%']=dfHomeResults["OR%"].values
    dfHomeResults['FTR%']=dfHomeResults["FTR%"].values
    dfHomeResults['PPP1']=dfHomeResults["PPP1"].values
    dfHomeResults['EFG%1']=dfHomeResults["EFG%1"].values
    dfHomeResults['TO%1']=dfHomeResults["TO%1"].values
    dfHomeResults['OR%1']=dfHomeResults["OR%1"].values
    dfHomeResults['FTR%1']=dfHomeResults["FTR%1"].values
    dfHomeResults["AdjO3ExpMA"]=dfHomeResults["ADJ O"].ewm(span=3,adjust=False).mean()
    dfHomeResults["AdjO3ExpMAS"]=dfHomeResults["AdjO3ExpMA"]
    dfHomeResults["AdjO3ExpMAS"]=dfHomeResults["AdjO3ExpMAS"].shift(1).fillna(0)
    dfHomeResults["AdjO3ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjO5ExpMA"]=dfHomeResults["ADJ O"].ewm(span=5,adjust=False).mean()
    dfHomeResults["AdjO5ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjO10ExpMA"]=dfHomeResults["ADJ O"].ewm(span=10,adjust=False).mean()
    dfHomeResults["AdjO10ExpMA"].fillna(0, inplace=True)
    
    
    dfHomeResults["AdjD3ExpMA"]=dfHomeResults["ADJ D"].ewm(span=3,adjust=False).mean()
    dfHomeResults["AdjD3ExpMAS"]=dfHomeResults["AdjD3ExpMA"]
    dfHomeResults["AdjD3ExpMAS"]=dfHomeResults["AdjD3ExpMAS"].shift(1).fillna(0)
    dfHomeResults["AdjD3ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjD5ExpMA"]=dfHomeResults["ADJ D"].ewm(span=5,adjust=False).mean()
    dfHomeResults["AdjD5ExpMA"].fillna(0, inplace=True)
    dfHomeResults["AdjD10ExpMA"]=dfHomeResults["ADJ D"].ewm(span=10,adjust=False).mean()
    dfHomeResults["AdjD10ExpMA"].fillna(0, inplace=True)
    
    
    HomeTeamData=dfHomeResults[["ATS","EMRating","EMRating3GameEMA","EMRating5GameExpMA","EMRating10GameExpMA","AdjO","AdjO3GameEMA","AdjD","AdjD3GameEMA","Date","HomeAway","Op Rank","Opponent","Result","Pace","PPP","EFG%","TO%","OR%","FTR%","PPP1","EFG%1","TO%1","OR%1","FTR%1","OverUnder","ATSVegas","OverUnderVegas","AdjO3ExpMA","AdjO3ExpMAS","AdjO5ExpMA","AdjO10ExpMA","AdjD3ExpMA","AdjD3ExpMAS","AdjD5ExpMA","AdjD10ExpMA"]].copy()

    HomeTeamData["EMRating"]=pd.to_numeric(HomeTeamData["EMRating"],errors='coerce').fillna(0).astype(np.int64)
    
    return HomeTeamData
def GetWinLossPercentGameWinneragainstVegasDec19(whichList,ColumntoTest,ColumntoTestAgainst):
    
    #SpreadNameWin=ColumntoTest+"ATSWin"
    #SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    winLossList=[]
    PercentList=[]
    #hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(whichList)):
        PercentList.append(kenpom_prob(abs(whichList[i][ColumntoTest][1]), std=10.5))
        ATSCover=whichList[i]["Vegas"][1]-whichList[i]["Actual"][1]
        
        
        if whichList[i][ColumntoTest][1] > whichList[i][ColumntoTestAgainst][1]:
            if ATSCover>=0:
                winLossList.append(0)
            else:
                winLossList.append(1)
        else: 
            if ATSCover>0:
                winLossList.append(1)
            else:
                winLossList.append(0)
        

                
    #print(winLossList)
    #print(PercentList)
    BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,BrierScoreToday,winLossList)


def getGameDayBettingInfo(DfInfo):
    thisList=[DfInfo["Teams"][1],DfInfo["Vegas"][1],DfInfo["TRank"][1],DfInfo["TG3"][1],DfInfo["MC5"][1],DfInfo["MC10"][1],DfInfo["Pom"][1],DfInfo["Actual"][1]]
    f=pd.DataFrame(thisList)
    return(f,thisList)
#Teams,Vegas,TRank,TG3,MC5,MC10,Pom,Actual


def getGameInfoFromBartDailyScheduleDec7(df1,thisDate,gameCounter):
#df[0].iloc[0][0]   
#u'Nov 22 12:30 PM CT'
    #theDateStringFromBart=df1[0].iloc[0][0]
    #theDayNumber=getTheDayoftheMonth(theDateStringFromBart.split()[1])
    #thisDate=getTheCorrectDateFromStringBartSchedule(theDateStringFromBart.split()[0],theDayNumber)
#df[0].iloc[0][1] 
#'38 Oklahoma vs 17 Wisconsin'
    theTeamStringFromBart=df1[0].iloc[0][gameCounter]
    print(theTeamStringFromBart)
    newhometeam,newawayteam,whereisGame=findHomeTeamandSpreadfromBartSchedule(theTeamStringFromBart)
#df[0].iloc[0][2] 
#'Wisconsin -3.4, 74-70 (63%)'
    print(newhometeam,newawayteam,whereisGame)
    thePointSpreadStringFromBart=df1[0].iloc[0][gameCounter+1]
    print(thePointSpreadStringFromBart)
    print(len(thePointSpreadStringFromBart.split()))
    
    
    #favorite=df1[0].iloc[0][2].split(', ')[0].replace(projectedspread,"")[:-1]
   
    actualwinner=df1[0].iloc[0][gameCounter+3].split(',')[0]
    actualscore=df1[0].iloc[0][gameCounter+3].split(', ')[1]
    projectedspread=df[0].iloc[0][gameCounter+1].split(', ')[0].split()[(len(df[0].iloc[0][gameCounter+1].split(', ')[0].split())-1)]
    projectedscore=df[0].iloc[0][gameCounter+1].split(', ')[1].split()[0]
    projectedwinpercent=df[0].iloc[0][gameCounter+1].split(', ')[1].split()[1][1:-1]
    projectedlosspercent=str(100-int(df[0].iloc[0][gameCounter+1].split(', ')[1].split()[1][1:-2]))+"%"
    
    favorite=df[0].iloc[0][gameCounter+1].split(', ')[0].replace(projectedspread,"")[:-1]
    theVegasSpreadData,theVegasOverData,theTG3Spread,theTG3Over,theMC5Spread,theMC5Over,theMC10Spread,theMC10Over,HomeTeamHot,AwayTeamHot,actualPace,estimatedPace,thePomSpread5,thePomTotal5,HomePlayO,AwayPlayO=getDataFromCsvforGamedayDec7(newhometeam,newawayteam,thisDate)
    
    if actualwinner==newhometeam:
        hometeamscore,awayteamscore,actualTotal,actualDiff=getScorefromString(actualscore)
        actualDiff=actualDiff*-1
    else:
        awayteamscore,hometeamscore,actualTotal,actualDiff=getScorefromString(actualscore)
    
            
    if favorite==newhometeam:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff=getScorefromString(projectedscore)
        projectedDiff=projectedDiff*-1
        hometeampercent=projectedwinpercent
        awayteampercent=projectedlosspercent
    else:
        projectedawayteamscore,projectedhometeamscore,projectedTotal,projectedDiff=getScorefromString(projectedscore)
        hometeampercent=projectedlosspercent
        awayteampercent=projectedwinpercent
    
    
    
    if len(thePointSpreadStringFromBart.split()[len(thePointSpreadStringFromBart.split())-1])<6:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff,hometeampercent,awayteampercent=getProjectedSpreadandTotalFromBartSchedule(thePointSpreadStringFromBart,newhometeam)
    
        j=pd.DataFrame.from_dict([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [projectedTotal, projectedDiff]),('Win%', [awayteampercent, hometeampercent])])
    else:
        j=pd.DataFrame.from_dict([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [0, 0]),('Win%', [0, 0])])
    j2=pd.DataFrame.from_dict([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('Trend', [AwayTeamHot,HomeTeamHot]),('Vegas', [theVegasOverData,theVegasSpreadData]),('Actual', [actualTotal, actualDiff]),('TRank', [projectedTotal, projectedDiff]),('TG3', [theTG3Over, theTG3Spread]),('MC5', [theMC5Over, theMC5Spread]),('MC10', [theMC10Over, theMC10Spread]),('Pom', [thePomTotal5, thePomSpread5]),('Final', [awayteamscore,hometeamscore]),('Win%', [awayteampercent, hometeampercent]),('Pace',[actualPace,estimatedPace]),('PlayingO',[AwayPlayO,HomePlayO])])
   
    return(j,j2)


def getDataFromCsvforGamedayDec7(WhichTeam,OtherTeam,WhichDate):
    df=GetThisTeamInfoFromCsv(WhichTeam,WhichFile)
    dfOther=GetThisTeamInfoFromCsv(OtherTeam,WhichFile)
    #newDate=convertFullDatetoShortDate(WhichDate)
    newDate=int(WhichDate)
    
    ATSVegasis=df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
    OUVegasis=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]
    TG3ATSis=df.loc[df['DateNew'] == newDate,"B3GSpread"].values[0]
    TG3OUis=df.loc[df['DateNew'] == newDate,"B3GTotal"].values[0]
    MC5ATSis=df.loc[df['DateNew'] == newDate,"MC5Spread"].values[0]
    MC5OUis=df.loc[df['DateNew'] == newDate,"MC5Total"].values[0]
    MC10ATSis=df.loc[df['DateNew'] == newDate,"MC10Spread"].values[0]
    MC10OUis=df.loc[df['DateNew'] == newDate,"MC10Total"].values[0]
    HomeTeamTrending=df.loc[df['DateNew'] == newDate,"SignalSumShift"].values[0]
    AwayTeamTrending=dfOther.loc[dfOther['DateNew'] == newDate,"SignalSumShift"].values[0]
    thePomeroyEstimatedPace=df.loc[df['DateNew'] == newDate,"PomTempo"].values[0]
    theActualTempo=df.loc[df['DateNew'] == newDate,"Pace"].values[0]
    PomATSis=df.loc[df['DateNew'] == newDate,"PomSpread"].values[0]
    PomOUis=df.loc[df['DateNew'] == newDate,"PomTotal"].values[0]
    
    HomeTeamPlayingO=df.loc[df['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
    AwayTeamPlayingO=dfOther.loc[dfOther['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
    
    #if HomeTeamTrending > 0:
    #    HomeTeamisHot=True
    #else:
    #    HomeTeamisHot=False
        
    return(ATSVegasis,OUVegasis,TG3ATSis,TG3OUis,MC5ATSis,MC5OUis,MC10ATSis,MC10OUis,HomeTeamTrending,AwayTeamTrending,thePomeroyEstimatedPace,theActualTempo,PomATSis,PomOUis,HomeTeamPlayingO,AwayTeamPlayingO)

def getOverplayingChart(TeamData,ThisTeam):
    
    TeamData["GameDifRating"]=TeamData["EMRating"]-TeamData["AdjEMCurrent"]
    TeamData["DifCumSum"]=TeamData["GameDifRating"].cumsum()
    TeamData["DifCumSumEMA"]=TeamData["DifCumSum"].ewm(span=5,adjust=False).mean()
    TeamData["DifCumSum"]=TeamData["DifCumSum"].shift(1).fillna(0)
    TeamData["DifCumSumEMA"]=TeamData["DifCumSumEMA"].shift(1).fillna(0)

    ChartTitleName=ThisTeam+" Overplaying and ATS"
    plt.title(ChartTitleName)
    TeamData["DifCumSum"].plot()
    TeamData["DifCumSumEMA"].plot()
    TeamData.index,q["ATS"].plot(kind='bar')
    plt.show()
    

def getOverplayingChartBothTeams(HomeTeamData,AwayTeamData,HomeTeam,AwayTeam):
    
    
    
    HomeTeamData["GameDifRating"]=HomeTeamData["EMRating"]-HomeTeamData["AdjEMCurrent"]
    HomeTeamData["DifCumSum"]=HomeTeamData["GameDifRating"].cumsum()
    HomeTeamData["DifCumSumEMA"]=HomeTeamData["DifCumSum"].ewm(span=5,adjust=False).mean()
    HomeTeamData["DifCumSum"]=HomeTeamData["DifCumSum"].shift(1).fillna(0)
    HomeTeamData["DifCumSumEMA"]=HomeTeamData["DifCumSumEMA"].shift(1).fillna(0)

    AwayTeamData["GameDifRating"]=AwayTeamData["EMRating"]-AwayTeamData["AdjEMCurrent"]
    AwayTeamData["DifCumSum"]=AwayTeamData["GameDifRating"].cumsum()
    AwayTeamData["DifCumSumEMA"]=AwayTeamData["DifCumSum"].ewm(span=5,adjust=False).mean()
    AwayTeamData["DifCumSum"]=AwayTeamData["DifCumSum"].shift(1).fillna(0)
    AwayTeamData["DifCumSumEMA"]=AwayTeamData["DifCumSumEMA"].shift(1).fillna(0)

   
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
    
    plt.show()
    
    

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
    plt.show()
    f.savefig(pp,format='pdf')
    
    
def GetDailyPomeroyDataCSV(WhichDate):
    
    getThisCsv="C:/Users/mpgen/PomeroyDailyRankings2019/PomeroyRankings"+WhichDate+ ".csv"
    TodaysPomeroy=pd.read_csv(getThisCsv)
    TodaysPomeroy.set_index("Team", inplace=True)
    TodaysPomeroy['AdjO']=pd.to_numeric(TodaysPomeroy['AdjO'], errors='coerce')
    TodaysPomeroy['AdjD']=pd.to_numeric(TodaysPomeroy['AdjD'], errors='coerce')
    TodaysPomeroy['AdjT']=pd.to_numeric(TodaysPomeroy['AdjT'], errors='coerce')
    TodaysPomeroy['AdjEM']=pd.to_numeric(TodaysPomeroy['AdjEM'], errors='coerce')

    return TodaysPomeroy

def GetDailyTRankDataCSV(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyRankings2019/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank

def GetDailyLast10TRankDataCSV(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyLast10Rankings2019/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
    
    
def GetDailyPomeroyDataCSV2020(WhichDate):
    
    getThisCsv="C:/Users/mpgen/PomeroyDailyRankings2020/PomeroyRankings"+WhichDate+ ".csv"
    TodaysPomeroy=pd.read_csv(getThisCsv)
    TodaysPomeroy.set_index("Team", inplace=True)
    TodaysPomeroy['AdjO']=pd.to_numeric(TodaysPomeroy['AdjO'], errors='coerce')
    TodaysPomeroy['AdjD']=pd.to_numeric(TodaysPomeroy['AdjD'], errors='coerce')
    TodaysPomeroy['AdjT']=pd.to_numeric(TodaysPomeroy['AdjT'], errors='coerce')
    TodaysPomeroy['AdjEM']=pd.to_numeric(TodaysPomeroy['AdjEM'], errors='coerce')

    return TodaysPomeroy
def GetDailyPomeroyDataCSV2021(WhichDate):
    
    getThisCsv="C:/Users/mpgen/PomeroyDailyRankings2021/PomeroyRankings"+WhichDate+ ".csv"
    TodaysPomeroy=pd.read_csv(getThisCsv)
    TodaysPomeroy.set_index("Team", inplace=True)
    TodaysPomeroy['AdjO']=pd.to_numeric(TodaysPomeroy['AdjO'], errors='coerce')
    TodaysPomeroy['AdjD']=pd.to_numeric(TodaysPomeroy['AdjD'], errors='coerce')
    TodaysPomeroy['AdjT']=pd.to_numeric(TodaysPomeroy['AdjT'], errors='coerce')
    TodaysPomeroy['AdjEM']=pd.to_numeric(TodaysPomeroy['AdjEM'], errors='coerce')

    return TodaysPomeroy

def GetDailyPomeroyDataCSV2022(WhichDate):
    
    getThisCsv="C:/Users/mpgen/PomeroyDailyRankings2022/PomeroyRankings"+WhichDate+ ".csv"
    TodaysPomeroy=pd.read_csv(getThisCsv)
    TodaysPomeroy.set_index("Team", inplace=True)
    TodaysPomeroy['AdjO']=pd.to_numeric(TodaysPomeroy['AdjO'], errors='coerce')
    TodaysPomeroy['AdjD']=pd.to_numeric(TodaysPomeroy['AdjD'], errors='coerce')
    TodaysPomeroy['AdjT']=pd.to_numeric(TodaysPomeroy['AdjT'], errors='coerce')
    TodaysPomeroy['AdjEM']=pd.to_numeric(TodaysPomeroy['AdjEM'], errors='coerce')

    return TodaysPomeroy
def GetDailyTRankDataCSV2020(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyRankings2020/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
def GetDailyTRankDataCSV2021(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyRankings2021/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
def GetDailyTRankDataCSV2022(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyRankings2022/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
def GetDailyLast10TRankDataCSV2020(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyLast10Rankings2020/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
def GetDailyLast10TRankDataCSV2021(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyLast10Rankings2021/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
def GetDailyLast10TRankDataCSV2022(WhichDate):
    
    getThisCsv="C:/Users/mpgen/TRankDailyLast10Rankings2022/"+WhichDate+ ".csv"
    TodaysTRank=pd.read_csv(getThisCsv)
    TodaysTRank.set_index("Team", inplace=True)

    return TodaysTRank
def CreateDailyTRankRankings(dateToGet):


    theNewColumnsList=["AdjEM","AdjOE","AdjDE"]
    d = []

    for i in range(len(dateToGet)):
        todaysRequest="http://barttorvik.com/trankslice.php?year=2019&sort=&conlimit=&begin=20181106&end="+dateToGet[i]+"&top=&quad=4&mingames=0&venue=All&type=All"
        res = requests.get(todaysRequest)
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0] 
        df = pd.read_html(str(table))[0]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2","j3","j4","j5","j6","j7","j8","j9","j10","J11","jjj","j12","j13"]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2"]

        df.columns=NewCNametoday
        df=df[["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB"]]

    
        df=df.drop('Rec', 1)
        df=df[df.ADJOE.str.contains("ADJOE") == False]
        df=df.reset_index()
        df=sanitizeEntireColumn(df,"Team")
        df.set_index("Team", inplace=True)
        df["AdjOE"]=pd.to_numeric(df["ADJOE"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjDE"]=pd.to_numeric(df["ADJDE"],errors='coerce').fillna(0).astype(np.float64)
        df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
        df=df[["Rk","AdjEM","AdjOE","AdjDE"]]
   
        fileNameToPrint="C:/Users/michael/TRankDailyRankings2019/"+dateToGet[i]+".csv"

        df.to_csv(fileNameToPrint)
        d.append(df)


def CreateDailyLast10TRankRankings(dateToGet):


    theNewColumnsList=["AdjEM","AdjOE","AdjDE"]
    d = []

    for i in range(len(dateToGet)):
        todaysRequest="http://barttorvik.com/trank.php?year=2019&sort=&lastx=5&hteam=&conlimit=All&state=All&begin=20181101&end="+dateToGet[i]+"&top=0&quad=4&venue=All&type=All&mingames=0#"
        res = requests.get(todaysRequest)
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0] 
        df = pd.read_html(str(table))[0]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2","j3","j4","j5","j6","j7","j8","j9","j10","J11","jjj","j12","j13"]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2"]

        df.columns=NewCNametoday
        df=df[["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB"]]

    
        df=df.drop('Rec', 1)
        df=df[df.ADJOE.str.contains("ADJOE") == False]
        df=df.reset_index()
        df=sanitizeEntireColumn(df,"Team")
        df.set_index("Team", inplace=True)
        df["AdjOE"]=pd.to_numeric(df["ADJOE"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjDE"]=pd.to_numeric(df["ADJDE"],errors='coerce').fillna(0).astype(np.float64)
        df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
        df=df[["Rk","AdjEM","AdjOE","AdjDE"]]
   
        fileNameToPrint="C:/Users/michael/TRankDailyLast10Rankings2019/"+dateToGet[i]+".csv"

        df.to_csv(fileNameToPrint)
        d.append(df)
        
def CreateDailyTRankRankings2020(dateToGet):


    theNewColumnsList=["AdjEM","AdjOE","AdjDE"]
    d = []

    for i in range(len(dateToGet)):
        todaysRequest="http://barttorvik.com/trankslice.php?year=2019&sort=&conlimit=&begin=20191106&end="+dateToGet+"&top=&quad=4&mingames=0&venue=All&type=All"
        res = requests.get(todaysRequest)
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0] 
        df = pd.read_html(str(table))[0]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2","j3","j4","j5","j6","j7","j8","j9","j10","J11","jjj","j12","j13"]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2"]

        df.columns=NewCNametoday
        df=df[["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB"]]

    
        df=df.drop('Rec', 1)
        df=df[df.ADJOE.str.contains("ADJOE") == False]
        df=df.reset_index()
        df=sanitizeEntireColumn(df,"Team")
        df.set_index("Team", inplace=True)
        df["AdjOE"]=pd.to_numeric(df["ADJOE"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjDE"]=pd.to_numeric(df["ADJDE"],errors='coerce').fillna(0).astype(np.float64)
        df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
        df=df[["Rk","AdjEM","AdjOE","AdjDE"]]
   
        fileNameToPrint="C:/Users/mpgen/TRankDailyRankings2020/"+dateToGet+".csv"

        df.to_csv(fileNameToPrint)
        d.append(df)
 
    
    
def CreateDailyTRankRankings2021(dateToGet):


    theNewColumnsList=["AdjEM","AdjOE","AdjDE"]
    d = []

    for i in range(len(dateToGet)):
        todaysRequest="http://barttorvik.com/trankslice.php?year=2019&sort=&conlimit=&begin=20201124&end="+dateToGet+"&top=&quad=4&mingames=0&venue=All&type=All"
        res = requests.get(todaysRequest)
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0] 
        df = pd.read_html(str(table))[0]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2","j3","j4","j5","j6","j7","j8","j9","j10","J11","jjj","j12","j13"]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2"]

        df.columns=NewCNametoday
        df=df[["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB"]]

    
        df=df.drop('Rec', 1)
        df=df[df.ADJOE.str.contains("ADJOE") == False]
        df=df.reset_index()
        df=sanitizeEntireColumn(df,"Team")
        df.set_index("Team", inplace=True)
        df["AdjOE"]=pd.to_numeric(df["ADJOE"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjDE"]=pd.to_numeric(df["ADJDE"],errors='coerce').fillna(0).astype(np.float64)
        df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
        df=df[["Rk","AdjEM","AdjOE","AdjDE"]]
   
        fileNameToPrint="C:/Users/mpgen/TRankDailyRankings2021/"+dateToGet+".csv"

        df.to_csv(fileNameToPrint)
        d.append(df)
    



def CreateDailyLast10TRankRankings2020(dateToGet):


    theNewColumnsList=["AdjEM","AdjOE","AdjDE"]
    d = []

    for i in range(len(dateToGet)):
        todaysRequest="http://barttorvik.com/trank.php?year=2019&sort=&lastx=10&hteam=&conlimit=All&state=All&begin=20191101&end="+dateToGet+"&top=0&quad=4&venue=All&type=All&mingames=0#"
        res = requests.get(todaysRequest)
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0] 
        df = pd.read_html(str(table))[0]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2","j3","j4","j5","j6","j7","j8","j9","j10","J11","jjj","j12","j13"]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2"]

        df.columns=NewCNametoday
        df=df[["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB"]]

    
        df=df.drop('Rec', 1)
        df=df[df.ADJOE.str.contains("ADJOE") == False]
        df=df.reset_index()
        df=sanitizeEntireColumn(df,"Team")
        df.set_index("Team", inplace=True)
        df["AdjOE"]=pd.to_numeric(df["ADJOE"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjDE"]=pd.to_numeric(df["ADJDE"],errors='coerce').fillna(0).astype(np.float64)
        df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
        df=df[["Rk","AdjEM","AdjOE","AdjDE"]]
   
        fileNameToPrint="C:/Users/mpgen/TRankDailyLast10Rankings2020/"+dateToGet+".csv"

        df.to_csv(fileNameToPrint)
        d.append(df)
def CreateDailyLast10TRankRankings2021(dateToGet):


    theNewColumnsList=["AdjEM","AdjOE","AdjDE"]
    d = []

    for i in range(len(dateToGet)):
        todaysRequest="http://barttorvik.com/trank.php?year=2019&sort=&lastx=10&hteam=&conlimit=All&state=All&begin=20201124&end="+dateToGet+"&top=0&quad=4&venue=All&type=All&mingames=0#"
        res = requests.get(todaysRequest)
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0] 
        df = pd.read_html(str(table))[0]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2","j3","j4","j5","j6","j7","j8","j9","j10","J11","jjj","j12","j13"]
        NewCNametoday=["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB","j","j2"]

        df.columns=NewCNametoday
        df=df[["Rk","Team","Conference","Games","Rec","ADJOE","ADJDE","BART","efg%","efgd%","TOOff","TODef","FT","FTD","2pt%","2ptd%","3pt%","3PD%","ADJT","WAB"]]

    
        df=df.drop('Rec', 1)
        df=df[df.ADJOE.str.contains("ADJOE") == False]
        df=df.reset_index()
        df=sanitizeEntireColumn(df,"Team")
        df.set_index("Team", inplace=True)
        df["AdjOE"]=pd.to_numeric(df["ADJOE"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjDE"]=pd.to_numeric(df["ADJDE"],errors='coerce').fillna(0).astype(np.float64)
        df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
        df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
        df=df[["Rk","AdjEM","AdjOE","AdjDE"]]
   
        fileNameToPrint="C:/Users/mpgen/TRankDailyLast10Rankings2021/"+dateToGet+".csv"

        df.to_csv(fileNameToPrint)
        d.append(df)
def AddHistoricalRankingsDec16(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFiles2019'
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
        
        w=GetDailyTRankDataCSV(DftoChange.loc[i,"DateNew"])
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #test1=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])

        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreLagDec16(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreLagDec16(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreLagDec16(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreLagDec16(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtLagDec16(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtLagDec16(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            theEstimatedSpread=theEstimatedSpread*-1
            theEstimatedSpread10G=theEstimatedSpread10G*-1

          
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSum"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return DftoChange



def AddHistoricalRankingsNov30(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFiles2019'
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
        
        w=GetDailyTRankDataCSV(DftoChange.loc[i,"DateNew"])
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #test1=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])

        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-1]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-1]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-1]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-1]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScore(OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScore(OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-1]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-1]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-1]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-1]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScore(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScore(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-1]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-1]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-1]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-1]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            theEstimatedSpread=theEstimatedSpread*-1
            theEstimatedSpread10G=theEstimatedSpread10G*-1

          
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSum"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
    
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return DftoChange



def AddHistoricalRankingsDec23(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFiles2019'
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
        
        w=GetDailyTRankDataCSV(DftoChange.loc[i,"DateNew"])
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])

        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            theEstimatedSpread=theEstimatedSpread*-1
            theEstimatedSpread10G=theEstimatedSpread10G*-1

          
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSum"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return DftoChange



def AddHistoricalRankingsDec24(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFiles2019'
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
        
        w=GetDailyTRankDataCSV(DftoChange.loc[i,"DateNew"])
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])

        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],float(DftoChange.loc[i,"PomAdjTCurrentOpp"]),DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],float(DftoChange.loc[i,"PomAdjTCurrent"]),LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            #theEstimatedSpread=theEstimatedSpread*-1
            #theEstimatedSpread10G=theEstimatedSpread10G*-1

          
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        #DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        #DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        #DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        #DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        #DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        #DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        #edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        #DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSum"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return DftoChange




def AddHistoricalRankingsDec242020(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFilesStarter2021'
    
    # change folder
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
# change for 2021        
        w=GetDailyTRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(DftoChange.loc[i,"DateNew"])
        #print(w)
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV2021(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])

        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
 
        
        
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],float(DftoChange.loc[i,"PomAdjTCurrentOpp"]),DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],float(DftoChange.loc[i,"PomAdjTCurrent"]),LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            #theEstimatedSpread=theEstimatedSpread*-1
            #theEstimatedSpread10G=theEstimatedSpread10G*-1

        B3GHomeTeamSpread=B3GHomeTeamSpread*-1
        theEstimatedSpread=theEstimatedSpread*-1
        theEstimatedSpread10G=theEstimatedSpread10G*-1 
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        
        #DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        #edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        #DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSumShifted"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return (DftoChange)


def AddHistoricalRankingsJan52021(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFilesStarter2021'
    
    # change folder
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
# change for 2021        
        w=GetDailyTRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(DftoChange.loc[i,"DateNew"])
        #print(w)
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV2021(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])
        
        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
 

        TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)



        #MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings/tm_seasons_stats_ranks"+dateforRankings5+".csv")
        MG_Rank=Get_MG_Rankings_CSV2021(DftoChange.loc[i,"DateNew"])
        MG_Rank["updated"]=update_type(MG_Rank.tm,TeamDatabase2.UpdatedTRankName)
        MG_Rank.set_index("updated", inplace=True)
 
        MG_DF1_ADJ_Off=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjO']
        MG_DF1_ADJ_Def=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjD']
        MG_DF1_ADJ_Off['adj_stat']=MG_DF1_ADJ_Off['adj_stat'].fillna(0)
        MG_DF1_ADJ_Def['adj_stat']=MG_DF1_ADJ_Def['adj_stat'].fillna(0)
        AwayTeamM=TeamDatabase2.loc[TheOpponent,"SportsReference"]
        HomeTeamM=TeamDatabase2.loc[NameofThisTeam5,"SportsReference"]
        print(NameofThisTeam5)
        print(TheOpponent)
        print(HomeTeamM)
        print(AwayTeamM)
        DftoChange.loc[i,"AdjOE_MG"]=MG_DF1_ADJ_Off.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG"]=MG_DF1_ADJ_Def.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG"]=DftoChange.loc[i,"AdjOE_MG"]-DftoChange.loc[i,"AdjDE_MG"]
        
        DftoChange.loc[i,"AdjOE_MG_Opp"]=MG_DF1_ADJ_Off.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG_Opp"]=MG_DF1_ADJ_Def.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG_Opp"]=DftoChange.loc[i,"AdjOE_MG_Opp"]-DftoChange.loc[i,"AdjDE_MG_Opp"]
        
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],float(DftoChange.loc[i,"PomAdjTCurrentOpp"]),DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],float(DftoChange.loc[i,"PomAdjTCurrent"]),LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],thePGameTempo,LeagueOE)
            #Set_MG_Data(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                #Set_MG_Data_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            MHomeTeamSpread=MHomeTeamSpread*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            #theEstimatedSpread=theEstimatedSpread*-1
            #theEstimatedSpread10G=theEstimatedSpread10G*-1

        B3GHomeTeamSpread=B3GHomeTeamSpread*-1
        theEstimatedSpread=theEstimatedSpread*-1
        theEstimatedSpread10G=theEstimatedSpread10G*-1 
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange.loc[i,"MG_Spread"]=MHomeTeamSpread
        DftoChange.loc[i,"MG_Total"]=MTotalPoints
        #DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        #edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        #DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSumShifted"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MG_Spread","MG_Total")
    
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return (DftoChange)

def AddHistoricalRankingsJan52022(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFilesStarter2022'
    
    # change folder
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
# change for 2021        
        w=GetDailyTRankDataCSV2022(DftoChange.loc[i,"DateNew"])
        #print(DftoChange.loc[i,"DateNew"])
        #print(w)
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV2021(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])
        
        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
 

        TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)



        #MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings/tm_seasons_stats_ranks"+dateforRankings5+".csv")
        MG_Rank=Get_MG_Rankings_CSV2022(DftoChange.loc[i,"DateNew"])
        MG_Rank["updated"]=update_type(MG_Rank.tm,TeamDatabase2.UpdatedTRankName)
        MG_Rank.set_index("updated", inplace=True)
 
        MG_DF1_ADJ_Off=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjO']
        MG_DF1_ADJ_Def=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjD']
        MG_DF1_ADJ_Off['adj_stat']=MG_DF1_ADJ_Off['adj_stat'].fillna(0)
        MG_DF1_ADJ_Def['adj_stat']=MG_DF1_ADJ_Def['adj_stat'].fillna(0)
        AwayTeamM=TeamDatabase2.loc[TheOpponent,"SportsReference"]
        HomeTeamM=TeamDatabase2.loc[NameofThisTeam5,"SportsReference"]
        print(NameofThisTeam5)
        print(TheOpponent)
        print(HomeTeamM)
        print(AwayTeamM)
        DftoChange.loc[i,"AdjOE_MG"]=MG_DF1_ADJ_Off.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG"]=MG_DF1_ADJ_Def.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG"]=DftoChange.loc[i,"AdjOE_MG"]-DftoChange.loc[i,"AdjDE_MG"]
        
        DftoChange.loc[i,"AdjOE_MG_Opp"]=MG_DF1_ADJ_Off.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG_Opp"]=MG_DF1_ADJ_Def.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG_Opp"]=DftoChange.loc[i,"AdjOE_MG_Opp"]-DftoChange.loc[i,"AdjDE_MG_Opp"]
        
        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],float(DftoChange.loc[i,"PomAdjTCurrentOpp"]),DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],float(DftoChange.loc[i,"PomAdjTCurrent"]),LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],thePGameTempo,LeagueOE)
            #Set_MG_Data(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                #Set_MG_Data_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjO3GameEMA"],OppData.loc[len(OppData.index)-1,"AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            MHomeTeamSpread=MHomeTeamSpread*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            #theEstimatedSpread=theEstimatedSpread*-1
            #theEstimatedSpread10G=theEstimatedSpread10G*-1

        B3GHomeTeamSpread=B3GHomeTeamSpread*-1
        theEstimatedSpread=theEstimatedSpread*-1
        theEstimatedSpread10G=theEstimatedSpread10G*-1 
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange.loc[i,"MG_Spread"]=MHomeTeamSpread
        DftoChange.loc[i,"MG_Total"]=MTotalPoints
        #DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        #edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        #DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSumShifted"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MG_Spread","MG_Total")
    
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return (DftoChange)

def AddHistoricalRankingsMonteDec24(DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFiles2019Starter'
   
    DftoChange["Test"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
        
       
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
       
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])

        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
  
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],LeagueTempo,LeagueOE)
            #BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            #CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            theAwayO,theAwayD,theHomeO,theHomeD=getBG3Numbers(dateofThisGame,OppData,DftoChange)
        
            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(theAwayO,theAwayD,theHomeO,theHomeD,thePGameTempo,LeagueOE)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                #BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                #CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                theAwayO,theAwayD,theHomeO,theHomeD=getBG3Numbers(dateofThisGame,DftoChange,OppData)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(theAwayO,theAwayD,theHomeO,theHomeD,thePGameTempo,LeagueOE)
           
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:  
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                #BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                #CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                theAwayO,theAwayD,theHomeO,theHomeD=getBG3Numbers(dateofThisGame,DftoChange,OppData)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(theAwayO,theAwayD,theHomeO,theHomeD,thePGameTempo,LeagueOE)
           
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            #CurHomeTeamSpread=CurHomeTeamSpread*-1
            #PHomeTeamSpread=PHomeTeamSpread*-1
            #BHomeTeamSpread=BHomeTeamSpread*-1
            B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            theEstimatedSpread=theEstimatedSpread*-1
            theEstimatedSpread10G=theEstimatedSpread10G*-1

          
      
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
       
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  

    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
   
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return DftoChange


def getOverplayingChartDec4(TeamData,ThisTeam):
    
    #TeamData["GameDifRating"]=TeamData["EMRating"]-TeamData["AdjEMCurrent"]
    #TeamData["DifCumSum"]=TeamData["GameDifRating"].cumsum()
    #TeamData["DifCumSumEMA"]=TeamData["DifCumSum"].ewm(span=5,adjust=False).mean()
    #TeamData["DifCumSum"]=TeamData["DifCumSum"].shift(1).fillna(0)
    #TeamData["DifCumSumEMA"]=TeamData["DifCumSumEMA"].shift(1).fillna(0)

    ChartTitleName=ThisTeam+" Overplaying and ATS"
    plt.title(ChartTitleName)
    TeamData["DifCumSum"].plot()
    TeamData["DifCumSumEMA"].plot()
    TeamData.index,TeamData["ATS"].plot(kind='bar')
    plt.show()
    
#getOverplayingChart(q,"Furman")


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
    
    plt.show()
    f.savefig(pp,format='pdf')
#getOverplayingChartBothTeams(test1,test2,AwayTeam,HomeTeam)


def getBG3Numbers(dateofThisGameToday,AwayTeamData,HomeTeamData):
    
    f=AwayTeamData[AwayTeamData['DateNew'] == dateofThisGameToday].index
    #print(f) 
    #print(dateofThisGameToday)
    if f[0]==0:
        AwayO=0
        AwayD=0
        
    else:
        
        AwayO=AwayTeamData.iloc[f[0]-1]["AdjO3GameEMA"]
        AwayD=AwayTeamData.iloc[f[0]-1]["AdjD3GameEMA"]
        
        
    f1=HomeTeamData[HomeTeamData['DateNew'] == dateofThisGameToday].index
    #print(f) 
    #print(dateofThisGameToday)
    if f1[0]==0:
        HomeO=0
        HomeD=0
        
    else:
        
        HomeO=HomeTeamData.iloc[f1[0]-1]["AdjO3GameEMA"]
        HomeD=HomeTeamData.iloc[f1[0]-1]["AdjD3GameEMA"]  
        
    return(AwayO,AwayD,HomeO,HomeD)




def getMonteCarloGameScoreLagDec16(dateofThisGameToday,AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    w=AwayTeamData
    
    if w["DateNew"].iloc[-1] == dateofThisGameToday:
        print("Skipped Game")
        TexasO=list(w["AdjO"][:-1])
        TexasD=list(w["AdjD"][:-1])
        print(TexasO)
    else:
        TexasO=list(w["AdjO"])
        TexasD=list(w["AdjD"])
    w1=HomeTeamData
    
    if w1["DateNew"].iloc[-1] == dateofThisGameToday:
        print("Skipped Game")
        StO=list(w1["AdjO"][:-1])
        StD=list(w1["AdjD"][:-1])
        print(StO)
    else:
        StO=list(w1["AdjO"])
        StD=list(w1["AdjD"])   
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNew(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    #print(EstimatedSpread)
    return q,q1,EstimatedTotal,EstimatedSpread


def getMonteCarloGameScoreNeutralCourtLagDec16(dateofThisGameToday,AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    w=AwayTeamData
    if w["DateNew"].iloc[-1] == dateofThisGameToday:
        print("skipped game")
        TexasO=list(w["AdjO"][:-1])
        TexasD=list(w["AdjD"][:-1])
        print(TexasO)
    else:
        TexasO=list(w["AdjO"])
        TexasD=list(w["AdjD"])
    w1=HomeTeamData
    
    if w1["DateNew"].iloc[-1] == dateofThisGameToday:
        print("skipped game")
        StO=list(w1["AdjO"][:-1])
        StD=list(w1["AdjD"][:-1])
        print(StO)
    else:
        StO=list(w1["AdjO"])
        StD=list(w1["AdjD"])  
        

    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNewNeutral(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread


def getMonteCarloGameScoreDec23(dateofThisGameToday,AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    
    f=AwayTeamData[AwayTeamData['DateNew'] == dateofThisGameToday].index
    #print(f) 
    #print(dateofThisGameToday)
    #if f[0]==0:
        #w=AwayTeamData
    if len(AwayTeamData[AwayTeamData['DateNew'] == dateofThisGameToday]) == 1:
        w=AwayTeamData
    else:
        w=AwayTeamData[0:f[0]]
   
    
    #f=AwayTeamData[AwayTeamData['DateNew'] == dateofThisGameToday].index
    #w=AwayTeamData[0:f[0]]

    
    #w=AwayTeamData
    TexasO=list(w["AdjO"])
    TexasD=list(w["AdjD"])
    #print(TexasO)
    #print(AwayTeamData)
    #print(HomeTeamData)
    
    f1=HomeTeamData[HomeTeamData['DateNew'] == dateofThisGameToday].index
    #print(f1)
    #if f1[0]==0:
    #    w1=HomeTeamData
    if len(HomeTeamData[HomeTeamData['DateNew'] == dateofThisGameToday]) == 1:
        w1=HomeTeamData    
        
    else:
        w1=HomeTeamData[0:f1[0]]
  
    #w1=HomeTeamData[0:f[0]]
    #w1=HomeTeamData
    StO=list(w1["AdjO"])
    StD=list(w1["AdjD"])
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNew(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread

def getMonteCarloGameScoreJan31(dateofThisGameToday,AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    
    
    #DftoChange['Dates'] = pd.to_datetime(DftoChange['DateNew'], format='%Y%m%d') 
    AwayTeamData['Dates'] = pd.to_datetime(AwayTeamData['DateNew'], format='%Y%m%d') 
    AwayTeamData = AwayTeamData.set_index('Dates')
    dateStr=str(dateofThisGameToday)[:-4]+'-'+str(dateofThisGameToday)[4:6]+'-'+str(dateofThisGameToday)[6:8]
    if len(AwayTeamData[:dateStr][:-1])>0:
        
        TexasO=list(AwayTeamData[:dateStr][:-1]['AdjO'].tail(HowManyGames))
        TexasD=list(AwayTeamData[:dateStr][:-1]['AdjD'].tail(HowManyGames))
    else:
        TexasO=list(AwayTeamData[:dateStr]['AdjO'].tail(HowManyGames))
        TexasD=list(AwayTeamData[:dateStr]['AdjD'].tail(HowManyGames))
    #dateofThisGameToday=int(dateofThisGameToday)
    #wO=AwayTeamData[AwayTeamData['DateNew'] <= dateofThisGameToday]['AdjO'].tail(6).head(5)
    #wD=AwayTeamData[AwayTeamData['DateNew'] <= dateofThisGameToday]['AdjD'].tail(6).head(5)
    #w=AwayTeamData
    #TexasO=list(wO)
    #TexasD=list(wD)
    #print(TexasO)
    #print(AwayTeamData)
    #print(HomeTeamData)
    HomeTeamData['Dates'] = pd.to_datetime(HomeTeamData['DateNew'], format='%Y%m%d') 
    HomeTeamData = HomeTeamData.set_index('Dates')
    dateStr=str(dateofThisGameToday)[:-4]+'-'+str(dateofThisGameToday)[4:6]+'-'+str(dateofThisGameToday)[6:8]
    if len(HomeTeamData[:dateStr][:-1])>0:
        StO=list(HomeTeamData[:dateStr][:-1]['AdjO'].tail(HowManyGames))
        StD=list(HomeTeamData[:dateStr][:-1]['AdjD'].tail(HowManyGames))
    else:
        StO=list(HomeTeamData[:dateStr]['AdjO'].tail(HowManyGames))
        StD=list(HomeTeamData[:dateStr]['AdjD'].tail(HowManyGames))
    #w1O=HomeTeamData[HomeTeamData['DateNew'] <= dateofThisGameToday]['AdjO'].tail(6).head(5)
    #w1D=HomeTeamData[HomeTeamData['DateNew'] <= dateofThisGameToday]['AdjD'].tail(6).head(5)
    #w1=HomeTeamData[0:f[0]]
    #w1=HomeTeamData
    #StO=list(w1O)
    #StD=list(w1D)
    
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)

    
    q,q1=getGameDistributionNew(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread

def getMonteCarloGameScoreNeutralCourtJan31(dateofThisGameToday,AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    AwayTeamData['Dates'] = pd.to_datetime(AwayTeamData['DateNew'], format='%Y%m%d') 
    AwayTeamData = AwayTeamData.set_index('Dates')
    dateStr=str(dateofThisGameToday)[:-4]+'-'+str(dateofThisGameToday)[4:6]+'-'+str(dateofThisGameToday)[6:8]
    if len(AwayTeamData[:dateStr][:-1])>0:
        
        TexasO=list(AwayTeamData[:dateStr][:-1]['AdjO'].tail(HowManyGames))
        TexasD=list(AwayTeamData[:dateStr][:-1]['AdjD'].tail(HowManyGames))
    else:
        TexasO=list(AwayTeamData[:dateStr]['AdjO'].tail(HowManyGames))
        TexasD=list(AwayTeamData[:dateStr]['AdjD'].tail(HowManyGames))
    

    #dateofThisGameToday=int(dateofThisGameToday)
    #wO=AwayTeamData[AwayTeamData['DateNew'] <= dateofThisGameToday]['AdjO'].tail(6).head(5)
    #wD=AwayTeamData[AwayTeamData['DateNew'] <= dateofThisGameToday]['AdjD'].tail(6).head(5)
    #w=AwayTeamData
    #TexasO=list(wO)
    #TexasD=list(wD)
    #print(TexasO)
    #print(AwayTeamData)
    #print(HomeTeamData)
    HomeTeamData['Dates'] = pd.to_datetime(HomeTeamData['DateNew'], format='%Y%m%d') 
    HomeTeamData = HomeTeamData.set_index('Dates')
    dateStr=str(dateofThisGameToday)[:-4]+'-'+str(dateofThisGameToday)[4:6]+'-'+str(dateofThisGameToday)[6:8]
    if len(HomeTeamData[:dateStr][:-1])>0:
        StO=list(HomeTeamData[:dateStr][:-1]['AdjO'].tail(HowManyGames))
        StD=list(HomeTeamData[:dateStr][:-1]['AdjD'].tail(HowManyGames))
    else:
        StO=list(HomeTeamData[:dateStr]['AdjO'].tail(HowManyGames))
        StD=list(HomeTeamData[:dateStr]['AdjD'].tail(HowManyGames))

    
   
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNewNeutral(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread
def getMonteCarloGameScoreNeutralCourtDec23(dateofThisGameToday,AwayTeamData,HomeTeamData,HowManyGames,HowManySims,aGameTempo):
    numberofGames=-HowManyGames
    #print(dateofThisGameToday)
    f=AwayTeamData[AwayTeamData['DateNew'] == dateofThisGameToday].index
    #if f[0]==0:
    #    w=AwayTeamData
    if len(AwayTeamData[AwayTeamData['DateNew'] == dateofThisGameToday]) == 1:
        w=AwayTeamData
    else:
        w=AwayTeamData[0:f[0]]
    
    #w=AwayTeamData[0:f[0]]
    #w=AwayTeamData
    TexasO=list(w["AdjO"])
    TexasD=list(w["AdjD"])
    f1=HomeTeamData[HomeTeamData['DateNew'] == dateofThisGameToday].index
    #if f1[0]==0:
     #   w1=HomeTeamData
    if len(HomeTeamData[HomeTeamData['DateNew'] == dateofThisGameToday]) == 1:
        w1=HomeTeamData 
    else:
        w1=HomeTeamData[0:f1[0]]
    
    #w1=HomeTeamData[0:f[0]]
    #w1=HomeTeamData
    StO=list(w1["AdjO"])
    StD=list(w1["AdjD"])
    TexasOSample=sample_wr(TexasO[numberofGames:],10000)
    TexasDSample=sample_wr(TexasD[numberofGames:],10000)
    StOSample=sample_wr(StO[numberofGames:],10000)
    StDSample=sample_wr(StD[numberofGames:],10000)
    q,q1=getGameDistributionNewNeutral(TexasOSample,TexasDSample,StOSample,StDSample,aGameTempo,LeagueOE)
    EstimatedTotal=np.mean(q)
    EstimatedSpread=np.mean(q1)
    return q,q1,EstimatedTotal,EstimatedSpread


def getDailyPredictionTeamsAgainstSpreadDec18(ListofModels,VegasSpreadToTest,MCSpread,MCEdge,HomeTeam1,AwayTeam1,AwaySigScore,HomeSigScore,AwayOver,HomeOver):
    listofTeamsSelected=[]
    for q in range(len(ListofModels)):
    
        if VegasSpreadToTest<0:
            if VegasSpreadToTest<ListofModels[q]:
                teamSelection=AwayTeam1
            else:
                teamSelection=HomeTeam1
        else:
            if VegasSpreadToTest<ListofModels[q]:
                teamSelection=AwayTeam1
            else:
                teamSelection=HomeTeam1
        listofTeamsSelected.append(teamSelection)
    listofTeamsSelected.append(HomeTeam1)
    totalScore=HomeSigScore-AwaySigScore
    listofTeamsSelected.append(totalScore)
    totalOverplay=HomeOver-AwayOver
    listofTeamsSelected.append(totalOverplay)
    listofTeamsSelected.append(VegasSpreadToTest)
    listofTeamsSelected.append(MCSpread)
    listofTeamsSelected.append(MCEdge)

    return(listofTeamsSelected)

  
    

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
    plt.show()

    
    

def GetTwoChartsTogetherAnalysisDec31(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,SecondStat,PomStatHome,PomStatAway,VegasStat):
    HomeTeamInfo["First 3 Game"]=HomeTeamInfo[FirstStat].rolling(3).mean()
    HomeTeamInfo["First 5 Game"]=HomeTeamInfo[FirstStat].rolling(5).mean()
    HomeTeamInfo["First 10 Game"]=HomeTeamInfo[FirstStat].rolling(10).mean()
    
    AwayTeamInfo["Second 3 Game"]=AwayTeamInfo[SecondStat].rolling(3).mean()
    AwayTeamInfo["Second 5 Game"]=AwayTeamInfo[SecondStat].rolling(5).mean()
    AwayTeamInfo["Second 10 Game"]=AwayTeamInfo[SecondStat].rolling(10).mean()
    
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+SecondStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.scatter(AwayTeamInfo.index,AwayTeamInfo[SecondStat])
    ax1.plot(AwayTeamInfo[PomStatAway])
    ax1.plot(AwayTeamInfo["Second 10 Game"])
    ax1.plot(AwayTeamInfo["Second 3 Game"])
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='blue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.scatter(HomeTeamInfo.index,HomeTeamInfo[FirstStat])
    ax2.plot(HomeTeamInfo[PomStatHome])
    ax2.plot(HomeTeamInfo["First 10 Game"])
    ax2.plot(HomeTeamInfo["First 3 Game"])
    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat],color='blue')
    plt.show()
    #f.savefig(pp,format='pdf')
 
        
def getGameInfoFromBartDailyScheduleNov232020(df1,thisDate,Counter):
#df[0].iloc[0][0]   
#u'Nov 22 12:30 PM CT'
    #theDateStringFromBart=df1[0].iloc[0][0]
    #theDayNumber=getTheDayoftheMonth(theDateStringFromBart.split()[1])
    #thisDate=getTheCorrectDateFromStringBartSchedule(theDateStringFromBart.split()[0],theDayNumber)
#df[0].iloc[0][1] 
#'38 Oklahoma vs 17 Wisconsin'
    theTime=df1[0].iloc[Counter][1]
    print(theTime)
    theTeamStringFromBart=df1[0].iloc[Counter][1]
    print(theTeamStringFromBart)
    newhometeam,newawayteam,whereisGame=findHomeTeamandSpreadfromBartSchedule(theTeamStringFromBart)
#df[0].iloc[0][2] 
#'Wisconsin -3.4, 74-70 (63%)'
    print(newhometeam,newawayteam,whereisGame)
    thePointSpreadStringFromBart=df1[0].iloc[Counter][2]
    print(thePointSpreadStringFromBart)
    print(len(thePointSpreadStringFromBart.split()))
    if len(thePointSpreadStringFromBart.split()[len(thePointSpreadStringFromBart.split())-1])<6:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff,hometeampercent,awayteampercent=getProjectedSpreadandTotalFromBartSchedule(thePointSpreadStringFromBart,newawayteam)
        TestFrame = [(thisDate, newawayteam,projectedTotal,awayteampercent), (whereisGame, newhometeam,projectedDiff,hometeampercent)]

        #j=pd.DataFrame.from_records([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [projectedTotal, projectedDiff]),('Win%', [awayteampercent, hometeampercent])])
        j=pd.DataFrame.from_records(TestFrame, columns=['Date', 'Teams','TRank','Win%'])

    else:
        TestFrame2 = [(thisDate, newawayteam,0,0), (whereisGame, newhometeam,0,0)]

        j=pd.DataFrame.from_records(TestFrame2, columns=['Date', 'Teams','TRank','Win%'])

        
        #j=pd.DataFrame.from_records([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [0, 0]),('Win%', [0, 0])])
    return(j,theTime)

        
def getGameInfoFromBartDailyScheduleNov152020(df1,thisDate,gameCounter):
#df[0].iloc[0][0]   
#u'Nov 22 12:30 PM CT'
    #theDateStringFromBart=df1[0].iloc[0][0]
    #theDayNumber=getTheDayoftheMonth(theDateStringFromBart.split()[1])
    #thisDate=getTheCorrectDateFromStringBartSchedule(theDateStringFromBart.split()[0],theDayNumber)
#df[0].iloc[0][1] 
#'38 Oklahoma vs 17 Wisconsin'
    theTeamStringFromBart=df1[0].iloc[0][gameCounter]
    print(theTeamStringFromBart)
    newhometeam,newawayteam,whereisGame=findHomeTeamandSpreadfromBartSchedule(theTeamStringFromBart)
#df[0].iloc[0][2] 
#'Wisconsin -3.4, 74-70 (63%)'
    print(newhometeam,newawayteam,whereisGame)
    thePointSpreadStringFromBart=df1[0].iloc[0][gameCounter+1]
    print(thePointSpreadStringFromBart)
    print(len(thePointSpreadStringFromBart.split()))
    if len(thePointSpreadStringFromBart.split()[len(thePointSpreadStringFromBart.split())-1])<6:
        projectedhometeamscore,projectedawayteamscore,projectedTotal,projectedDiff,hometeampercent,awayteampercent=getProjectedSpreadandTotalFromBartSchedule(thePointSpreadStringFromBart,newhometeam)
        TestFrame = [(thisDate, newawayteam,projectedTotal,awayteampercent), (whereisGame, newhometeam,projectedDiff,hometeampercent)]

        #j=pd.DataFrame.from_records([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [projectedTotal, projectedDiff]),('Win%', [awayteampercent, hometeampercent])])
        j=pd.DataFrame.from_records(TestFrame, columns=['Date', 'Teams','TRank','Win%'])

    else:
        TestFrame2 = [(thisDate, newawayteam,0,0), (whereisGame, newhometeam,0,0)]

        j=pd.DataFrame.from_records(TestFrame2, columns=['Date', 'Teams','TRank','Win%'])

        
        #j=pd.DataFrame.from_records([('Date', [thisDate, whereisGame]), ('Teams', [newawayteam, newhometeam]),('TRank', [0, 0]),('Win%', [0, 0])])
    return(j)

def CreatePomeroyDailyRankings2020(dateList,folderName):
    for i in range(len(dateList)):
        print(dateList[i])
        browser = login("mpgentleman@hotmail.com", 'HxYfLZqncV')
        url = 'https://kenpom.com/archive.php?d='+dateList[i]
        browser.open(url)
        eff = browser.get_current_page()
        table = eff.find_all('table')[0]
        eff_df = pd.read_html(str(table))

        # Dataframe tidying.
        eff_df = eff_df[0]
        eff_df = eff_df.iloc[:, 0:15]
        eff_df.columns = ['Rank','Team', 'Conf', 'AdjEM', 'AdjO', 'AdjO_Rank', 'AdjD', 'AdjD_Rank', 'AdjT', 'AdjT_Rank','Final_Rank', 'Final_AdjEM', 'FinalAdjO','Final_AdjO_Rank', 'FinalAdjD']

        # Remove the header rows that are interjected for readability.
        eff_df = eff_df[eff_df.Team != 'Team']
        # Remove NCAA tourny seeds for previous seasons.
        eff_df['Team'] = eff_df['Team'].str.replace('\d+', '')
        eff_df['Team'] = eff_df['Team'].str.rstrip()
        eff_df = eff_df.dropna()
        #csvName=
        #eff_df.to_csv()
        fileNameToPrint="C:/Users/mpgen/"+folderName+"/PomeroyRankings"+dateList[i].replace('-', '')+".csv"
        eff_df.to_csv(fileNameToPrint,index=False)
        
        
def GetTwoChartsTogetherCompFeb7(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,SecondStat,PomStatHome,VegasStat):
    #HomeTeamInfo["First 3 Game"]=HomeTeamInfo[FirstStat].rolling(3).mean()
    #HomeTeamInfo["First 5 Game"]=HomeTeamInfo[FirstStat].rolling(5).mean()
    #HomeTeamInfo["First 10 Game"]=HomeTeamInfo[FirstStat].rolling(10).mean()
    
    #AwayTeamInfo["Second 3 Game"]=AwayTeamInfo[SecondStat].rolling(3).mean()
    #AwayTeamInfo["Second 5 Game"]=AwayTeamInfo[SecondStat].rolling(5).mean()
    #AwayTeamInfo["Second 10 Game"]=AwayTeamInfo[SecondStat].rolling(10).mean()
    
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+SecondStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.plot(AwayTeamInfo.index,AwayTeamInfo[SecondStat])
    ax1.plot(AwayTeamInfo[FirstStat])
    ax1.plot(AwayTeamInfo[PomStatHome],color='red')
    #ax1.plot(AwayTeamInfo["Second 10 Game"])
    #ax1.plot(AwayTeamInfo["Second 3 Game"])
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat])
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.plot(HomeTeamInfo.index,HomeTeamInfo[SecondStat])
    ax2.plot(HomeTeamInfo[FirstStat])
    ax2.plot(HomeTeamInfo[PomStatHome],color='red')
    #ax2.plot(HomeTeamInfo["First 10 Game"])
    #ax2.plot(HomeTeamInfo["First 3 Game"])
    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat])
    plt.show()
    #f.savefig(pp,format='pdf')
def isNaNStr(num):
    return num != num

def isBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return False
    #myString is None OR myString is empty or blank
    return True
#isNaNStr(df[0].iloc[0][93*5+1+3])



def GetWinLossPercentGameWinnerOverplayingRankingsFeb6(whichList):
    listofRecords=[]
    #SpreadNameWin=ColumntoTest+"ATSWin"
    #SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    winLossList=[]
    #PercentList=[]
    #hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(whichList)):
        #PercentList.append(kenpom_prob(abs(whichList[i][ColumntoTest][1]), std=10.5))
        #theGameStrength=whichList[i]["Trend"][1]-whichList[i]["Trend"][0]
        theGameStrength=whichList[i]["PlayingO"][1]-whichList[i]["PlayingO"][0]
        ATSCover=whichList[i]["Vegas"][1]-whichList[i]["Actual"][1]
        #if whichList[i]["Vegas"][1] > whichList[i]["TG3"][1]:
        #    theGameStrength=theGameStrength+1
        #else:
        #    theGameStrength=theGameStrength-1
        
        if theGameStrength > 0:
            #listofRecords.append(whichList[i]["Teams"][1])
            #listofRecords.append(theGameStrength)
            if ATSCover>=0:
                winLossList.append(1)
                #listofRecords.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][1], round(whichList[i]["PlayingO"][1],2),1]
                listofRecordsOther=[whichList[i]["Teams"][0], round(whichList[i]["PlayingO"][0],2),0]
            
            else:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [0])])                                       
                listofRecords2=[whichList[i]["Teams"][1], round(whichList[i]["PlayingO"][1],2),0]
                listofRecordsOther=[whichList[i]["Teams"][0], round(whichList[i]["PlayingO"][0],2),1]
                #listofRecords.append(0)
        else:
            #listofRecords.append(whichList[i]["Teams"][0])
            #listofRecords.append(theGameStrength)
            
            if ATSCover>0:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [0])])
                listofRecords2=[whichList[i]["Teams"][0], round(whichList[i]["PlayingO"][0],2),0]
                listofRecordsOther=[whichList[i]["Teams"][1], round(whichList[i]["PlayingO"][1],2),1]
                #listofRecords.append(0)
            
            else:
                winLossList.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][0], round(whichList[i]["PlayingO"][0],2),1]
                listofRecordsOther=[whichList[i]["Teams"][1], round(whichList[i]["PlayingO"][1],2),0]
                #listofRecords.append(1)
        listofRecords.append(listofRecords2)
        listofRecords.append(listofRecordsOther)
                
    print(winLossList)
    #print(PercentList)
    #BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,winLossList,listofRecords)


def GetWinLossPercentGameWinnerOverUnder(whichList,TestColumn):
    listofRecords=[]

    winLossList=[]

    for i in range(len(whichList)):

        theOUpick=whichList[i][TestColumn][0]-whichList[i]["Vegas"][0]
        OUCover=whichList[i]["Actual"][0]-whichList[i]["Vegas"][0]

        
        if OUCover>0:
            #listofRecords.append(whichList[i]["Teams"][1])
            #listofRecords.append(theGameStrength)
            if theOUpick>=0:
                winLossList.append(1)

            
            else:
                winLossList.append(0)

        else:
            #listofRecords.append(whichList[i]["Teams"][0])
            #listofRecords.append(theGameStrength)
            
            if theOUpick>0:
                winLossList.append(0)

            
            else:
                winLossList.append(1)

        #listofRecords.append(listofRecords2)
        #listofRecords.append(listofRecordsOther)
                
    #print(winLossList)
    #print(PercentList)
    #BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,winLossList)
#GetWinLossPercentGameWinnerOverUnder(appended_dataframe,"Pom")



def GetWinLossPercentGameWinnerOverplayingFeb6(whichList):
    listofRecords=[]
    #SpreadNameWin=ColumntoTest+"ATSWin"
    #SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    winLossList=[]
    #PercentList=[]
    #hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(whichList)):
        #PercentList.append(kenpom_prob(abs(whichList[i][ColumntoTest][1]), std=10.5))
        #theGameStrength=whichList[i]["Trend"][1]-whichList[i]["Trend"][0]
        theGameStrength=whichList[i]["PlayingO"][1]-whichList[i]["PlayingO"][0]
        ATSCover=whichList[i]["Vegas"][1]-whichList[i]["Actual"][1]
        #if whichList[i]["Vegas"][1] > whichList[i]["TG3"][1]:
        #    theGameStrength=theGameStrength+1
        #else:
        #    theGameStrength=theGameStrength-1
        
        if theGameStrength > 0:
            #listofRecords.append(whichList[i]["Teams"][1])
            #listofRecords.append(theGameStrength)
            if ATSCover>=0:
                winLossList.append(1)
                #listofRecords.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][1], round(theGameStrength,2),1]
            
            else:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [0])])                                       
                listofRecords2=[whichList[i]["Teams"][1], round(theGameStrength,2),0]
                #listofRecords.append(0)
        else:
            #listofRecords.append(whichList[i]["Teams"][0])
            #listofRecords.append(theGameStrength)
            
            if ATSCover>0:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [0])])
                listofRecords2=[whichList[i]["Teams"][0], round(theGameStrength,2),0]
                #listofRecords.append(0)
            
            else:
                winLossList.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][0], round(theGameStrength,2),1]
                #listofRecords.append(1)
        listofRecords.append(listofRecords2)

                
    print(winLossList)
    #print(PercentList)
    #BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,winLossList,listofRecords)

def isNaNStr(num):
    return num != num

def GetWinLossPercentGameWinnerSignalScoreJan29(whichList):
    listofRecords=[]
    #SpreadNameWin=ColumntoTest+"ATSWin"
    #SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    winLossList=[]
    #PercentList=[]
    #hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(whichList)):
        #PercentList.append(kenpom_prob(abs(whichList[i][ColumntoTest][1]), std=10.5))
        theGameStrength=whichList[i]["Trend"][1]-whichList[i]["Trend"][0]
        theOverPlaying=whichList[i]["PlayingO"][1]-whichList[i]["PlayingO"][0]
        ATSCover=whichList[i]["Vegas"][1]-whichList[i]["Actual"][1]
        if whichList[i]["Vegas"][1] > whichList[i]["TG3"][1]:
            theGameStrength=theGameStrength+1
        else:
            theGameStrength=theGameStrength-1
        
        if theGameStrength > 0:
            #listofRecords.append(whichList[i]["Teams"][1])
            #listofRecords.append(theGameStrength)
            if ATSCover>=0:
                winLossList.append(1)
                #listofRecords.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][1], theGameStrength,theOverPlaying,1]
            
            else:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [0])])                                       
                listofRecords2=[whichList[i]["Teams"][1], theGameStrength,theOverPlaying,0]
                #listofRecords.append(0)
        else:
            #listofRecords.append(whichList[i]["Teams"][0])
            #listofRecords.append(theGameStrength)
            
            if ATSCover>0:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [0])])
                listofRecords2=[whichList[i]["Teams"][0], theGameStrength,theOverPlaying,0]
                #listofRecords.append(0)
            
            else:
                winLossList.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][0], theGameStrength,theOverPlaying,1]
                #listofRecords.append(1)
        listofRecords.append(listofRecords2)

                
    print(winLossList)
    #print(PercentList)
    #BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,winLossList,listofRecords)

def GetWinLossPercentGameWinnerSignalScoreJan9(whichList):
    listofRecords=[]
    #SpreadNameWin=ColumntoTest+"ATSWin"
    #SpreadNameLoss=ColumntoTest+"ATSLoss"
    #TotalName=ColumntoTestTotal+"Total"
    winLossList=[]
    #PercentList=[]
    #hh[SpreadNameLoss]=0
    #hh[TotalName]=0
    for i in range(len(whichList)):
        #PercentList.append(kenpom_prob(abs(whichList[i][ColumntoTest][1]), std=10.5))
        theGameStrength=whichList[i]["Trend"][1]-whichList[i]["Trend"][0]
        theOverPlaying=whichList[i]["PlayingO"][1]-whichList[i]["PlayingO"][0]
        ATSCover=whichList[i]["Vegas"][1]-whichList[i]["Actual"][1]
        if whichList[i]["Vegas"][1] > whichList[i]["BG3"][1]:
            theGameStrength=theGameStrength+1
        else:
            theGameStrength=theGameStrength-1
        
        if theGameStrength > 0:
            #listofRecords.append(whichList[i]["Teams"][1])
            #listofRecords.append(theGameStrength)
            if ATSCover>=0:
                winLossList.append(1)
                #listofRecords.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][1], theGameStrength,theOverPlaying,1]
            
            else:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][1]]),('Signal', [theGameStrength]),('Win', [0])])                                       
                listofRecords2=[whichList[i]["Teams"][1], theGameStrength,theOverPlaying,0]
                #listofRecords.append(0)
        else:
            #listofRecords.append(whichList[i]["Teams"][0])
            #listofRecords.append(theGameStrength)
            
            if ATSCover>0:
                winLossList.append(0)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [0])])
                listofRecords2=[whichList[i]["Teams"][0], theGameStrength,theOverPlaying,0]
                #listofRecords.append(0)
            
            else:
                winLossList.append(1)
                #listofRecords2=pd.DataFrame.from_items([('Team', [whichList[i]["Teams"][0]]),('Signal', [theGameStrength]),('Win', [1])])
                listofRecords2=[whichList[i]["Teams"][0], theGameStrength,theOverPlaying,1]
                #listofRecords.append(1)
        listofRecords.append(listofRecords2)

                
    print(winLossList)
    #print(PercentList)
    #BrierScoreToday=CalculateBrierScore(np.array(PercentList),np.array(winLossList))
    NumberofWinningGames=sum(winLossList)
    NumberofLosingGames=len(winLossList)-NumberofWinningGames
    theWinPercent= sum(winLossList)/len(winLossList) 
    #numberofLosses= hh[SpreadNameLoss].sum()
    return(theWinPercent,NumberofWinningGames,NumberofLosingGames,winLossList,listofRecords)

def getTeamSignal(q):
    q["PlayingOverRating"]=q["EMRating5GameExpMA"]-q["PomAdjEMCurrent"]
    
    q["PlayingOverRating3"]=q["EMRating3GameEMA"]-q["PomAdjEMCurrent"]
    q["PlayingOverRatingShift"]=q["PlayingOverRating"].shift(1).fillna(0)
    q["RollingMeanDiffEM"]=q["EMRating"].expanding(1).mean()-q["PomAdjEMCurrent"].expanding(1).mean()
    
    q["RollingMeanDiffEMShift"]=q["RollingMeanDiffEM"].shift(1).fillna(0)
    q["PlayingOverPomeroyTrue"]=np.where(q["PlayingOverRating"]>=0,1,-1)
    q["PlayingOverPomeroyShiftTrue"]=q["PlayingOverPomeroyTrue"].shift(1).fillna(0)
    q["PlayingOverDailyTrue"]=np.where(q["DifOverplayingandEMA"]>=0,1,-1)
    q["RollingMeanDiffEMTrue"]=np.where(q["RollingMeanDiffEM"]>=0,1,-1)
    q["RollingMeanDiffEMShiftTrue"]=np.where(q["RollingMeanDiffEMShift"]>=0,1,-1)
    
    q["SignalSum"]=q["PlayingOverPomeroyTrue"]+q["PlayingOverDailyTrue"]+q["RollingMeanDiffEMTrue"]
    #q["SignalSumShift"]=q["PlayingOverPomeroyShiftTrue"]+q["PlayingOverDailyTrue"]+q["RollingMeanDiffEMShiftTrue"]
    q["SignalSumShift"]=q["SignalSum"].shift(1).fillna(0)
    q["SignalSumShiftAfterGame"]=q["SignalSumShift"].shift(1).fillna(0)
    #q['ATS_MG'] = q['MG_Spread'] + (q['ATSVegas']-q)
    return(q)
    

def CreateDailyTRankRankings2021NewArchive(dateToGet):
    driver = webdriver.Chrome(ChromeDriverManager().install())

    url='https://barttorvik.com/trank-time-machine.php?date='+dateToGet+'&year=2021'

    driver.get(url)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    html = driver.page_source

    soup = BeautifulSoup(html)

    name1=soup.find_all('tr')

    pd_list=[]

    for i in range(len(name1)):
    #print(name1[i].get_text().split('\n')[2])
    #print(name1[i])
        try:
        
            pd_list.append([name1[i].get_text().split('\n')[2],name1[i].get_text().split('\n')[5],name1[i].get_text().split('\n')[7]])
        except:
            print('miss')

    colN=['Team','AdjOE','AdjDE']
    df=pd.DataFrame(pd_list[1:],columns=colN)
    df=sanitizeEntireColumn(df,"Team")
    df.set_index("Team", inplace=True)
    df["AdjOE"]=pd.to_numeric(df["AdjOE"],errors='coerce').fillna(0).astype(np.float64)
    df["AdjDE"]=pd.to_numeric(df["AdjDE"],errors='coerce').fillna(0).astype(np.float64)
    #df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
    df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
    df=df[["AdjEM","AdjOE","AdjDE"]]
    df
    fileNameToPrint="C:/Users/mpgen/TRankDailyRankings2021/"+dateToGet+".csv"

    df.to_csv(fileNameToPrint)
    fileNameToPrint="C:/Users/mpgen/TRankDailyLast10Rankings2021/"+dateToGet+".csv"
    df.to_csv(fileNameToPrint)
    
def CreateDailyTRankRankings2021NewArchiveJan12(dateToGet,dateToWrite):
    driver = webdriver.Chrome(ChromeDriverManager().install())

    url='https://barttorvik.com/trank-time-machine.php?date='+dateToGet+'&year=2021'

    driver.get(url)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    html = driver.page_source

    soup = BeautifulSoup(html)

    name1=soup.find_all('tr')

    pd_list=[]

    for i in range(len(name1)):
    #print(name1[i].get_text().split('\n')[2])
    #print(name1[i])
        try:
        
            pd_list.append([name1[i].get_text().split('\n')[2],name1[i].get_text().split('\n')[5],name1[i].get_text().split('\n')[7]])
        except:
            print('miss')

    colN=['Team','AdjOE','AdjDE']
    df=pd.DataFrame(pd_list[1:],columns=colN)
    df=sanitizeEntireColumn(df,"Team")
    df.set_index("Team", inplace=True)
    df["AdjOE"]=pd.to_numeric(df["AdjOE"],errors='coerce').fillna(0).astype(np.float64)
    df["AdjDE"]=pd.to_numeric(df["AdjDE"],errors='coerce').fillna(0).astype(np.float64)
    #df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
    df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
    df=df[["AdjEM","AdjOE","AdjDE"]]
    df
    fileNameToPrint="C:/Users/mpgen/TRankDailyRankings2021/"+dateToWrite+".csv"

    df.to_csv(fileNameToPrint)
    fileNameToPrint="C:/Users/mpgen/TRankDailyLast10Rankings2021/"+dateToWrite+".csv"
    df.to_csv(fileNameToPrint)
def CreateDailyTRankRankings2022NewArchiveJan12(dateToGet,dateToWrite):
    driver = webdriver.Chrome(ChromeDriverManager().install())

    url='https://barttorvik.com/trank-time-machine.php?date='+dateToGet+'&year=2022'
    url='https://barttorvik.com/trank-time-machine.php?date='+dateToGet
    driver.get(url)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    html = driver.page_source

    soup = BeautifulSoup(html)

    name1=soup.find_all('tr')

    pd_list=[]

    for i in range(len(name1)):
    #print(name1[i].get_text().split('\n')[2])
    #print(name1[i])
        try:
        
            pd_list.append([name1[i].get_text().split('\n')[2],name1[i].get_text().split('\n')[5],name1[i].get_text().split('\n')[7]])
        except:
            print('miss')

    colN=['Team','AdjOE','AdjDE']
    df=pd.DataFrame(pd_list[1:],columns=colN)
    df=sanitizeEntireColumn(df,"Team")
    df.set_index("Team", inplace=True)
    df["AdjOE"]=pd.to_numeric(df["AdjOE"],errors='coerce').fillna(0).astype(np.float64)
    df["AdjDE"]=pd.to_numeric(df["AdjDE"],errors='coerce').fillna(0).astype(np.float64)
    #df["Adj T."]=pd.to_numeric(df["ADJT"],errors='coerce').fillna(0).astype(np.float64)
    df["AdjEM"]=df["AdjOE"]-df["AdjDE"]
    df=df[["AdjEM","AdjOE","AdjDE"]]
    #df
    fileNameToPrint="C:/Users/mpgen/TRankDailyRankings2022/"+dateToWrite+".csv"

    df.to_csv(fileNameToPrint)
    fileNameToPrint="C:/Users/mpgen/TRankDailyLast10Rankings2022/"+dateToWrite+".csv"
    df.to_csv(fileNameToPrint)

def CreatePomeroyDailyRankings2020Jan12(dateList,folderName,dateWrite):
    for i in range(len(dateList)):
        print(dateList)
        browser = login("mpgentleman@hotmail.com", 'HxYfLZqncV')
        url = 'https://kenpom.com/archive.php?d='+dateList[i]
        browser.open(url)
        eff = browser.get_current_page()
        table = eff.find_all('table')[0]
        eff_df = pd.read_html(str(table))

        # Dataframe tidying.
        eff_df = eff_df[0]
        eff_df = eff_df.iloc[:, 0:15]
        #eff_df.columns = ['Rank','Team', 'Conf', 'AdjEM', 'AdjO', 'AdjO_Rank', 'AdjD', 'AdjD_Rank', 'AdjT', 'AdjT_Rank','Final_Rank', 'Final_AdjEM', 'FinalAdjO','Final_AdjO_Rank','AR']
        eff_df.columns = ['Rank','Team', 'Conf', 'Record','AdjEM', 'AdjO', 'AdjO_Rank', 'AdjD', 'AdjD_Rank', 'AdjT', 'AdjT_Rank','Final_Rank', 'Final_AdjEM', 'FinalAdjO','Final_AdjO_Rank']

        # Remove the header rows that are interjected for readability.
        eff_df = eff_df[eff_df.Team != 'Team']
        # Remove NCAA tourny seeds for previous seasons.
        eff_df['Team'] = eff_df['Team'].str.replace('\d+', '')
        eff_df['Team'] = eff_df['Team'].str.rstrip()
        eff_df = eff_df.dropna()
    #eff_df['AdjEM']=eff_df['AdjEM'].astype('int64')
        #csvName=
        #eff_df.to_csv()
        fileNameToPrint="C:/Users/mpgen/"+folderName+"/PomeroyRankings"+dateWrite[i].replace('-', '')+".csv"
        eff_df.to_csv(fileNameToPrint,index=False)
        
    
        
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def sanitizeEntireColumnSchedule(df,ColumntoSanitize):

    for i in range(len(df.index)):
        df.loc[i,ColumntoSanitize]=sanitize_schedule(df.loc[i,ColumntoSanitize])
    return(df)


#name=qq['HOME']
#endings = [' %s' % (i+1) for i in range(16)]
#endings.reverse()
def sanitize_schedule(name):
    endings=[' BTN',' ESPN',' ESPN2',' ESPN+',' SECN',' ACCN',' PAC12',' CBSSN',' LHN',' SECN+',' ESPNU',' ACCNX',' ESPN3'," CBS"," NBCSN"," BIG12"," FS1"," FOX"," BIG12|ESPN+"," ACCNX"," ESPN+"," ESPN3"," FS1"," FS2"]
    for ending in endings:
        if name.endswith(ending):
            name = name[:-len(ending)]
    return(name)



def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 100 # grid size
    x1s = np.linspace(50, 145, num=nb_of_x)
    x2s = np.linspace(50, 145, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = multivariate_normal(
                np.matrix([[x1[i,j]], [x2[i,j]]]), 
                d, mean, covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)

def getSportsReferenceDailySchedule(datetoGetData):
    TeamDB=pd.read_csv("TeamDatabase.csv")
    TeamDB.set_index("SportsReference", inplace=True)
    
    year5=int(datetoGetData.split('-')[2])
    month5=int(datetoGetData.split('-')[0])
    day5=int(datetoGetData.split('-')[1])
    if day5<10:
        newDay='0'+datetoGetData.split('-')[1]
    else:
        newDay=datetoGetData.split('-')[1]
    if month5<10:
        newmonth='0'+datetoGetData.split('-')[0]
    else:
        newmonth=datetoGetData.split('-')[0] 
    
    newDate=datetoGetData.split('-')[2]+newmonth+newDay
    dateTest=datetime(year5, month5,day5)
    games_today = Boxscores(dateTest)
    
    
    dailySch=[]
    for i in range(len(games_today.games[datetoGetData])):
        if not games_today.games[datetoGetData][i]['non_di']:
            awayteamlookup=TeamDB.loc[games_today.games[datetoGetData][i]['away_name'],"OldTRankName"]
            hometeamlookup=TeamDB.loc[games_today.games[datetoGetData][i]['home_name'],"OldTRankName"]
            dailySch.append([datetoGetData,newDate,games_today.games[datetoGetData][i]['away_name'],awayteamlookup,games_today.games[datetoGetData][i]['home_name'],hometeamlookup,games_today.games[datetoGetData][i]['non_di']])
    cols5=['Date','newDate','Away','AwayName','Home','HomeName','non-Div']    
    DailyScheduleTest=pd.DataFrame(dailySch,columns=cols5)
    return(DailyScheduleTest)
    

def getDataFromCsvforGamedayJan3(WhichTeam,OtherTeam,WhichDate):
    #GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2021")
    
    df=GetThisTeamInfoFromCsv(OtherTeam,WhichFile)
    dfOther=GetThisTeamInfoFromCsv(WhichTeam,WhichFile)
    #newDate=convertFullDatetoShortDate(WhichDate)
    newDate=int(WhichDate)
    
    HomeAway=df.loc[df['DateNew'] == newDate,"HomeAway"].values[0]
    ActualOverUnder=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]+df.loc[df['DateNew'] == newDate,"OverUnder"].values[0]
    ActualDif=df.loc[df['DateNew'] == newDate,"ATS"].values[0]-df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
    ActualDif=ActualDif*-1
    ATSVegasis=df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
    OUVegasis=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]
    TRankATSis=df.loc[df['DateNew'] == newDate,"TRankSpread"].values[0]
    TRankOUis=df.loc[df['DateNew'] == newDate,"TRankTotal"].values[0]
    
    TG3ATSis=df.loc[df['DateNew'] == newDate,"B3GSpread"].values[0]
    TG3OUis=df.loc[df['DateNew'] == newDate,"B3GTotal"].values[0]
    MC5ATSis=df.loc[df['DateNew'] == newDate,"MC5Spread"].values[0]
    MC5OUis=df.loc[df['DateNew'] == newDate,"MC5Total"].values[0]
    MC10ATSis=df.loc[df['DateNew'] == newDate,"MC10Spread"].values[0]
    MC10OUis=df.loc[df['DateNew'] == newDate,"MC10Total"].values[0]
    HomeTeamTrending=df.loc[df['DateNew'] == newDate,"SignalSumShift"].values[0]
    AwayTeamTrending=dfOther.loc[dfOther['DateNew'] == newDate,"SignalSumShift"].values[0]
    thePomeroyEstimatedPace=df.loc[df['DateNew'] == newDate,"PomTempo"].values[0]
    theActualTempo=df.loc[df['DateNew'] == newDate,"Pace"].values[0]
    PomATSis=df.loc[df['DateNew'] == newDate,"PomSpread"].values[0]
    PomOUis=df.loc[df['DateNew'] == newDate,"PomTotal"].values[0]
    
    HomeTeamPlayingO=df.loc[df['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
    AwayTeamPlayingO=dfOther.loc[dfOther['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
    AwayScore=(ActualOverUnder+ActualDif)/2
    HomeScore=AwayScore-ActualDif
    #if HomeTeamTrending > 0:
    #    HomeTeamisHot=True
    #else:
    #    HomeTeamisHot=False
    TestF=[(newDate,WhichTeam,AwayTeamTrending,OUVegasis,ActualOverUnder,TRankOUis,PomOUis,TG3OUis,MC5OUis,MC10OUis,AwayScore,thePomeroyEstimatedPace),(HomeAway,OtherTeam,HomeTeamTrending,ATSVegasis,ActualDif,TRankATSis,PomATSis,TG3ATSis,MC5ATSis,MC10ATSis,HomeScore,theActualTempo)]
   
    return(TestF)


def getDataFromCsvforGamedayJan9(WhichTeam,OtherTeam,WhichDate):
    #GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2021")
    
    df=GetThisTeamInfoFromCsv(OtherTeam,WhichFile)
    dfOther=GetThisTeamInfoFromCsv(WhichTeam,WhichFile)
    #newDate=convertFullDatetoShortDate(WhichDate)
    newDate=int(WhichDate)
    
    HomeAway=df.loc[df['DateNew'] == newDate,"HomeAway"].values[0]
    ActualOverUnder=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]+df.loc[df['DateNew'] == newDate,"OverUnder"].values[0]
    ActualDif=df.loc[df['DateNew'] == newDate,"ATS"].values[0]-df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
    ActualDif=ActualDif*-1
    ATSVegasis=df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
    OUVegasis=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]
    TRankATSis=df.loc[df['DateNew'] == newDate,"TRankSpread"].values[0]
    TRankOUis=df.loc[df['DateNew'] == newDate,"TRankTotal"].values[0]
    
    TG3ATSis=df.loc[df['DateNew'] == newDate,"B3GSpread"].values[0]
    TG3OUis=df.loc[df['DateNew'] == newDate,"B3GTotal"].values[0]
    MC5ATSis=df.loc[df['DateNew'] == newDate,"MC5Spread"].values[0]
    MC5OUis=df.loc[df['DateNew'] == newDate,"MC5Total"].values[0]
    MC10ATSis=df.loc[df['DateNew'] == newDate,"MC10Spread"].values[0]
    MC10OUis=df.loc[df['DateNew'] == newDate,"MC10Total"].values[0]
    
    #MG_Rank10ATSis=df.loc[df['DateNew'] == newDate,"MG_Spread"].values[0]
    #MG_Rank10OUis=df.loc[df['DateNew'] == newDate,"MG_Total"].values[0]
    
    HomeTeamTrending=df.loc[df['DateNew'] == newDate,"SignalSumShift"].values[0]
    AwayTeamTrending=dfOther.loc[dfOther['DateNew'] == newDate,"SignalSumShift"].values[0]
    thePomeroyEstimatedPace=df.loc[df['DateNew'] == newDate,"PomTempo"].values[0]
    theActualTempo=df.loc[df['DateNew'] == newDate,"Pace"].values[0]
    PomATSis=df.loc[df['DateNew'] == newDate,"PomSpread"].values[0]
    PomOUis=df.loc[df['DateNew'] == newDate,"PomTotal"].values[0]
    
    HomeTeamPlayingO=df.loc[df['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
    AwayTeamPlayingO=dfOther.loc[dfOther['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
    AwayScore=(ActualOverUnder+ActualDif)/2
    HomeScore=AwayScore-ActualDif
    #if HomeTeamTrending > 0:
    #    HomeTeamisHot=True
    #else:
    #    HomeTeamisHot=False
    TestF=[(newDate,WhichTeam,AwayTeamTrending,OUVegasis,ActualOverUnder,TRankOUis,PomOUis,TG3OUis,MC5OUis,MC10OUis,AwayScore,thePomeroyEstimatedPace,AwayTeamPlayingO),(HomeAway,OtherTeam,HomeTeamTrending,ATSVegasis,ActualDif,TRankATSis,PomATSis,TG3ATSis,MC5ATSis,MC10ATSis,HomeScore,theActualTempo,HomeTeamPlayingO)]
   
    return(TestF)



def getDataFromCsvforGamedayJan10(WhichFile,WhichTeam,OtherTeam,WhichDate):
    #GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2021")
    
    df=GetThisTeamInfoFromCsv(OtherTeam,WhichFile)
    dfOther=GetThisTeamInfoFromCsv(WhichTeam,WhichFile)
    #newDate=convertFullDatetoShortDate(WhichDate)
    newDate=int(WhichDate)
    #print(newDate)
    if len(df.loc[df['DateNew'] == newDate,"HomeAway"])>0:
        HomeAway=df.loc[df['DateNew'] == newDate,"HomeAway"].values[0]
        ActualOverUnder=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]+df.loc[df['DateNew'] == newDate,"OverUnder"].values[0]
        ActualDif=df.loc[df['DateNew'] == newDate,"ATS"].values[0]-df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
        ActualDif=ActualDif*-1
        ATSVegasis=df.loc[df['DateNew'] == newDate,"ATSVegas"].values[0]
        OUVegasis=df.loc[df['DateNew'] == newDate,"OverUnderVegas"].values[0]
        TRankATSis=df.loc[df['DateNew'] == newDate,"TRankSpread"].values[0]
        TRankOUis=df.loc[df['DateNew'] == newDate,"TRankTotal"].values[0]
    
        TG3ATSis=df.loc[df['DateNew'] == newDate,"B3GSpread"].values[0]
        TG3OUis=df.loc[df['DateNew'] == newDate,"B3GTotal"].values[0]
        MC5ATSis=df.loc[df['DateNew'] == newDate,"MC5Spread"].values[0]
        MC5OUis=df.loc[df['DateNew'] == newDate,"MC5Total"].values[0]
        MC10ATSis=df.loc[df['DateNew'] == newDate,"MC10Spread"].values[0]
        MC10OUis=df.loc[df['DateNew'] == newDate,"MC10Total"].values[0]
        #print(MC10OUisM)
        MG_Rank10ATSis=df.loc[df['DateNew'] == newDate,"MG_Spread"].values[0]
        MG_Rank10OUis=df.loc[df['DateNew'] == newDate,"MG_Total"].values[0]
    
        HomeTeamTrending=df.loc[df['DateNew'] == newDate,"SignalSumShift"].values[0]
        AwayTeamTrending=dfOther.loc[dfOther['DateNew'] == newDate,"SignalSumShift"].values[0]
        thePomeroyEstimatedPace=df.loc[df['DateNew'] == newDate,"PomTempo"].values[0]
        theActualTempo=df.loc[df['DateNew'] == newDate,"Pace"].values[0]
        PomATSis=df.loc[df['DateNew'] == newDate,"PomSpread"].values[0]
        PomOUis=df.loc[df['DateNew'] == newDate,"PomTotal"].values[0]
    
        HomeTeamPlayingO=df.loc[df['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
        AwayTeamPlayingO=dfOther.loc[dfOther['DateNew'] == newDate,"PlayingOverRatingShift"].values[0]
        AwayScore=(ActualOverUnder+ActualDif)/2
        HomeScore=AwayScore-ActualDif
        MG_Rank_Dif_ATSis=df.loc[df['DateNew'] == newDate,"MG_Rank_Score_Dif"].values[0]
        #print(MG_Rank_Dif_ATSis)
    #if HomeTeamTrending > 0:
    #    HomeTeamisHot=True
    #else:
    #    HomeTeamisHot=False
        TestF=[(newDate,WhichTeam,AwayTeamTrending,OUVegasis,ActualOverUnder,MG_Rank10OUis,TRankOUis,PomOUis,TG3OUis,MC5OUis,MC10OUis,AwayScore,thePomeroyEstimatedPace,AwayTeamPlayingO,MG_Rank_Dif_ATSis),(HomeAway,OtherTeam,HomeTeamTrending,ATSVegasis,ActualDif,MG_Rank10ATSis,TRankATSis,PomATSis,TG3ATSis,MC5ATSis,MC10ATSis,HomeScore,theActualTempo,HomeTeamPlayingO,MG_Rank_Dif_ATSis)]
   
        return(TestF)



def getGameDayBettingInfoJan3(DfInfo):
    thisList=[DfInfo["Teams"][1],DfInfo["Vegas"][1],DfInfo["TRank"][1],DfInfo["BG3"][1],DfInfo["MC5"][1],DfInfo["MC10"][1],DfInfo["Pom"][1],DfInfo["Actual"][1]]
    f=pd.DataFrame(thisList)
    return(f,thisList)

def getGameDayBettingInfoJan11(DfInfo):
    thisList=[DfInfo["Teams"][1],DfInfo["Vegas"][1],DfInfo["MG"][1],DfInfo["TRank"][1],DfInfo["BG3"][1],DfInfo["MC5"][1],DfInfo["MC10"][1],DfInfo["Pom"][1],DfInfo["Actual"][1],DfInfo["MG_Rank_Dif"][1]]
    f=pd.DataFrame(thisList)
    return(f,thisList)
def getGameDayBettingInfo_OU_Jan11(DfInfo):
    thisList=[DfInfo["Teams"][1],DfInfo["Vegas"][0],DfInfo["MG"][0],DfInfo["TRank"][0],DfInfo["BG3"][0],DfInfo["MC5"][0],DfInfo["MC10"][0],DfInfo["Pom"][0],DfInfo["Actual"][0]]
    f=pd.DataFrame(thisList)
    return(f,thisList)

def Set_MG_Data(BartDF,AwayTeamB1,HomeTeamB1,TheTempo):
    #BartDF.set_index("RevisedSportsReference", inplace=True)

    MG_DF1_ADJ_Off=BartDF[BartDF['stat_name']=='tm_mod_AdjO']
    MG_DF1_ADJ_Def=BartDF[BartDF['stat_name']=='tm_mod_AdjD']

 
    Team1AdjOff=MG_DF1_ADJ_Off.loc[AwayTeamB1,"adj_stat"]
    Team2AdjOff=MG_DF1_ADJ_Off.loc[HomeTeamB1,"adj_stat"]
    
    #Team1AdjT=BartDF.loc[AwayTeamB1,"ADJT"]
    #Team2AdjT=BartDF.loc[HomeTeamB1,"ADJT"]
    
    Team1AdjDef=MG_DF1_ADJ_Def.loc[AwayTeamB1,"adj_stat"]
    Team2AdjDef=MG_DF1_ADJ_Def.loc[HomeTeamB1,"adj_stat"]
    
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,TheTempo,LeagueOE)
    return BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo

def Set_MG_Data_Neutral(BartDF,AwayTeamB1,HomeTeamB1,TheTempo):
    #BartDF.set_index("RevisedSportsReference", inplace=True)
    MG_DF1_ADJ_Off=BartDF[BartDF['stat_name']=='tm_mod_AdjO']
    MG_DF1_ADJ_Def=BartDF[BartDF['stat_name']=='tm_mod_AdjD']


 
    Team1AdjOff=MG_DF1_ADJ_Off.loc[AwayTeamB1,"adj_stat"]
    Team2AdjOff=MG_DF1_ADJ_Off.loc[HomeTeamB1,"adj_stat"]
    
        
    #Team1AdjT=BartDF.loc[AwayTeamB1,"ADJT"]
    #Team2AdjT=BartDF.loc[HomeTeamB1,"ADJT"]

    Team1AdjDef=MG_DF1_ADJ_Def.loc[AwayTeamB1,"adj_stat"]
    Team2AdjDef=MG_DF1_ADJ_Def.loc[HomeTeamB1,"adj_stat"]
    
    BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(Team1AdjOff,Team1AdjDef,Team2AdjOff,Team2AdjDef,TheTempo,LeagueOE)
    return BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo


def Get_MG_Rankings_CSV2021(WhichDate):
    changedDate=WhichDate[4:6]+'_'+WhichDate[6:8]+'_'+WhichDate[:4]
 #20201125
    TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")

    TeamDatabase2.set_index("RevisedSportsReference", inplace=True)

    MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings/tm_seasons_stats_ranks"+changedDate+".csv")
    MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
    MG_DF1.set_index("updated", inplace=True)
    return(MG_DF1)

def Get_MG_Rankings_CSV2022(WhichDate):
    changedDate=WhichDate[4:6]+'_'+WhichDate[6:8]+'_'+WhichDate[:4]
 #20201125
    TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")

    TeamDatabase2.set_index("RevisedSportsReference", inplace=True)
    print(changedDate)
    MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings2022/tm_seasons_stats_ranks"+changedDate+" .csv")
    MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
    MG_DF1.set_index("updated", inplace=True)
    # Changed T and added space
    return(MG_DF1)

 


def createSeabornChartPandasJan13(DfNew):
    gg1=[DfNew["MG"][1],DfNew["MC5"][1]]
    thenewDF=pd.DataFrame({"Spreads":gg1})

    thenewDF["Teams"]=DfNew["Teams"][1]
    thenewDF["Vegas"]=DfNew["Vegas"][1]
    #thenewDF["Actual"]=DfNew["Actual"][1]
    return(thenewDF)
    
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
    plt.show()
    #f.savefig(pp,format='pdf')
    
def GetThisTeamInfoFromCsv(ThisTeam,WhichFile):


    TeamInfo=pd.read_csv("C:/Users/mpgen/"+WhichFile+"/"+ThisTeam+"Data.csv")
    return(TeamInfo)


def getInverseRow(df,i):
    
    season2021InverseTemp=df.iloc[i]
    season2021Inverse=pd.DataFrame(season2021InverseTemp).T
#season2021Copy=pd.DataFrame(season2021InverseTemp).T
    season2021Inverse['tm']=df.iloc[i]['opp']
    season2021Inverse['opp']=df.iloc[i]['tm']
    season2021Inverse['tm_code']=df.iloc[i]['opp_code']
    season2021Inverse['opp_code']=df.iloc[i]['tm_code']
    season2021Inverse['tm_wins']=df.iloc[i]['tm_losses']
    season2021Inverse['tm_losses']=df.iloc[i]['tm_wins']
    season2021Inverse['tm_hca']=int(df.iloc[i]['tm_hca'])*-1
    season2021Inverse['tm_pts']=df.iloc[i]['opp_pts']
    season2021Inverse['opp_pts']=df.iloc[i]['tm_pts']
    season2021Inverse['tm_ptdiff']=df.iloc[i]['tm_ptdiff']*-1
#tm_offensive_rating
    season2021Inverse['tm_offensive_rating']=df.iloc[i]['tm_defensive_rating']
    season2021Inverse['tm_defensive_rating']=df.iloc[i]['tm_offensive_rating']
    season2021Inverse['tm_net_eff']=df.iloc[i]['tm_net_eff']*-1

    season2021Inverse['tm_defensive_rebound_percentage']=df.iloc[i]['opp_defensive_rebound_percentage']
    season2021Inverse['tm_effective_field_goal_percentage']=df.iloc[i]['opp_effective_field_goal_percentage']
    season2021Inverse['tm_field_goal_percentage']=df.iloc[i]['opp_field_goal_percentage']
    season2021Inverse['tm_free_throw_attempt_rate']=df.iloc[i]['opp_free_throw_attempt_rate']
    season2021Inverse['tm_free_throw_percentage']=df.iloc[i]['opp_free_throw_percentage']

    season2021Inverse['opp_defensive_rebound_percentage']=df.iloc[i]['tm_defensive_rebound_percentage']
    season2021Inverse['opp_effective_field_goal_percentage']=df.iloc[i]['tm_effective_field_goal_percentage']
    season2021Inverse['opp_field_goal_percentage']=df.iloc[i]['tm_field_goal_percentage']
    season2021Inverse['opp_free_throw_attempt_rate']=df.iloc[i]['tm_free_throw_attempt_rate']
    season2021Inverse['opp_free_throw_percentage']=df.iloc[i]['tm_free_throw_percentage']
    season2021Inverse['is_home']=0
    season2021Inverse['ATSvalue']=df.iloc[i]['ATSvalue']
    season2021Inverse['tm_offensive_rating']=df.iloc[i]['tm_defensive_rating']
    season2021Inverse['tm_defensive_rating']=df.iloc[i]['tm_offensive_rating']
    season2021Inverse['tm_net_eff']=df.iloc[i]['tm_net_eff']*-1
    #'tm_mod_AdjO','tm_mod_AdjD','tm_mod_net_eff'
    return(season2021Inverse)

def getInverseRow(df,i):
    
    season2021InverseTemp=df.iloc[i]
    season2021Inverse=pd.DataFrame(season2021InverseTemp).T
#season2021Copy=pd.DataFrame(season2021InverseTemp).T
    season2021Inverse['tm']=df.iloc[i]['opp']
    season2021Inverse['opp']=df.iloc[i]['tm']
    season2021Inverse['tm_code']=df.iloc[i]['opp_code']
    season2021Inverse['opp_code']=df.iloc[i]['tm_code']
    season2021Inverse['tm_wins']=df.iloc[i]['tm_losses']
    season2021Inverse['tm_losses']=df.iloc[i]['tm_wins']
    season2021Inverse['tm_hca']=int(df.iloc[i]['tm_hca'])*-1
    season2021Inverse['tm_pts']=df.iloc[i]['opp_pts']
    season2021Inverse['opp_pts']=df.iloc[i]['tm_pts']
    season2021Inverse['tm_ptdiff']=df.iloc[i]['tm_ptdiff']*-1
#tm_offensive_rating
    season2021Inverse['tm_offensive_rating']=df.iloc[i]['tm_defensive_rating']
    season2021Inverse['tm_defensive_rating']=df.iloc[i]['tm_offensive_rating']
    season2021Inverse['tm_net_eff']=df.iloc[i]['tm_net_eff']*-1

    season2021Inverse['tm_defensive_rebound_percentage']=df.iloc[i]['opp_defensive_rebound_percentage']
    season2021Inverse['tm_effective_field_goal_percentage']=df.iloc[i]['opp_effective_field_goal_percentage']
    season2021Inverse['tm_field_goal_percentage']=df.iloc[i]['opp_field_goal_percentage']
    season2021Inverse['tm_free_throw_attempt_rate']=df.iloc[i]['opp_free_throw_attempt_rate']
    season2021Inverse['tm_free_throw_percentage']=df.iloc[i]['opp_free_throw_percentage']

    season2021Inverse['opp_defensive_rebound_percentage']=df.iloc[i]['tm_defensive_rebound_percentage']
    season2021Inverse['opp_effective_field_goal_percentage']=df.iloc[i]['tm_effective_field_goal_percentage']
    season2021Inverse['opp_field_goal_percentage']=df.iloc[i]['tm_field_goal_percentage']
    season2021Inverse['opp_free_throw_attempt_rate']=df.iloc[i]['tm_free_throw_attempt_rate']
    season2021Inverse['opp_free_throw_percentage']=df.iloc[i]['tm_free_throw_percentage']
    season2021Inverse['is_home']=0
    season2021Inverse['ATSvalue']=df.iloc[i]['ATSvalue']
def getInverseRow_Modified(df,i):
    
    season2021InverseTemp=df.iloc[i]
    season2021Inverse=pd.DataFrame(season2021InverseTemp).T
#season2021Copy=pd.DataFrame(season2021InverseTemp).T
    season2021Inverse['tm']=df.iloc[i]['opp']
    season2021Inverse['opp']=df.iloc[i]['tm']
    season2021Inverse['tm_code']=df.iloc[i]['opp_code']
    season2021Inverse['opp_code']=df.iloc[i]['tm_code']
    season2021Inverse['tm_wins']=df.iloc[i]['tm_losses']
    season2021Inverse['tm_losses']=df.iloc[i]['tm_wins']
    season2021Inverse['tm_hca']=int(df.iloc[i]['tm_hca'])*-1
    season2021Inverse['tm_pts']=df.iloc[i]['opp_pts']
    season2021Inverse['opp_pts']=df.iloc[i]['tm_pts']
    season2021Inverse['tm_ptdiff']=df.iloc[i]['tm_ptdiff']*-1
#tm_offensive_rating
    season2021Inverse['tm_offensive_rating']=df.iloc[i]['tm_defensive_rating']
    season2021Inverse['tm_defensive_rating']=df.iloc[i]['tm_offensive_rating']
    season2021Inverse['tm_net_eff']=df.iloc[i]['tm_net_eff']*-1

    season2021Inverse['tm_defensive_rebound_percentage']=df.iloc[i]['opp_defensive_rebound_percentage']
    season2021Inverse['tm_effective_field_goal_percentage']=df.iloc[i]['opp_effective_field_goal_percentage']
    season2021Inverse['tm_field_goal_percentage']=df.iloc[i]['opp_field_goal_percentage']
    season2021Inverse['tm_free_throw_attempt_rate']=df.iloc[i]['opp_free_throw_attempt_rate']
    season2021Inverse['tm_free_throw_percentage']=df.iloc[i]['opp_free_throw_percentage']

    season2021Inverse['opp_defensive_rebound_percentage']=df.iloc[i]['tm_defensive_rebound_percentage']
    season2021Inverse['opp_effective_field_goal_percentage']=df.iloc[i]['tm_effective_field_goal_percentage']
    season2021Inverse['opp_field_goal_percentage']=df.iloc[i]['tm_field_goal_percentage']
    season2021Inverse['opp_free_throw_attempt_rate']=df.iloc[i]['tm_free_throw_attempt_rate']
    season2021Inverse['opp_free_throw_percentage']=df.iloc[i]['tm_free_throw_percentage']
    season2021Inverse['is_home']=0
    season2021Inverse['ATSvalue']=df.iloc[i]['ATSvalue']
    season2021Inverse['tm_mod_AdjO']=df.iloc[i]['tm_mod_AdjD']
    season2021Inverse['tm_mod_AdjD']=df.iloc[i]['tm_mod_AdjO']
    season2021Inverse['tm_mod_net_eff']=df.iloc[i]['tm_mod_net_eff']*-1
    #'tm_mod_AdjO','tm_mod_AdjD','tm_mod_net_eff'
    return(season2021Inverse)


    return(season2021Inverse)




def getDataBetweenDates(startingDate,endingDate):
    # Format is datetime(2000,11,25)
    games = Boxscores(startingDate,endingDate )
    gamesListData=[]
    gamesList=[]
    for key in games.games:
        for i in range(len(games.games[key])):
            game_data = Boxscore(games.games[key][i]['boxscore'])
            #print(game_data.dataframe['location'])
            print(games.games[key][i]['home_name'])
            #print(games.games[key][i]['date'])
            if game_data.dataframe is not None:
                df = game_data.dataframe
        
        
        
        
                tempdateString=game_data.dataframe['date'].index[0].split('-')

                tempintdate=int(tempdateString[0]+tempdateString[1]+tempdateString[2])
                print(tempintdate)
                if games.games[key][i]['non_di']:
                    siteInfo="H"
                else:
                #print(tempdfname[tempdfname['DateNew']==tempintdate]['HomeAway'][0])  
                    teamlookup=TeamDB.loc[games.games[key][i]['home_name'],"OldTRankName"]
                    print(teamlookup)
                    tempdfname=GetThisTeamInfoFromCsv(teamlookup,"TeamDataFiles2021")
                    siteInfo=tempdfname[tempdfname['DateNew']==tempintdate]['HomeAway'].values[0]
                    ATSvalue=tempdfname[tempdfname['DateNew']==tempintdate]['ATS'].values[0]
                    
                    
                gamesListData.append([game_data.dataframe['location'].index[0],games.games[key][i]['non_di'],games.games[key][i]['away_name'],games.games[key][i]['away_score'],games.games[key][i]['home_name'],games.games[key][i]['home_score'],siteInfo,ATSvalue])

                gamesList.append([df.iloc[0].values])
                
                
    return(gamesList,gamesListData)

def getDataBetweenDates_with_Modified_Rankings(startingDate,endingDate):
    # Format is datetime(2000,11,25)
    TeamDB=pd.read_csv("TeamDatabase.csv")
    TeamDB.set_index("SportsReference", inplace=True)
    games = Boxscores(startingDate,endingDate )
    gamesListData=[]
    gamesList=[]
    for key in games.games:
        for i in range(len(games.games[key])):
            game_data = Boxscore(games.games[key][i]['boxscore'])
            #print(game_data.dataframe['location'])
            print(games.games[key][i]['home_name'])
            #print(games.games[key][i]['date'])
            if game_data.dataframe is not None:
                df = game_data.dataframe
        
        
        
        
                tempdateString=game_data.dataframe['date'].index[0].split('-')

                tempintdate=int(tempdateString[0]+tempdateString[1]+tempdateString[2])
                print(tempintdate)
                if games.games[key][i]['non_di']:
                    siteInfo="H"
                else:
                #print(tempdfname[tempdfname['DateNew']==tempintdate]['HomeAway'][0])  
                    teamlookup=TeamDB.loc[games.games[key][i]['home_name'],"OldTRankName"]
                    print(teamlookup)
                    tempdfname=GetThisTeamInfoFromCsv(teamlookup,"TeamDataFiles2021")
                    if len(tempdfname[tempdfname['DateNew']==tempintdate]['HomeAway'])>0:
                        siteInfo=tempdfname[tempdfname['DateNew']==tempintdate]['HomeAway'].values[0]
                        ATSvalue=tempdfname[tempdfname['DateNew']==tempintdate]['ATS'].values[0]
                        PomAdjOECurrent=tempdfname[tempdfname['DateNew']==tempintdate]['PomAdjOECurrent'].values[0]
                        PomAdjDECurrent=tempdfname[tempdfname['DateNew']==tempintdate]['PomAdjDECurrent'].values[0]
                        PomAdjOECurrentOpp=tempdfname[tempdfname['DateNew']==tempintdate]['PomAdjOECurrentOpp'].values[0]
                        PomAdjDECurrentOpp=tempdfname[tempdfname['DateNew']==tempintdate]['PomAdjDECurrentOpp'].values[0]
                    
                gamesListData.append([game_data.dataframe['location'].index[0],games.games[key][i]['non_di'],games.games[key][i]['away_name'],games.games[key][i]['away_score'],games.games[key][i]['home_name'],games.games[key][i]['home_score'],siteInfo,ATSvalue,PomAdjOECurrent,PomAdjDECurrent,PomAdjOECurrentOpp,PomAdjDECurrentOpp])
                gamesList.append([df.iloc[0].values])
                
                
    return(gamesList,gamesListData)

def createGameDayStatsDF(gamesList):
    dfColstm_opp=['opp_assist_percentage', 'opp_assists', 'opp_block_percentage',
       'opp_blocks', 'opp_defensive_rating',
       'opp_defensive_rebound_percentage', 'opp_defensive_rebounds',
       'opp_effective_field_goal_percentage', 'opp_field_goal_attempts',
       'opp_field_goal_percentage', 'opp_field_goals',
       'opp_free_throw_attempt_rate', 'opp_free_throw_attempts',
       'opp_free_throw_percentage', 'opp_free_throws', 'opp_losses',
       'opp_minutes_played', 'opp_offensive_rating',
       'opp_offensive_rebound_percentage', 'opp_offensive_rebounds',
       'opp_personal_fouls', 'opp_points', 'opp_ranking',
       'opp_steal_percentage', 'opp_steals', 'opp_three_point_attempt_rate',
       'opp_three_point_field_goal_attempts',
       'opp_three_point_field_goal_percentage',
       'opp_three_point_field_goals', 'opp_total_rebound_percentage',
       'opp_total_rebounds', 'opp_true_shooting_percentage',
       'opp_turnover_percentage', 'opp_turnovers',
       'opp_two_point_field_goal_attempts',
       'opp_two_point_field_goal_percentage', 'opp_two_point_field_goals',
       'opp_win_percentage', 'opp_wins', 'date', 'tm_assist_percentage',
       'tm_assists', 'tm_block_percentage', 'tm_blocks',
       'tm_defensive_rating', 'tm_defensive_rebound_percentage',
       'tm_defensive_rebounds', 'tm_effective_field_goal_percentage',
       'tm_field_goal_attempts', 'tm_field_goal_percentage',
       'tm_field_goals', 'tm_free_throw_attempt_rate',
       'tm_free_throw_attempts', 'tm_free_throw_percentage',
       'tm_free_throws', 'tm_losses', 'tm_minutes_played',
       'tm_offensive_rating', 'tm_offensive_rebound_percentage',
       'tm_offensive_rebounds', 'tm_personal_fouls', 'tm_points',
       'tm_ranking', 'tm_steal_percentage', 'tm_steals',
       'tm_three_point_attempt_rate', 'tm_three_point_field_goal_attempts',
       'tm_three_point_field_goal_percentage',
       'tm_three_point_field_goals', 'tm_total_rebound_percentage',
       'tm_total_rebounds', 'tm_true_shooting_percentage',
       'tm_turnover_percentage', 'tm_turnovers',
       'tm_two_point_field_goal_attempts',
       'tm_two_point_field_goal_percentage', 'tm_two_point_field_goals',
       'tm_win_percentage', 'tm_wins', 'location', 'losing_abbr',
       'losing_name', 'pace', 'winner', 'winning_abbr', 'winning_name']
    Test=pd.DataFrame(gamesList,columns=['Temp'])
    #df.iloc[0].values
    dfTemp=pd.DataFrame(Test['Temp'].to_list(), columns=dfColstm_opp)
    return(dfTemp)

def createDFwithDiv1andSite_with_Modified(gamesListDataInfo):
    
    othercols=['DateT','game_id','Div1','opp','opp_pts','tm','tm_pts','site','ATSvalue','tm_PomAdjO','tm_PomAdjD','opp_PomAdjO','opp_PomAdjD']
    dfTemp2=pd.DataFrame(gamesListDataInfo, columns=othercols)
    #print(len(dfTemp2))
    string=dfTemp2.iloc[0]['game_id'].split('-')

    dfTemp2['newDate']=string[1]+'-'+string[2]+'-'+string[0]
    dfT=dfTemp2
    dfT.drop(dfT[dfT['Div1'] ==True].index, inplace = True)
    dfT=dfT.dropna(how='all')
    dfT = dfT[dfT['opp_pts'].notna()]


    dfT['tm_ptdiff']=dfT['tm_pts']-dfT['opp_pts']



    content4=np.unique(dfT[['opp', 'tm']].values)
    index = dict(zip(content4, [ x for x in range(len(content4))])) 
    dfT['opp_code']= dfT['opp'].map(index)
    dfT['tm_code']= dfT['tm'].map(index)
    dfT.drop(dfT[dfT['Div1'] ==True].index, inplace = True)
    dfT=dfT.dropna(how='all')
    dfT = dfT[dfT['opp_pts'].notna()]
    dfT['is_home']=0
    dfT['is_neutral']=0
    dfT['tm_hca']=0

    for i in range(len(dfT)):
        #df.iloc[i]['is_home']=
        dateWanted=dfT.iloc[i]['newDate']
        homeTeam=dfT.iloc[i]['tm']
        #e=gamesbyDate[dateWanted]
        if dfT.iloc[i]['site'] == 'N':
            dfT.iloc[i, dfT.columns.get_loc('is_home')] = 1
            dfT.iloc[i, dfT.columns.get_loc('is_neutral')] = 1
            dfT.iloc[i, dfT.columns.get_loc('tm_hca')] = 0
        else:
            dfT.iloc[i, dfT.columns.get_loc('is_neutral')] = 0
            dfT.iloc[i, dfT.columns.get_loc('is_home')] = 1
            dfT.iloc[i, dfT.columns.get_loc('tm_hca')] = 1
        
    return(dfT)    

def createDFwithDiv1andSite(gamesListDataInfo):
    
    othercols=['game_id','Div1','opp','opp_pts','tm','tm_pts','site','ATSvalue']
    dfTemp2=pd.DataFrame(gamesListDataInfo, columns=othercols)
    string=dfTemp2.iloc[0]['game_id'].split('-')

    dfTemp2['newDate']=string[1]+'-'+string[2]+'-'+string[0]
    dfT=dfTemp2
    dfT.drop(dfT[dfT['Div1'] ==True].index, inplace = True)
    dfT=dfT.dropna(how='all')
    dfT = dfT[dfT['opp_pts'].notna()]


    dfT['tm_ptdiff']=dfT['tm_pts']-dfT['opp_pts']



    content4=np.unique(dfT[['opp', 'tm']].values)
    index = dict(zip(content4, [ x for x in range(len(content4))])) 
    dfT['opp_code']= dfT['opp'].map(index)
    dfT['tm_code']= dfT['tm'].map(index)
    dfT.drop(dfT[dfT['Div1'] ==True].index, inplace = True)
    dfT=dfT.dropna(how='all')
    dfT = dfT[dfT['opp_pts'].notna()]
    dfT['is_home']=0
    dfT['is_neutral']=0
    dfT['tm_hca']=0

    for i in range(len(dfT)):
        #df.iloc[i]['is_home']=
        dateWanted=dfT.iloc[i]['newDate']
        homeTeam=dfT.iloc[i]['tm']
        #e=gamesbyDate[dateWanted]
        if dfT.iloc[i]['site'] == 'N':
            dfT.iloc[i, dfT.columns.get_loc('is_home')] = 1
            dfT.iloc[i, dfT.columns.get_loc('is_neutral')] = 1
            dfT.iloc[i, dfT.columns.get_loc('tm_hca')] = 0
        else:
            dfT.iloc[i, dfT.columns.get_loc('is_neutral')] = 0
            dfT.iloc[i, dfT.columns.get_loc('is_home')] = 1
            dfT.iloc[i, dfT.columns.get_loc('tm_hca')] = 1
        
        

    return(dfT)


def combineBothSportsDF(dfGame,dfSite):
    season2021=pd.concat([dfGame,dfSite],axis=1)
    season2021['season']='2021'

    season2021['tm_net_eff']=season2021['tm_offensive_rating']-season2021['tm_defensive_rating']
    
    
    
    
    
    
    goodColumns=['season','newDate','game_id','tm_code','tm','opp_code','opp','is_home', 'is_neutral', 'tm_hca', 'tm_wins', 'tm_losses','tm_pts','opp_pts','tm_ptdiff','pace','tm_defensive_rating', 'tm_defensive_rebound_percentage','tm_effective_field_goal_percentage','tm_field_goal_percentage','tm_free_throw_attempt_rate','tm_free_throw_percentage','tm_offensive_rating', 'opp_defensive_rating', 'opp_defensive_rebound_percentage','opp_effective_field_goal_percentage','opp_field_goal_percentage','opp_free_throw_attempt_rate','opp_free_throw_percentage','opp_offensive_rating','tm_net_eff','ATSvalue']
    season2021Final=season2021[goodColumns]
    return(season2021Final)
def combineBothSportsDF_Jan2021(dfGame,dfSite):
    season2021=pd.concat([dfGame,dfSite],axis=1)
    season2021['season']='2021'

    season2021['tm_net_eff']=season2021['tm_offensive_rating']-season2021['tm_defensive_rating']
    
    
    
    
    
    
    goodColumns=['season','newDate','game_id','tm_code','tm','opp_code','opp','site','is_home', 'is_neutral', 'tm_hca', 'tm_wins', 'tm_losses','tm_pts','opp_pts','tm_ptdiff','pace','tm_defensive_rating', 'tm_defensive_rebound_percentage','tm_effective_field_goal_percentage','tm_field_goal_percentage','tm_free_throw_attempt_rate','tm_free_throw_percentage','tm_offensive_rating', 'opp_defensive_rating', 'opp_defensive_rebound_percentage','opp_effective_field_goal_percentage','opp_field_goal_percentage','opp_free_throw_attempt_rate','opp_free_throw_percentage','opp_offensive_rating','tm_net_eff','ATSvalue','opp_PomAdjO','opp_PomAdjD']
    season2021Final=season2021[goodColumns]
    return(season2021Final)


def Add_Modified_Eff(season2021,LeagueAdjO,LeagueAdjD):

    season2021['tm_mod_AdjO']=0
    season2021['tm_mod_AdjD']=0
    
    for i in range(len(season2021)):
        #print(i)
        RawAdjO=season2021.iloc[i]['tm_offensive_rating']
        RawAdjD=season2021.iloc[i]['tm_defensive_rating']
        PomAdjOECurrentOpp=season2021.iloc[i]['opp_PomAdjO']
        PomAdjDECurrentOpp=season2021.iloc[i]['opp_PomAdjD']
        if season2021.iloc[i]['site'] == 'N':
    
            season2021.iloc[i, season2021.columns.get_loc('tm_mod_AdjO')],season2021.iloc[i, season2021.columns.get_loc('tm_mod_AdjD')]=get_Modified_Efficiency(LeagueAdjO,LeagueAdjD,RawAdjO,RawAdjD,PomAdjOECurrentOpp,PomAdjDECurrentOpp,0)

        elif season2021.iloc[i]['site'] == 'H':
            season2021.iloc[i, season2021.columns.get_loc('tm_mod_AdjO')],season2021.iloc[i, season2021.columns.get_loc('tm_mod_AdjD')]=get_Modified_Efficiency(LeagueAdjO,LeagueAdjD,RawAdjO,RawAdjD,PomAdjOECurrentOpp,PomAdjDECurrentOpp,1.4)
        else:
            
            season2021.iloc[i, season2021.columns.get_loc('tm_mod_AdjO')],season2021.iloc[i, season2021.columns.get_loc('tm_mod_AdjD')]=get_Modified_Efficiency(LeagueAdjO,LeagueAdjD,RawAdjO,RawAdjD,PomAdjOECurrentOpp,PomAdjDECurrentOpp,-1.4)

    
    
    
    #season2021['tm_mod_net_eff']=season2021['tm_mod_AdjO']-season2021['tm_mod_AdjD']
    #goodColumns=['season','newDate','game_id','tm_code','tm','opp_code','opp','is_home', 'is_neutral', 'tm_hca', 'tm_wins', 'tm_losses','tm_pts','opp_pts','tm_ptdiff','pace','tm_defensive_rating', 'tm_defensive_rebound_percentage','tm_effective_field_goal_percentage','tm_field_goal_percentage','tm_free_throw_attempt_rate','tm_free_throw_percentage','tm_offensive_rating', 'opp_defensive_rating', 'opp_defensive_rebound_percentage','opp_effective_field_goal_percentage','opp_field_goal_percentage','opp_free_throw_attempt_rate','opp_free_throw_percentage','opp_offensive_rating','tm_net_eff','ATSvalue','tm_mod_net_eff','tm_mod_AdjO','tm_mod_AdjD']
    #season2021Final=season2021[goodColumns]
    
    return(season2021)


def addInverseRowtoDF(season2021Final):
    goodColumns2=['season',
     'newDate',
     'game_id',
     'tm_code',
     'tm',
     'opp_code',
     'opp',
     'is_home',
     'is_neutral',
     'tm_hca',
     'tm_wins',
     'tm_losses',
     'tm_pts',
     'opp_pts',
     'tm_ptdiff',
     'pace',
     'tm_offensive_rating',
     'tm_defensive_rating',
     'tm_net_eff',
     'tm_defensive_rebound_percentage',
     'tm_effective_field_goal_percentage',
     'tm_field_goal_percentage',
     'tm_free_throw_attempt_rate',
     'tm_free_throw_percentage',
     'opp_defensive_rating',
     'opp_defensive_rebound_percentage',
     'opp_effective_field_goal_percentage',
     'opp_field_goal_percentage',
     'opp_free_throw_attempt_rate',
     'opp_free_throw_percentage',
     'opp_offensive_rating','ATSvalue','tm_turnover_percentage', 'tm_offensive_rebound_percentage','opp_turnover_percentage','opp_offensive_rebound_percentage']
    Final=season2021Final[goodColumns2]
    Final = Final[Final['opp_pts'].notna()]

    SeasonTemp=Final
    for i in range(len(Final.index)):
        newRow=getInverseRow(Final,i)
    #print(i)
    #print(newRow)
        SeasonTemp=pd.concat([newRow,SeasonTemp])
    return(SeasonTemp)
def addInverseRowtoDF_Modified(season2021Final):
    goodColumns2=['season','DateT',
     'newDate',
     'game_id',
     'tm_code',
     'tm',
     'opp_code',
     'opp',
     'is_home',
     'is_neutral',
     'tm_hca',
     'tm_wins',
     'tm_losses',
     'tm_pts',
     'opp_pts',
     'tm_ptdiff',
     'pace',
     'tm_offensive_rating',
     'tm_defensive_rating',
     'tm_net_eff',
     'tm_defensive_rebound_percentage',
     'tm_effective_field_goal_percentage',
     'tm_field_goal_percentage',
     'tm_free_throw_attempt_rate',
     'tm_free_throw_percentage',
     'opp_defensive_rating',
     'opp_defensive_rebound_percentage',
     'opp_effective_field_goal_percentage',
     'opp_field_goal_percentage',
     'opp_free_throw_attempt_rate',
     'opp_free_throw_percentage',
     'opp_offensive_rating','ATSvalue','tm_mod_AdjO','tm_mod_AdjD','tm_mod_net_eff','tm_turnover_percentage', 'tm_offensive_rebound_percentage','opp_turnover_percentage','opp_offensive_rebound_percentage']
    Final=season2021Final[goodColumns2]
    Final = Final[Final['opp_pts'].notna()]

    SeasonTemp=Final
    for i in range(len(Final.index)):
        newRow=getInverseRow_Modified(Final,i)
    #print(i)
    #print(newRow)
        SeasonTemp=pd.concat([newRow,SeasonTemp])
    return(SeasonTemp)
def createSeasonAvgDF(dfT):
    
    teamListData=[]
    teams = Teams()
    for team in teams:
        print(team.name)
    #dfG=team.dataframe
        teamListData.append([team.name,team.simple_rating_system,team.pace,team.net_rating,team.offensive_rating,team.effective_field_goal_percentage,team.free_throw_attempt_rate,team.offensive_rebound_percentage,team.turnover_percentage,team.opp_effective_field_goal_percentage,team.opp_free_throw_attempt_rate,team.opp_offensive_rebound_percentage,team.opp_turnover_percentage,team.opp_offensive_rating])

    teamcolListforExport=['tm_old',
     'tm_net_eff',
     'pace','net_rating',
     'tm_offensive_rating',
     'tm_effective_field_goal_percentage',
     'tm_free_throw_attempt_rate',
     'tm_offensive_rebound_percentage',
     'tm_turnover_percentage',
     'opp_effective_field_goal_percentage',
     'opp_free_throw_attempt_rate',
     'opp_offensive_rebound_percentage',
     'opp_turnover_percentage',
     'opp_offensive_rating']

    teamexport=pd.DataFrame(teamListData,columns=teamcolListforExport)
    teamexport.to_csv("SportsReferenceTeamsStats.csv")
    team_stats2021=pd.read_csv("SportsReferenceTeamsStats.csv")
    TeamDB5=pd.read_csv("TeamDatabase.csv")
    #TeamDB.set_index("SportsReference", inplace=True)
    #contentstats=np.unique(TeamDB['RevisedSportsReference'].values)
    contentstats=dict(zip(TeamDB5.RevisedSportsReference,TeamDB5.SportsReference))
    #index5stats = dict(zip(contentstats, [ x for x in range(len(contentstats))])) 
    team_stats2021['tm']= team_stats2021['tm_old'].map(contentstats)


    #team_stats2021['tm']=TeamDatabase.loc[NameofThisTeam5,"RevisedSportsReference"]
    content4=np.unique(dfT[['opp', 'tm']].values)
    index = dict(zip(content4, [ x for x in range(len(content4))])) 
    team_stats2021['tm_code']= team_stats2021['tm'].map(index)
    team_stats2021['tm_defensive_rating']=team_stats2021['tm_offensive_rating']-team_stats2021['tm_net_eff']
    team_stats2021['season']=2021
    team_stats2021['ATSvalue']=0
    return(team_stats2021)


def combineteamStats_seasonStats(team_stats2021,season_stats2021):
    metrics2021 = pd.read_csv("metrics2021_margin_modified.csv")

    team_stats2021['season']=2022
    team_seasons_stats_melt = pd.melt(
      team_stats2021[['season', 'tm_code', 'tm'] + season_stats2021['stat_name'].tolist()],
      id_vars = ['season', 'tm_code', 'tm'],
      value_vars = metrics2021['stat_name'].tolist(),
      var_name = 'stat_name',
      value_name = 'raw_stat'
      )
    return(team_seasons_stats_melt)

def getRidgeRegressionDF(team_games,season_stats2021):
    team_games['season']=team_games['season'].astype(str).astype(int)
    # Create combo fields for season_tm & season_opp to be used in regression
    team_games['season_tm'] = (team_games['season'].map(str) + '_' + 
      team_games['tm_code'].map(str))

    team_games['season_opp'] = (team_games['season'].map(str) + '_' + 
      team_games['opp_code'].map(str))

    # Set up ridge regression model parameters
    reg = linear_model.Ridge(alpha = 1, fit_intercept = True)

    reg_results_collection = pd.DataFrame(columns = ['season', 'stat_name',
      'coef_name', 'ridge_reg_coef', 'ridge_reg_value'])


    # Iterate through each season and stat
    for index, row in season_stats2021.iterrows():
    
        this_season_game_stat = (team_games[team_games['season'] == row['season']]
        [['season_tm', 'tm_hca', 'season_opp', row['stat_name']]].
        reset_index()
        )
        print(this_season_game_stat)
        this_season_game_stat.iloc[:,-1:]=this_season_game_stat.iloc[:,-1:].fillna(0)
        this_season_game_dummy_vars = pd.get_dummies(
        this_season_game_stat[['season_tm', 'tm_hca', 'season_opp']]
        )
        print(row['stat_name'])
        print(row['season'])
      # Fit ridge regression to given statistic using season game dummy variables
        reg.fit(
        X = this_season_game_dummy_vars,
        y = this_season_game_stat[row['stat_name']]
        )

      # Extract regression coefficients and put into data fram with coef names
        this_reg_results = pd.DataFrame(
        {
          # Add season and name of stat for this set of results
          'season': row['season'],
          'stat_name': row['stat_name'],
          # Coef name, which contains both season and tm_code
          'coef_name': this_season_game_dummy_vars.columns.values,
          # Coef that results from ridge regression
          'ridge_reg_coef': reg.coef_
        }
        )

      # Add intercept back in to reg coef to get 'adjusted' value
        this_reg_results['ridge_reg_value'] = (this_reg_results['ridge_reg_coef'] + 
        reg.intercept_
        )

        reg_results_collection = pd.concat([
        reg_results_collection,
        this_reg_results
        ],
        ignore_index = True
        )

    reg_results_collection

    # Only keep ratings from 'season_tm' perspective for this
    tm_seasons_adjresults = (reg_results_collection[
      reg_results_collection['coef_name'].str.slice(0, 9) == 'season_tm'].
      rename(columns = {"ridge_reg_value": "adj_stat"}).
      reset_index(drop = True)
      )
    tm_seasons_adjresults['tm_code'] = (tm_seasons_adjresults['coef_name'].
      str.slice(15)).map(float)

    return(tm_seasons_adjresults)

def createSeasonRankings(metrics2021,team_seasons_stats_melt,tm_seasons_adjresults):
    #@title Turn Each Stat into 0-100 Rating, Rank Each Team Season by Stat
    # Merge raw and adjusted team stats
    tm_seasons_raw_adj = pd.merge(
      team_seasons_stats_melt, 
      tm_seasons_adjresults.drop(['coef_name', 'ridge_reg_coef'], axis = 1),
      how = 'outer',
      on = ['season', 'tm_code', 'stat_name']
      )

    # Mean/SD calculated across teams instead of games to count each team same;
    # not weighting teams w/ more games (likely those that advanced further) more
    season_stats_meansd = (tm_seasons_raw_adj.
      groupby(['season', 'stat_name'])['raw_stat', 'adj_stat'].
      agg(['mean', 'std']).
      reset_index()
      )

    # Rename columns to get rid of hierarchical index
    season_stats_meansd.columns = ['season', 'stat_name', 'season_raw_stat_mean', 
      'season_raw_stat_sd', 'season_adj_stat_mean', 'season_adj_stat_sd']

    # Merge in season-level mean/sd, and whether metric ranks asc/desc
    tm_seasons_stats_ranks = pd.merge(
      pd.merge(
        tm_seasons_raw_adj,
        season_stats_meansd,
        how = 'left',
        on = ['season', 'stat_name']
      ),  
      metrics2021[['stat_name', 'rank_asc']],
      how = 'left',
      on = ['stat_name']
      )

    # Create field to help rank D1 teams only (not including group of Non-D1 teams)
    tm_seasons_stats_ranks['rk_group'] = np.where(
      tm_seasons_stats_ranks['tm_code'] == -1, 'NON-D1', 'D1'
      )

    # Loop over raw and adjusted version of each metric
    for metric_type in ['raw', 'adj']:
      # Translate each stat to 0-100 'rating' by normalizing vs season mean/sd
      tm_seasons_stats_ranks[metric_type + '_rtg'] = scipy.stats.norm.cdf(
        np.where(tm_seasons_stats_ranks['rank_asc'], -1, 1) *
        (tm_seasons_stats_ranks[metric_type + '_stat'] 
          - tm_seasons_stats_ranks['season_' + metric_type + '_stat_mean']) /
         tm_seasons_stats_ranks['season_' + metric_type + '_stat_sd']
        ) * 100

      # Rank on rtg field, in correct dir, since 0 = worse & 100 = better by design
      tm_seasons_stats_ranks[metric_type + '_rk'] = np.where(
        # No rankings for group of Non-D1 teams
        tm_seasons_stats_ranks['rk_group'] == 'NON-D1', np.nan,
        (tm_seasons_stats_ranks.
        # Group by season, stat name, & rank group (so Non-D1 teams don't 'mix' in)
        groupby(['season', 'stat_name', 'rk_group'])[metric_type + '_rtg'].
        rank(ascending = False)
        ))

    return(tm_seasons_stats_ranks)


def AddHistoricalRankingsJan152021(TeamDatabase,DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFilesStarter2021'
    WhichFile='TeamDataFiles2021'
    
    
    # change folder
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
# change for 2021        
        w=GetDailyTRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(DftoChange.loc[i,"DateNew"])
        #print(w)
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV2021(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV2021(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #print(OppData)
        
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])
        
        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
 

        TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)



        #MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings/tm_seasons_stats_ranks"+dateforRankings5+".csv")
        MG_Rank=Get_MG_Rankings_CSV2021(DftoChange.loc[i,"DateNew"])
        MG_Rank["updated"]=update_type(MG_Rank.tm,TeamDatabase2.UpdatedTRankName)
        MG_Rank.set_index("updated", inplace=True)
 
        MG_DF1_ADJ_Off=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjO']
        MG_DF1_ADJ_Def=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjD']
        MG_DF1_ADJ_Off['adj_stat']=MG_DF1_ADJ_Off['adj_stat'].fillna(0)
        MG_DF1_ADJ_Def['adj_stat']=MG_DF1_ADJ_Def['adj_stat'].fillna(0)
        
        MG_DF1_ADJ_Margin_Eff=MG_Rank[MG_Rank['stat_name']=='tm_margin_net_eff']
        MG_DF1_ADJ_Margin_Eff['adj_stat']=MG_DF1_ADJ_Margin_Eff['adj_stat'].fillna(0)
        
        AwayTeamM=TeamDatabase2.loc[TheOpponent,"SportsReference"]
        HomeTeamM=TeamDatabase2.loc[NameofThisTeam5,"SportsReference"]
        print(NameofThisTeam5)
        print(TheOpponent)
        print(HomeTeamM)
        print(AwayTeamM)
        DftoChange.loc[i,"AdjOE_MG"]=MG_DF1_ADJ_Off.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG"]=MG_DF1_ADJ_Def.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG"]=DftoChange.loc[i,"AdjOE_MG"]-DftoChange.loc[i,"AdjDE_MG"]
        
        DftoChange.loc[i,"Adj_Margin_EM_MG"]=MG_DF1_ADJ_Margin_Eff.loc[HomeTeamM,"adj_stat"]

        
        DftoChange.loc[i,"AdjOE_MG_Opp"]=MG_DF1_ADJ_Off.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG_Opp"]=MG_DF1_ADJ_Def.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG_Opp"]=DftoChange.loc[i,"AdjOE_MG_Opp"]-DftoChange.loc[i,"AdjDE_MG_Opp"]
        DftoChange.loc[i,"Adj_Margin_EM_MG_Opp"]=MG_DF1_ADJ_Margin_Eff.loc[AwayTeamM,"adj_stat"]

        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],float(DftoChange.loc[i,"PomAdjTCurrentOpp"]),DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],float(DftoChange.loc[i,"PomAdjTCurrent"]),LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],thePGameTempo,LeagueOE)
            MG_Rank_Score_Dif=get_MG_Margin_Dif_Ratio(MG_Rank,AwayTeamM,HomeTeamM,thePGameTempo)
            #Set_MG_Data(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],DftoChange.loc[i,"AdjO3ExpMAS"],OppData.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData[OppData['Opponent']==NameofThisTeam5]['AdjO3ExpMAS'].values[0],OppData[OppData['Opponent']==NameofThisTeam5]['AdjD3ExpMAS'].values[0],DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)

            if len(OppData.index)<2:
                OppDataT=OppData
            else:
                #Oppindex=OppData[OppData['DateNew']==DftoChange[DftoChange['DateNew']==DftoChange.loc[i,"DateNew"]].iloc[0]['DateNew']].index[0]
                OppDataT=OppData
                #OppDataT=OppData.iloc[:Oppindex]
                
            if len(DftoChange.index)<2:
                DftoChangeT=DftoChange
            else:
                DftoChangeT=DftoChange
            
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScore(OppDataT,DftoChangeT,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScore(OppDataT,DftoChangeT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreJan31(dateofThisGame,OppDataT,DftoChangeT,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreJan31(dateofThisGame,OppDataT,DftoChangeT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
 
                
        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                MG_Rank_Score_Dif=get_MG_Margin_Dif_Ratio(MG_Rank,HomeTeamM,AwayTeamM,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData.loc[i,"AdjO3ExpMAS"],OppData.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData[OppData['Opponent']==NameofThisTeam5]['AdjO3ExpMAS'].values[0],OppData[OppData['Opponent']==NameofThisTeam5]['AdjD3ExpMAS'].values[0],thePGameTempo,LeagueOE)


                if len(OppData.index)<2:
                    OppDataT=OppData
                else:
                    #Oppindex=OppData[OppData['DateNew']==DftoChange[DftoChange['DateNew']==DftoChange.loc[i,"DateNew"]].iloc[0]['DateNew']].index[0]
                    OppDataT=OppData
                    #OppDataT=OppData.iloc[:Oppindex]
                    #OppDataT=OppData[:-1]
                if len(DftoChange.index)<2:
                    DftoChangeT=DftoChange
                else:
                    DftoChangeT=DftoChange
                
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScore(DftoChangeT,OppDataT,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScore(DftoChangeT,OppDataT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreJan31(dateofThisGame,DftoChangeT,OppDataT,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreJan31(dateofThisGame,DftoChangeT,OppDataT,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:
                print(dateofThisGame)
                print(DftoChange.loc[i,"AdjO3ExpMAS"])
                #print(OppData)
                #print(DftoChange)
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                MG_Rank_Score_Dif=get_MG_Margin_Dif_Neutral(MG_Rank,AwayTeamM,HomeTeamM,thePGameTempo)
                MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
                #Set_MG_Data_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData.loc[i,"AdjO3ExpMAS"],OppData.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange[DftoChange['DateNew']==dateofThisGame]['AdjO3ExpMAS'].values[0],DftoChange[DftoChange['DateNew']==dateofThisGame]['AdjD3ExpMAS'].values[0],OppData[OppData['DateNew']==dateofThisGame]['AdjO3ExpMAS'].values[0],OppData[OppData['DateNew']==dateofThisGame]['AdjD3ExpMAS'].values[0],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData[OppData['Opponent']==NameofThisTeam5]['AdjO3ExpMAS'].values[0],OppData[OppData['Opponent']==NameofThisTeam5]['AdjD3ExpMAS'].values[0],thePGameTempo,LeagueOE)

                if len(OppData.index)<2:
                    OppDataT=OppData
                else:
                    #OppDataT=OppData[:-1]
                    #Oppindex=OppData[OppData['DateNew']==DftoChange[DftoChange['DateNew']==DftoChange.loc[i,"DateNew"]].iloc[0]['DateNew']].index[0]
                    OppDataT=OppData
                    #OppDataT=OppData.iloc[:Oppindex]
                if len(DftoChange.index)<2:
                    DftoChangeT=DftoChange
                else:
                    DftoChangeT=DftoChange
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChangeT,OppDataT,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChangeT,OppDataT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtJan31(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtJan31(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            MHomeTeamSpread=MHomeTeamSpread*-1
            #MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            #theEstimatedSpread=theEstimatedSpread*-1
            #theEstimatedSpread10G=theEstimatedSpread10G*-1
        if whereisThisGame == "H":
            #CurHomeTeamSpread=CurHomeTeamSpread*-1
            #PHomeTeamSpread=PHomeTeamSpread*-1
            #BHomeTeamSpread=BHomeTeamSpread*-1
            #MHomeTeamSpread=MHomeTeamSpread*-1
            B3GHomeTeamSpread=theEstimatedSpread*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            theEstimatedSpread=theEstimatedSpread*-1
            theEstimatedSpread10G=theEstimatedSpread10G*-1
            MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
            
        B3GHomeTeamSpread=theEstimatedSpread*-1
        #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
        theEstimatedSpread=theEstimatedSpread*-1
        theEstimatedSpread10G=theEstimatedSpread10G*-1 
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        #DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"B3GTotal"]=BTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange.loc[i,"MG_Spread"]=MHomeTeamSpread
        DftoChange.loc[i,"MG_Total"]=MTotalPoints
        DftoChange.loc[i,"MG_Rank_Score_Dif"]=MG_Rank_Score_Dif
        DftoChange.loc[i,"MG_Rank_Total"]=MTotalPoints
        #DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        #edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        #DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSumShifted"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MG_Spread","MG_Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MG_Rank_Score_Dif","MG_Rank_Total")
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
   
        
    return (DftoChange)

def AddHistoricalRankingsJan152022(TeamDatabase,DftoChange,NameofThisTeam5):
    WhichFile='TeamDataFilesStarter2022'
    WhichFile='TeamDataFiles2022'
    
    
    # change folder
    DftoChange["AdjOECurrent"]=0
    DftoChange["AdjDECurrent"]=0
    DftoChange["AdjEMCurrent"]=0
    DftoChange["AdjOECurrentOpp"]=0
    DftoChange["AdjDECurrentOpp"]=0
    DftoChange["AdjEMCurrentOpp"]=0
    
    #DftoChange["AdjOECurrentOppEMA"]=0
    #DftoChange["AdjDECurrentOppEMA"]=0
    
    
    DftoChange["PomAdjOECurrent"]=0
    DftoChange["PomAdjDECurrent"]=0
    DftoChange["PomAdjEMCurrent"]=0
    DftoChange["PomAdjTCurrent"]=0
    DftoChange["PomSpread"]=0
    DftoChange["PomTotal"]=0
    DftoChange["PomTempo"]=0
    
    DftoChange["TRankSpread"]=0
    DftoChange["TRankTotal"]=0
    
    DftoChange["B3GSpread"]=0
    DftoChange["B3GTotal"]=0
    DftoChange["TRankLast10Spread"]=0
    DftoChange["TRankLast10Total"]=0
    
    DftoChange["MC5Spread"]=0
    DftoChange["MC5Total"]=0
    
    
    DftoChange["PomAdjOECurrentOpp"]=0
    DftoChange["PomAdjDECurrentOpp"]=0
    DftoChange["PomAdjEMCurrentOpp"]=0
    DftoChange["PomAdjTCurrentOpp"]=0
    
    DftoChange["AdjOELast10"]=0
    DftoChange["AdjDELast10"]=0
    DftoChange["AdjEMLast10"]=0
    DftoChange["AdjOEOppLast10"]=0
    DftoChange["AdjDEOppLast10"]=0
    DftoChange["AdjEMOppLast10"]=0 
    
    DftoChange["MC10Spread"]=0
    DftoChange["MC10Total"]=0
    DftoChange["MC10Edge"]=0
    WhichTeamP=sanitize_teamname(TeamDatabase.loc[NameofThisTeam5,"PomeroyName"])
   
    for i in range(len(DftoChange.index)):
# change for 2021        
        w=GetDailyTRankDataCSV2022(DftoChange.loc[i,"DateNew"])
        #print(DftoChange.loc[i,"DateNew"])
        #print(w)
        TheOpponent=DftoChange.loc[i,"Opponent"]
        TheOpponentP=sanitize_teamname(TeamDatabase.loc[TheOpponent,"PomeroyName"])
        whereisThisGame=DftoChange.loc[i,"HomeAway"]
        dateofThisGame=int(DftoChange.loc[i,"DateNew"])
        PomCurrent=GetDailyPomeroyDataCSV2022(DftoChange.loc[i,"DateNew"])
        BartLast10=GetDailyLast10TRankDataCSV2022(DftoChange.loc[i,"DateNew"])
        #print(str(dateofThisGame))
        
        OppData=GetThisTeamInfoFromCsv(TheOpponent,WhichFile)
        #print(OppData)
        
        #f=OppData[OppData['DateNew'] == dateofThisGame].index
        #OppDataCut=OppData[0:f[0]]
        #OppEMAdate=int(OppData.loc[f[0],'DateNew'])
        
        
        #DftoChange["AdjOECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjO3GameEMA"]
        #DftoChange["AdjDECurrentOppEMA"]=OppData[OppData['DateNew'] == OppEMAdate]["AdjD3GameEMA"]
    
        DftoChange.loc[i,"AdjOECurrent"]=w.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrent"]=w.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrent"]=w.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOECurrentOpp"]=w.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDECurrentOpp"]=w.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMCurrentOpp"]=w.loc[TheOpponent]["AdjEM"]
      
        DftoChange.loc[i,"PomAdjOECurrent"]=PomCurrent.loc[WhichTeamP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrent"]=PomCurrent.loc[WhichTeamP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrent"]=PomCurrent.loc[WhichTeamP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrent"]=PomCurrent.loc[WhichTeamP]["AdjT"]
        
        DftoChange.loc[i,"PomAdjOECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjO"]
        DftoChange.loc[i,"PomAdjDECurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjD"]
        DftoChange.loc[i,"PomAdjEMCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjEM"]
        DftoChange.loc[i,"PomAdjTCurrentOpp"]=PomCurrent.loc[TheOpponentP]["AdjT"]
        


        
        DftoChange.loc[i,"AdjOELast10"]=BartLast10.loc[NameofThisTeam5]["AdjOE"]
        DftoChange.loc[i,"AdjDELast10"]=BartLast10.loc[NameofThisTeam5]["AdjDE"]
        DftoChange.loc[i,"AdjEMLast10"]=BartLast10.loc[NameofThisTeam5]["AdjEM"]
        DftoChange.loc[i,"AdjOEOppLast10"]=BartLast10.loc[TheOpponent]["AdjOE"]
        DftoChange.loc[i,"AdjDEOppLast10"]=BartLast10.loc[TheOpponent]["AdjDE"]
        DftoChange.loc[i,"AdjEMOppLast10"]=BartLast10.loc[TheOpponent]["AdjEM"]
 

        TeamDatabase2=pd.read_csv("C:/Users/mpgen/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)



        #MG_DF1=pd.read_csv("C:/Users/mpgen/MGRankings/tm_seasons_stats_ranks"+dateforRankings5+".csv")
        MG_Rank=Get_MG_Rankings_CSV2022(DftoChange.loc[i,"DateNew"])
        MG_Rank["updated"]=update_type(MG_Rank.tm,TeamDatabase2.UpdatedTRankName)
        MG_Rank.set_index("updated", inplace=True)
 
        MG_DF1_ADJ_Off=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjO']
        MG_DF1_ADJ_Def=MG_Rank[MG_Rank['stat_name']=='tm_mod_AdjD']
        MG_DF1_ADJ_Off['adj_stat']=MG_DF1_ADJ_Off['adj_stat'].fillna(0)
        MG_DF1_ADJ_Def['adj_stat']=MG_DF1_ADJ_Def['adj_stat'].fillna(0)
        
        MG_DF1_ADJ_Margin_Eff=MG_Rank[MG_Rank['stat_name']=='tm_margin_net_eff']
        MG_DF1_ADJ_Margin_Eff['adj_stat']=MG_DF1_ADJ_Margin_Eff['adj_stat'].fillna(0)
        
        AwayTeamM=TeamDatabase2.loc[TheOpponent,"SportsReference"]
        HomeTeamM=TeamDatabase2.loc[NameofThisTeam5,"SportsReference"]
        print(NameofThisTeam5)
        print(TheOpponent)
        print(HomeTeamM)
        print(AwayTeamM)
        DftoChange.loc[i,"AdjOE_MG"]=MG_DF1_ADJ_Off.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG"]=MG_DF1_ADJ_Def.loc[HomeTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG"]=DftoChange.loc[i,"AdjOE_MG"]-DftoChange.loc[i,"AdjDE_MG"]
        
        DftoChange.loc[i,"Adj_Margin_EM_MG"]=MG_DF1_ADJ_Margin_Eff.loc[HomeTeamM,"adj_stat"]

        
        DftoChange.loc[i,"AdjOE_MG_Opp"]=MG_DF1_ADJ_Off.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjDE_MG_Opp"]=MG_DF1_ADJ_Def.loc[AwayTeamM,"adj_stat"]
        DftoChange.loc[i,"AdjEM_MG_Opp"]=DftoChange.loc[i,"AdjOE_MG_Opp"]-DftoChange.loc[i,"AdjDE_MG_Opp"]
        DftoChange.loc[i,"Adj_Margin_EM_MG_Opp"]=MG_DF1_ADJ_Margin_Eff.loc[AwayTeamM,"adj_stat"]

        
        if whereisThisGame == "H":
            
            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],float(DftoChange.loc[i,"PomAdjTCurrentOpp"]),DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],float(DftoChange.loc[i,"PomAdjTCurrent"]),LeagueTempo,LeagueOE)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],thePGameTempo,LeagueOE)
            CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],thePGameTempo,LeagueOE)
            MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],thePGameTempo,LeagueOE)
            MG_Rank_Score_Dif=get_MG_Margin_Dif_Ratio(MG_Rank,AwayTeamM,HomeTeamM,thePGameTempo)
            #Set_MG_Data(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3GameEMA"],DftoChange.loc[i,"AdjD3GameEMA"],["AdjO3GameEMA"],TeamHistoryData[TeamHistoryData['DateNew'] == dateofTheGame]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,OppData,DftoChange,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],DftoChange.loc[i,"AdjO3ExpMAS"],OppData.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)
            #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(OppData[OppData['Opponent']==NameofThisTeam5]['AdjO3ExpMAS'].values[0],OppData[OppData['Opponent']==NameofThisTeam5]['AdjD3ExpMAS'].values[0],DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)

            if len(OppData.index)<2:
                OppDataT=OppData
            else:
                #Oppindex=OppData[OppData['DateNew']==DftoChange[DftoChange['DateNew']==DftoChange.loc[i,"DateNew"]].iloc[0]['DateNew']].index[0]
                OppDataT=OppData
                #OppDataT=OppData.iloc[:Oppindex]
                
            if len(DftoChange.index)<2:
                DftoChangeT=DftoChange
            else:
                DftoChangeT=DftoChange
            
            #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScore(OppDataT,DftoChangeT,MonteCarloNumberofGames,10000,thePGameTempo)
            #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScore(OppDataT,DftoChangeT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreJan31(dateofThisGame,OppDataT,DftoChangeT,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreJan31(dateofThisGame,OppDataT,DftoChangeT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
 
                
        
        else:
            if whereisThisGame == "A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePrediction(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                MG_Rank_Score_Dif=get_MG_Margin_Dif_Ratio(MG_Rank,HomeTeamM,AwayTeamM,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData.loc[i,"AdjO3ExpMAS"],OppData.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBart(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData[OppData['Opponent']==NameofThisTeam5]['AdjO3ExpMAS'].values[0],OppData[OppData['Opponent']==NameofThisTeam5]['AdjD3ExpMAS'].values[0],thePGameTempo,LeagueOE)


                if len(OppData.index)<2:
                    OppDataT=OppData
                else:
                    #Oppindex=OppData[OppData['DateNew']==DftoChange[DftoChange['DateNew']==DftoChange.loc[i,"DateNew"]].iloc[0]['DateNew']].index[0]
                    OppDataT=OppData
                    #OppDataT=OppData.iloc[:Oppindex]
                    #OppDataT=OppData[:-1]
                if len(DftoChange.index)<2:
                    DftoChangeT=DftoChange
                else:
                    DftoChangeT=DftoChange
                
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScore(DftoChangeT,OppDataT,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScore(DftoChangeT,OppDataT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreJan31(dateofThisGame,DftoChangeT,OppDataT,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreJan31(dateofThisGame,DftoChangeT,OppDataT,SecondMonteCarloNumberofGames,10000,thePGameTempo)

            else:
                print(dateofThisGame)
                print(DftoChange.loc[i,"AdjO3ExpMAS"])
                #print(OppData)
                #print(DftoChange)
                #test1=Df
                #test2=OppData
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NewgetGamePredictionNeutralCourt(DftoChange.loc[i,"PomAdjOECurrent"],DftoChange.loc[i,"PomAdjDECurrent"],DftoChange.loc[i,"PomAdjTCurrent"],DftoChange.loc[i,"PomAdjOECurrentOpp"],DftoChange.loc[i,"PomAdjDECurrentOpp"],DftoChange.loc[i,"PomAdjTCurrentOpp"],LeagueTempo,LeagueOE)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOECurrent"],DftoChange.loc[i,"AdjDECurrent"],DftoChange.loc[i,"AdjOECurrentOpp"],DftoChange.loc[i,"AdjDECurrentOpp"],thePGameTempo,LeagueOE)
                CurAwayTeamScore,CurHomeTeamScore,CurTotalPoints,CurHomeTeamSpread,theCurGameTemp0=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOELast10"],DftoChange.loc[i,"AdjDELast10"],DftoChange.loc[i,"AdjOEOppLast10"],DftoChange.loc[i,"AdjDEOppLast10"],thePGameTempo,LeagueOE)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjOE_MG"],DftoChange.loc[i,"AdjDE_MG"],DftoChange.loc[i,"AdjOE_MG_Opp"],DftoChange.loc[i,"AdjDE_MG_Opp"],thePGameTempo,LeagueOE)
                MG_Rank_Score_Dif=get_MG_Margin_Dif_Neutral(MG_Rank,AwayTeamM,HomeTeamM,thePGameTempo)
                MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
                #Set_MG_Data_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.iloc[len(DftoChange.index)-2]["AdjO3GameEMA"],DftoChange.iloc[len(DftoChange.index)-2]["AdjD3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjO3GameEMA"],OppData.iloc[len(OppData.index)-2]["AdjD3GameEMA"],thePGameTempo,LeagueOE)
                #print(B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo)
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtDec23(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData.loc[i,"AdjO3ExpMAS"],OppData.loc[i,"AdjD3ExpMAS"],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange[DftoChange['DateNew']==dateofThisGame]['AdjO3ExpMAS'].values[0],DftoChange[DftoChange['DateNew']==dateofThisGame]['AdjD3ExpMAS'].values[0],OppData[OppData['DateNew']==dateofThisGame]['AdjO3ExpMAS'].values[0],OppData[OppData['DateNew']==dateofThisGame]['AdjD3ExpMAS'].values[0],thePGameTempo,LeagueOE)
                #B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NewgetGamePredictionBartNeutralCourt(DftoChange.loc[i,"AdjO3ExpMAS"],DftoChange.loc[i,"AdjD3ExpMAS"],OppData[OppData['Opponent']==NameofThisTeam5]['AdjO3ExpMAS'].values[0],OppData[OppData['Opponent']==NameofThisTeam5]['AdjD3ExpMAS'].values[0],thePGameTempo,LeagueOE)

                if len(OppData.index)<2:
                    OppDataT=OppData
                else:
                    #OppDataT=OppData[:-1]
                    #Oppindex=OppData[OppData['DateNew']==DftoChange[DftoChange['DateNew']==DftoChange.loc[i,"DateNew"]].iloc[0]['DateNew']].index[0]
                    OppDataT=OppData
                    #OppDataT=OppData.iloc[:Oppindex]
                if len(DftoChange.index)<2:
                    DftoChangeT=DftoChange
                else:
                    DftoChangeT=DftoChange
                #r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourt(DftoChangeT,OppDataT,MonteCarloNumberofGames,10000,thePGameTempo)
                #Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourt(DftoChangeT,OppDataT,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                r,r1,theEstimatedTotal,theEstimatedSpread=getMonteCarloGameScoreNeutralCourtJan31(dateofThisGame,DftoChange,OppData,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=getMonteCarloGameScoreNeutralCourtJan31(dateofThisGame,DftoChange,OppData,SecondMonteCarloNumberofGames,10000,thePGameTempo)

        if whereisThisGame != "H":
            CurHomeTeamSpread=CurHomeTeamSpread*-1
            PHomeTeamSpread=PHomeTeamSpread*-1
            BHomeTeamSpread=BHomeTeamSpread*-1
            MHomeTeamSpread=MHomeTeamSpread*-1
            #MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            #theEstimatedSpread=theEstimatedSpread*-1
            #theEstimatedSpread10G=theEstimatedSpread10G*-1
        if whereisThisGame == "H":
            #CurHomeTeamSpread=CurHomeTeamSpread*-1
            #PHomeTeamSpread=PHomeTeamSpread*-1
            #BHomeTeamSpread=BHomeTeamSpread*-1
            #MHomeTeamSpread=MHomeTeamSpread*-1
            B3GHomeTeamSpread=theEstimatedSpread*-1
            #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
            theEstimatedSpread=theEstimatedSpread*-1
            theEstimatedSpread10G=theEstimatedSpread10G*-1
            MG_Rank_Score_Dif=MG_Rank_Score_Dif*-1
            
        B3GHomeTeamSpread=theEstimatedSpread*-1
        #B3GHomeTeamSpread=B3GHomeTeamSpread*-1
        theEstimatedSpread=theEstimatedSpread*-1
        theEstimatedSpread10G=theEstimatedSpread10G*-1 
        DftoChange.loc[i,"PomSpread"]=PHomeTeamSpread
        DftoChange.loc[i,"PomTotal"]=PTotalPoints
        DftoChange.loc[i,"PomTempo"]=thePGameTempo
        DftoChange.loc[i,"TRankSpread"]=BHomeTeamSpread
        DftoChange.loc[i,"TRankTotal"]=BTotalPoints
        DftoChange.loc[i,"B3GSpread"]=B3GHomeTeamSpread
        #DftoChange.loc[i,"B3GTotal"]=B3GTotalPoints
        DftoChange.loc[i,"B3GTotal"]=BTotalPoints
        DftoChange.loc[i,"TRankLast10Spread"]=CurHomeTeamSpread
        DftoChange.loc[i,"TRankLast10Total"]=CurTotalPoints
        DftoChange.loc[i,"MC5Spread"]=theEstimatedSpread
        DftoChange.loc[i,"MC5Total"]=theEstimatedTotal
        DftoChange.loc[i,"MC10Spread"]=theEstimatedSpread10G
        DftoChange.loc[i,"MC10Total"]=theEstimatedTotal10G
        DftoChange.loc[i,"MG_Spread"]=MHomeTeamSpread
        DftoChange.loc[i,"MG_Total"]=MTotalPoints
        DftoChange.loc[i,"MG_Rank_Score_Dif"]=MG_Rank_Score_Dif
        DftoChange.loc[i,"MG_Rank_Total"]=MTotalPoints
        #DftoChange["OppDifOverplayingandEMA"]=OppData.iloc[len(OppData.index)-2]["DifOverplayingandEMA"]
        
        #edgeAgainstVegasSpread10G=stats.percentileofscore(Spread10G, DftoChange.loc[i,"ATSVegas"])
 
        #DftoChange.loc[i,"MC10Edge"]=edgeAgainstVegasSpread10G
  
    DftoChange["GameDifRating"]=DftoChange["EMRating"]-DftoChange["PomAdjEMCurrent"]
    DftoChange["DifCumSum"]=DftoChange["GameDifRating"].cumsum()
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSum"].ewm(span=5,adjust=False).mean()
    
    DftoChange["DifCumSumShifted"]=DftoChange["DifCumSum"].shift(1).fillna(0)
    DftoChange["DifCumSumEMA"]=DftoChange["DifCumSumEMA"].shift(1).fillna(0)
    DftoChange["DifOverplayingandEMA"]=DftoChange["DifCumSum"]-DftoChange["DifCumSumEMA"]
  
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"PomSpread","PomTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"B3GSpread","B3GTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankSpread","TRankTotal")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"TRankLast10Spread","TRankLast10Total")
 
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC5Spread","MC5Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MG_Spread","MG_Total")
    DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MG_Rank_Score_Dif","MG_Rank_Total")
    #DftoChange=AddWinLossAgainstSpreadDifferentColumns(DftoChange,"MC10Spread","MC10Total")
    DftoChange['MG_SpreadWinATSResult'] = DftoChange.apply(getMGWinRecord, axis=1)
    DftoChange['VegasImpliedWinPercent']=DftoChange['ATSVegas'].apply(getWinPercentfromSpread)
    DftoChange['MGImpliedWinPercent']=DftoChange['MG_Spread'].apply(getWinPercentfromSpread)

        
    return (DftoChange)
def get_Modified_Efficiency(LeagueAdjO,LeagueAdjD,RawAdjO,RawAdjD,PomAdjOECurrentOpp,PomAdjDECurrentOpp,HCAPercent):
    

    ModAdjOHome=(RawAdjO*LeagueAdjO)/(PomAdjDECurrentOpp+HCAPercent)-HCAPercent
    ModAdjDHome=(RawAdjD*LeagueAdjD)/(PomAdjOECurrentOpp-HCAPercent)+HCAPercent
    return(ModAdjOHome,ModAdjDHome)

def getDailyPredictionTeams_OU_Dec18(ListofModels,VegasSpreadToTest,MCSpread,MCEdge,HomeTeam1,timeH):
    listofTeamsSelected=[]
    for q in range(len(ListofModels)):
    
        #if VegasSpreadToTest<0:
        if VegasSpreadToTest<ListofModels[q]:
            teamSelection='Over'
        else:
            teamSelection='Under'

        listofTeamsSelected.append(teamSelection)
    listofTeamsSelected.append(HomeTeam1)
    #totalScore=HomeSigScore-AwaySigScore
    listofTeamsSelected.append(timeH)
    #totalOverplay=HomeOver-AwayOver
    #listofTeamsSelected.append(totalOverplay)
    listofTeamsSelected.append(VegasSpreadToTest)
    listofTeamsSelected.append(MCSpread)
    listofTeamsSelected.append(MCEdge)

    return(listofTeamsSelected)

def get_List_of_Teams_Played(todaysGames):
    column_values = todaysGames[["AwayName", "HomeName"]].values
    unique_values =  np.unique(column_values)
    Unique_List=unique_values.tolist()
    return(Unique_List)

def get_MG_Margin_Dif(MG_Rank,AwayTeam,HomeTeam,GameTempo):
    MG_DF1_Mar=MG_Rank[MG_Rank['stat_name']=='tm_margin_net_eff']
    
    Team1MarAway=MG_DF1_Mar.loc[AwayTeam,"adj_stat"]
    Team2MarHome=MG_DF1_Mar.loc[HomeTeam,"adj_stat"]
    Score_Dif=(100+Team2MarHome)*GameTempo/100+1.5-((100+Team1MarAway)*GameTempo/100-1.5)
    return(Score_Dif)
def get_MG_Margin_Dif_Neutral(MG_Rank,AwayTeam,HomeTeam,GameTempo):
    MG_DF1_Mar=MG_Rank[MG_Rank['stat_name']=='tm_margin_net_eff']
    
    Team1MarAway=MG_DF1_Mar.loc[AwayTeam,"adj_stat"]
    Team2MarHome=MG_DF1_Mar.loc[HomeTeam,"adj_stat"]
    Score_Dif=(100+Team2MarHome)*GameTempo/100-((100+Team1MarAway)*GameTempo/100)
    
    return(Score_Dif)

def get_MG_Margin_Dif_Ratio(MG_Rank,AwayTeam,HomeTeam,GameTempo):
    MG_DF1_Mar=MG_Rank[MG_Rank['stat_name']=='tm_margin_net_eff']
    
    Team1MarAway=MG_DF1_Mar.loc[AwayTeam,"adj_stat"]
    Team2MarHome=MG_DF1_Mar.loc[HomeTeam,"adj_stat"]
    Score_Dif=(100+Team2MarHome)*GameTempo/100*1.014-((100+Team1MarAway)*GameTempo/100*.986)
    return(Score_Dif)
def getMGWinRecord(s):
    if (s['MG_SpreadWinATS'] == 1):
        return 3
    else:
        return -3
def getWinPercentfromSpread(x):
    output=1-scipy.stats.norm(0,10.5).cdf(x)
    return(output)
   
