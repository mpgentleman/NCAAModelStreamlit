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
from numpy import exp,array,zeros,inf
import requests
import io
from pandas.api.types import is_numeric_dtype
import os
from lets_plot import *
LetsPlot.setup_html()
from streamlit_letsplot import st_letsplot
import lets_plot
from lets_plot import *
import json
import Bracketology as Bkt
from streamlit_option_menu import option_menu
from lets_plot.frontend_context._configuration import _as_html

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image # image
from numpy.random import random

from collections import Counter, OrderedDict, defaultdict

#import os, os.path

from functools import wraps
from time import time
import scipy
from hashlib import md5
from itertools import chain
#from __future__ import division
from random import choice, shuffle
from numpy.random import random #import only one function from somewhere
from numpy.random import randint
#from numpy.random import random
#from itertools import izip_longest
from itertools import zip_longest
import datetime
#from numpy.random import randint

import math
import streamlit.components.v1 as components
from collections import namedtuple
from datetime import datetime, timedelta


def showTeamLetsPlotCharts2024(test1,VegasMetric,shortMVA,longMVA,scoringMetric,mytitle,myTeam):
    
    result1 = pd.melt(test1, id_vars=["Opp"], value_vars=[VegasMetric], var_name="Metric", value_name="Value")
    resultT = pd.melt(test1, id_vars=["Opp"], value_vars=[longMVA, shortMVA], var_name="Metric", value_name="Value")
    result2 = pd.melt(test1, id_vars=["Opp"], value_vars=[scoringMetric], var_name="Metric", value_name="Value")
    chart1 = ggplot(resultT)+geom_line(aes(x='Opp', y='Value',color='Metric'),stat="identity")+geom_point(aes(x='Opp', y='Value'),stat="identity",data=result2)+ ggsize(700, 600)+ ylab(VegasMetric)+geom_bar(aes(x='Opp', y='Value'),stat="identity",data=result1)+ggtitle(myTeam+' '+mytitle)
    
    p2 = gggrid([chart1], ncol=1)+ ggsize(800, 500)
    plot_dict = p2.as_dict()
    components.html(_as_html(plot_dict), height=500 + 20,width=800 + 20,scrolling=True,)
def showTeamLetsPlotOverplayingCharts2024(test1,VegasMetric,shortMVA,longMVA,mytitle,myTeam):
    
    result1 = pd.melt(test1, id_vars=["Opp"], value_vars=[VegasMetric], var_name="Metric", value_name="Value")
    resultT = pd.melt(test1, id_vars=["Opp"], value_vars=[longMVA, shortMVA], var_name="Metric", value_name="Value")
    #result2 = pd.melt(test1, id_vars=["Opp"], value_vars=[scoringMetric], var_name="Metric", value_name="Value")
    chart1 = ggplot(resultT)+geom_line(aes(x='Opp', y='Value',color='Metric'),stat="identity")+ ggsize(700, 600)+ ylab(VegasMetric)+geom_bar(aes(x='Opp', y='Value'),stat="identity",data=result1)+ggtitle(myTeam+' '+mytitle)
    
    p2 = gggrid([chart1], ncol=1)+ ggsize(800, 500)
    plot_dict = p2.as_dict()
    components.html(_as_html(plot_dict), height=500 + 20,width=800 + 20,scrolling=True,)
    
def showTeamLetsPlotMultiCharts2024(test1,VegasMetric,shortMVA,longMVA,ranking,scoringMetric,mytitle,myTeam):
    
    result1 = pd.melt(test1, id_vars=["Opp"], value_vars=[VegasMetric], var_name="Metric", value_name="Value")
    resultT = pd.melt(test1, id_vars=["Opp"], value_vars=[longMVA, shortMVA,ranking], var_name="Metric", value_name="Value")
    result2 = pd.melt(test1, id_vars=["Opp"], value_vars=[scoringMetric], var_name="Metric", value_name="Value")
    chart1 = ggplot(resultT)+geom_line(aes(x='Opp', y='Value',color='Metric'),stat="identity")+geom_point(aes(x='Opp', y='Value'),stat="identity",data=result2)+ ggsize(700, 600)+ ylab(VegasMetric)+geom_bar(aes(x='Opp', y='Value'),stat="identity",data=result1)+ggtitle(myTeam+' '+mytitle)
    
    p2 = gggrid([chart1], ncol=1)+ ggsize(800, 500)
    plot_dict = p2.as_dict()
    components.html(_as_html(plot_dict), height=500 + 20,width=800 + 20,scrolling=True,)    
def html(body):
    st.markdown(body, unsafe_allow_html=True)


def card_begin_str(header):
    return (
        "<style>div.card{background-color:lightblue;border-radius: 5px;box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>"
        '<div class="card">'
        '<div class="container">'
        f"<h3><b>{header}</b></h3>"
    )


def card_end_str():
    return "</div></div>"


def card(header, body):
    lines = [card_begin_str(header), f"<p>{body}</p>", card_end_str()]
    html("".join(lines))


def br(n):
    html(n * "<br>")




def getIndividualPlayerData():
    url = 'https://barttorvik.com/2024_all_advgames.json.gz'
    response = requests.get(url)
    text = response.text
    start_index = text.find('[[')
    end_index = text.rfind(']]') + 2
    json_text = text[start_index:end_index]
    data = json.loads(json_text)
    col1 =['Date','date','1 ','2','6','Opponent','muid','7','Minutes','ORTG','USAGE','EFG','TS%','OR%','DR%','Assist%','TO%','Dunk mades','Dunk Att','Rim mades','Rim Att','Mid made','Mid Att','2 pt made','2 pt Att','3 Pt made','3 Pt Att','Ft Made','FT Att','BPM','OPM','DPM','NET ','Points','OR','DR','Assists','TO','Steals','Blocks','STL%','BLK%','PF','43','BPM round','NET','46','Team','Player','49','Year','PlayerNumber','Year']

    df = pd.DataFrame(data,columns = col1)
    df['Rebounds'] = df['OR'] +df['DR']
    df['PTS+REB+AST'] = df['Rebounds']  + df['Points']  + df['Assists'] 
    df['PtsAvg'] = df['Points'].mean()

    return(df)

def showIndividualPlayerCharts(df,player):
    df1 = df[df['Player']==player]
    density1 = ggplot(df1) + geom_density(aes("Points"), color="blue", fill="blue", alpha=0.1, size=1)+ ggsize(1000, 1000)
    density2 = ggplot(df1) + geom_density(aes("Rebounds"), color="blue", fill="blue", alpha=0.1, size=1)
    density3 = ggplot(df1) + geom_density(aes("Assists"), color="blue", fill="blue", alpha=0.1, size=1)
    density4 = ggplot(df1) + geom_density(aes("PTS+REB+AST"), color="blue", fill="blue", alpha=0.1, size=1)
    p1 = ggplot(df1, aes("Date", "Points")) +geom_path(size=1) +geom_point(size=5)+geom_hline(yintercept=df1['Points'].median(), size=1, color="blue", linetype='longdash')
    p2 = ggplot(df1, aes("Date", "Rebounds")) +geom_path(size=1) +geom_point(size=5)+geom_hline(yintercept=df1['Rebounds'].median(), size=1, color="blue", linetype='longdash')
    p3 = ggplot(df1, aes("Date", "Assists")) +geom_path(size=1) +geom_point(size=5)+geom_hline(yintercept=df1['Assists'].median(), size=1, color="blue", linetype='longdash')
    p4 = ggplot(df1, aes("Date", "PTS+REB+AST")) +geom_path(size=1) +geom_point(size=5)+geom_hline(yintercept=df1['PTS+REB+AST'].median(), size=1, color="blue", linetype='longdash')



    p2 =  gggrid([density1,p1,density2,p2,density3,p3,density4,p4], ncol=2) + ggsize(1000, 1000)
    plot_dict = p2.as_dict()
    st.subheader(player + ' Distribution Charts')
    components.html(_as_html(plot_dict), height=800 + 20,width=800 + 20,scrolling=True,)
    
def showPlayerStatTables(df,player):
    df1 = df[df['Player']==player]
    cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
)
    colors = [(0.6, 0.76, 0.98), (0, 0.21, 0.46)] # Experiment with this
    cm1 = LinearSegmentedColormap.from_list('test', colors, N=256)
    df2 = df1[['Date','Opponent','BPM','OPM','DPM','NET','Minutes','ORTG','Points','Rebounds','Assists','PTS+REB+AST','TO','Steals','Blocks']]
    df2 = df2.set_index('Date')
    team_rating_cols = ['ORTG','BPM','OPM','DPM',]
    depth_rating_cols = ['Minutes','USAGE']
    shooting_cols = ['Points','OR','DR','PTS+REB+AST','Assists','TO','Steals','Blocks']
    col_defs = (
    [
        
        ColumnDefinition(
            name="Date",
            textprops={"ha": "left", "weight": "bold"},
            width=1,
        ),

        ColumnDefinition(
            name="Opponent",
            textprops={"ha": "center"},
            width=3,
        ),
        ColumnDefinition(
            name="ORTG",
            group="Advanced Stats",
            textprops={"ha": "center"},
            cmap=normed_cmap(df["ORTG"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            width=1,
        ),
        ColumnDefinition(
            name="Minutes",
            group="Advanced Stats",
            textprops={"ha": "center"},
            formatter= "{:.0f}",
            cmap=normed_cmap(df["Minutes"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            width=1,
        ),
        ColumnDefinition(
            name="BPM",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["BPM"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Advanced Stats",
        ),
        ColumnDefinition(
            name="OPM",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["OPM"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Advanced Stats",
        ),
        ColumnDefinition(
            name="DPM",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["DPM"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Advanced Stats",
        ),
        ColumnDefinition(
            name="NET",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["NET"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            #group="Advanced Stats",
        ),
        ColumnDefinition(
            name="Points",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["Points"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Game Stats",
        ),
        
        ColumnDefinition(
            name="Rebounds",
            width=1.1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["Rebounds"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Game Stats",
        ),
        ColumnDefinition(
            name="Assists",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["Assists"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Game Stats",
        ),
        ColumnDefinition(
            name="PTS+REB+AST",
            width=2,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["PTS+REB+AST"], cmap=matplotlib.cm.PiYG, num_stds=5),
            group="Game Stats",
        ),
        ColumnDefinition(
            name="TO",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["TO"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Game Stats",
        ),
        ColumnDefinition(
            name="Steals",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["Steals"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Game Stats",
        ),
        ColumnDefinition(
            name="Blocks",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.0f}",
            cmap=normed_cmap(df["Blocks"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Game Stats",
        ),
    ]
    
)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"
    fig, ax = plt.subplots(figsize=(18, 8))

    table = Table(
    df2,
    column_definitions=col_defs,
    row_dividers=True,
    footer_divider=True,
    ax=ax,
    textprops={"fontsize": 14},
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": 1, "linestyle": "-"},
    ).autoset_fontcolors(colnames=["OPM", "DPM",'NET','Minutes','ORTG','Points','Rebounds','Assists','PTS+REB+AST','TO','Steals','Blocks'])
    st.pyplot(fig)   
def showBracketTable(df):
    
    df = df[['Team','2nd Round','Sweet 16','Elite 8','Final 4','Championship','Win','Odds']]
    df = df.set_index("Team")
    cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256)
    colors = [(0.6, 0.76, 0.98), (0, 0.21, 0.46)] # Experiment with this
    cm1 = LinearSegmentedColormap.from_list('test', colors, N=256)
    tourney_cols = ['2nd Round','Sweet 16','Elite 8','Final 4','Championship']
    depth_rating_cols = ['Win','Odds']
#shooting_cols = ['Points','EFG','3PT%','FT%',]
#['eam','Region','Rank','2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win','Odds']
    col_defs = (
    [
        
        ColumnDefinition(
            name="Team",
            textprops={"ha": "left"},
            width=1,
        ),
        
        ColumnDefinition(
            name="2nd Round",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.2f}",
            cmap=normed_cmap(df["2nd Round"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="By Round",
        ),
        ColumnDefinition(
            name="Sweet 16",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.2f}",
            cmap=normed_cmap(df["Sweet 16"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="By Round",
        ),
        ColumnDefinition(
            name="Elite 8",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.2f}",
            cmap=normed_cmap(df["Elite 8"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="By Round",
        ),
        ColumnDefinition(
            name="Final 4",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.2f}",
            cmap=normed_cmap(df["Final 4"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="By Round",
        ),
         ColumnDefinition(
            name="Championship",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.2f}",
            cmap=normed_cmap(df["Championship"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="By Round",
        ),
        ColumnDefinition(
            name="Win",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.2f}",
            cmap=normed_cmap(df["Win"], cmap=matplotlib.cm.PiYG, num_stds=5),
            group="Winning Odds",
        ),
        ColumnDefinition(
            name="Odds",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["Odds"].head(30), cmap=matplotlib.cm.coolwarm_r, num_stds=2),
            group="Winning Odds",
        ),
    ]
)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"
    fig, ax = plt.subplots(figsize=(30,30))
 
    table = Table(
    df,
    column_definitions=col_defs,
    row_dividers=True,
    footer_divider=True,
    ax=ax,
    textprops={"fontsize": 16},
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": 1, "linestyle": "-"},
    ).autoset_fontcolors(colnames=['2nd Round','Sweet 16','Elite 8','Final 4','Championship','Win','Odds'])
    st.pyplot(fig)
def playgame(team1, team2, T):
    """There's a difference between flipping a game in an existing
    bracket, and playing a game from scratch. If we're going to just
    use Boltzmann statistics to play a game from scratch, we can make
    life easy by using the Boltzmann factor to directly pick a
    winner.
    """
    ediff = deltaU(team1, team2)
    boltzmann_factor = exp(-ediff/T)

    win_prob = boltzmann_factor/(1+boltzmann_factor) if boltzmann_factor < inf else 1
    # So, prob of team 1 winning is then boltzmann_factor/(1+boltzmann_factor)
    if random() >= win_prob:
        return (team1,team2)
    else:
        return (team2,team1)

def playgamesfortesting(team1, team2, ntrials, T):
    print("Boltzmann tells that the ratio of team1 winning to team 2"+ 
          "winning should be")
    print(exp(-deltaU(team1,team2)/T))
    wins = {team1:0,team2:0}
    for i in range(ntrials):
        winner,loser = playgame(team1,team2,T)
        wins[winner] = wins[winner] + 1
    print("wins {} {} {} {} {}".format(wins, wins[team1]/wins[team2], 
                                       wins[team2]/wins[team1], 
                                       wins[team1]/ntrials, 
                                       wins[team2]/ntrials))

    
# changed to playgameCDF from boltzman. boltzman favors underdogs too much
def playround(teams, T):
    winners = []
    losers = []
    for (team1, team2) in pairs(teams):
        #winner, loser = playgameCDF(team1,team2,T)
        winner, loser = playgameCDF2023(team1,team2,T)
        winners.append(winner)
        losers.append(loser)
    return winners,losers

def bracket_energy(all_winners):
    total_energy = 0.0
    for i in range(len(all_winners)-1):
        games = pairs(all_winners[i])
        winners = all_winners[i+1]
        for (team1, team2),winner in zip(games, winners):
            if winner == team1:
                total_energy += default_energy_function(team1, team2)
            else:
                total_energy += default_energy_function(team2, team1)
    return total_energy
def getroundmap(bracket, include_game_number):
    games_in_rounds = [2**i for i in reversed(range(len(bracket)-1))]
    round = {}
    g = 0
    for (i,gir) in enumerate(games_in_rounds):
        for j in range(gir):
            if include_game_number:
                round[g] = (i,j)
            else:
                round[g] = i
            g += 1
    return round
def energy_of_flipping(current_winner, current_loser):
    """Given the current winner and the current loser, this calculates
    the energy of swapping, i.e. having the current winner lose.
    """
    return (default_energy_function(current_loser, current_winner) - 
            default_energy_function(current_winner, current_loser))


def NewgetGamePredictionNeutralCourt(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE):
  
    GameTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo

    Team1Score=(Team1AdjOff/LeagueOE*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*Team1AdjDef)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return PointDiff
# Here are the "magic functions" I mentioned to get pairs of teams.

#from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def pairs(iterable):
    return grouper(2,iterable)

def runbracket1(teamsdict,ntrials, T):
    results = {'all':simulate(teamsdict,ntrials,'all',T)}
    return results

def simulate(teamsdict,ntrials, region, T, printonswap=False, printbrackets=True):
    """
    If region is "west" "midwest" "south" or "east" we'll run a bracket based 
    just on those teams.
    If it's "all" we'll run a full bracket.
    If it's a list of teams, we'll run a bracket based just on that list.
    So, one way you might want to do things is to simulate 10000 runs for each 
    of the four brackets,
    then run your final four explicitly, e.g.
    T = 1.5
    simulate(10000,'midwest',T)
    # record results
    simulate(10000,'south',T)
    # record results
    simulate(10000,'west',T)
    # record results
    simulate(10000,'east',T)
    # record results
    simulate(10000,['Louisville','Kansas','Wisconsin','Indiana'],T)
    """

    if type(region)  in (type([]), type(())):
        teams = region[:]
    else:
        teams = teamsdict[region]
    print(teams)
    b = Bracket(teams,T)
    energy = b.energy()
    ng = sum(b.games_in_rounds) # total number of games
    # Let's collect some statistics
    brackets = []
    for trial in range(ntrials):
        g = randint(0, ng) # choose a random game to swap
        #print "attempted swap for game",g#,"in round",round[g]
        #newbracket = deepcopy(b)
        newbracket = b.copy()
        newbracket.swap(g)
        newenergy = newbracket.energy()
        ediff = newenergy - energy
        if ediff <= 0:
            b = newbracket
            energy = newenergy
            if printonswap:
                print("LOWER")
                print(b)
        else:
            if random() < exp(-ediff/T):
                b = newbracket
                energy = newenergy
                if printonswap:
                    print( "HIGHER")
                    print(b)
        brackets.append(b)


    lb, mcb, mcb_count, unique_brackets, lowest_sightings = \
        Stats.gather_uniquestats(brackets)
    sr = SimulationResults(brackets,unique_brackets,lb,lowest_sightings,mcb,mcb_count)
    #if printbrackets:
        #print("Lowest energy bracket")
        #print(lb)
        #print("Most common bracket (%s)"%mcb_count)
        #print(mcb)
    return sr


#Rankings,teams, strength,T
class Bracket(object):
    def __init__(self, teams,T,bracket=None):
        """
        
        Arguments:
        - `teams`:
        - `T`:
        """
        self.teams = teams
        self.T = T
        #self.Rankings = Rankings
        #self.strength = strength
        if bracket is None:
            self.bracket = runbracket(self.teams, self.T)
        else:
            self.bracket = bracket
        self.games_in_rounds = [2**i for i in 
                                reversed(range(len(self.bracket)-1))]
        self.roundmap = getroundmap(self.bracket, include_game_number=False)
        self.roundmap_with_game_numbers = getroundmap(self.bracket, 
                                                      include_game_number=True)
    def copy(self):
        return self.__class__(self.teams, self.T,  
                              bracket=[l[:] for l in self.bracket])
    def energy(self):
        return bracket_energy(self.bracket)
    def __str__(self):
        return bracket_to_string(self.bracket)
    __repr__ = __str__
    def __hash__(self):
        return hash(tuple([tuple(aw) for aw in self.bracket]))
    def game(self, g):
        """Return (team1,team2,winner).
        """
        t1, t2, win = self._getgameidxs(g)
        return (self.bracket[t1[0]][t1[1]], self.bracket[t2[0]][t2[1]],
                self.bracket[win[0]][win[1]])

    def _round_teaminround_to_game(self,r,gir):
        return sum(self.games_in_rounds[:r]) + int(gir/2)

    def _getgameidxs(self, g):
        # we'll return (round,game) for each of team1, team2, winner
        # 0 1 2 3 4 5 6 7 # teams 1
        # 0 0 1 1 2 2 3 3 # games 1
        # 0 2 4 6 # teams 2
        # 0 0 1 1 # games 2
        round, game_in_round = self.roundmap_with_game_numbers[g]
        return ((round,2*game_in_round), (round,2*game_in_round+1), 
                (round+1,game_in_round))
    def _setwinner(self, g, winner):
        """ JUST SETS THE WINNER, DOES NOT LOOK TO NEXT ROUND! USE SWAP FOR 
        THAT! 
        """
        t1,t2,win = self._getgameidxs(g)
        self.bracket[win[0]][win[1]] = winner
    def swap(self, g):
        """
        NOTE: This does not check 
        """
        team1, team2, winner = self.game(g)
        if team1 == winner:
            self._setwinner(g, team2)
        else:
            self._setwinner(g, team1)
        wr, wt = self._getgameidxs(g)[2]
        ng = self._round_teaminround_to_game(wr, wt)
        while ng < sum(self.games_in_rounds):
            #print "Now need to check game",wr,wt,ng,self.game(ng)
            winner, loser = playgame(self.game(ng)[0], self.game(ng)[1], self.T)
            self._setwinner(ng, winner)
            wr, wt = self._getgameidxs(ng)[2]
            ng = self._round_teaminround_to_game(wr, wt)
    def upsets(self):
        result = 0
        for g in range(sum(self.games_in_rounds)):
            t1,t2,win = self.game(g)
            if t1 == win:
                los = t2
            else:
                los = t1
            if energy_of_flipping(win,los) < 0:
                result += 1
        return result
    
    
def runbracket(teams, T):
    # How many rounds do we need?
    nrounds = int(np.log2(len(teams)))
    winners = teams #they won to get here!
    all_winners = [winners]
    for round in range(nrounds):
        winners, losers = playround(winners, T)
        all_winners.append(winners)
    return all_winners
def playround(teams, T):
    winners = []
    losers = []
    for (team1, team2) in pairs(teams):
        #winner, loser = playgameCDF(team1,team2,T)
        winner, loser = playgameCDF2024(team1,team2,T)
        winners.append(winner)
        losers.append(loser)
    return winners,losers
def playgameCDF2024(team1, team2, T):
    """There's a difference between flipping a game in an existing
    bracket, and playing a game from scratch. If we're going to just
    use Boltzmann statistics to play a game from scratch, we can make
    life easy by using the Boltzmann factor to directly pick a
    winner.
    """
    #print(team1,team2)
    #ediff = deltaU(team1, team2)
    #boltzmann_factor = exp(-ediff/T)
    #PHomeTeamSpread=NewgetGamePredictionNeutralCourt(PomeroyDict[team1]["AdjO"],PomeroyDict[team1]["AdjD"],PomeroyDict[team1]["AdjT"],PomeroyDict[team2]["AdjO"],PomeroyDict[team2]["AdjD"],PomeroyDict[team2]["AdjT"],LeagueTempo,LeagueOE)
    #if MODEL == 'TRank':
    #    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(MYRANKS[team1]["AdjOE"],MYRANKS[team1]["AdjDE"],MYRANKS[team1]["pace"],MYRANKS[team2]["AdjOE"],MYRANKS[team2]["AdjDE"],MYRANKS[team2]["pace"],LeagueTempo,LeagueOE)
    #else:
    #    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(MYRANKS[team1]["AdjOE"],MYRANKS[team1]["AdjDE"],MYRANKS[team1]["pace"],MYRANKS[team2]["AdjOE"],MYRANKS[team2]["AdjDE"],MYRANKS[team2]["pace"],LeagueTempo,LeagueOE)
    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(MYRANKS[team1]["AdjOE"],MYRANKS[team1]["AdjDE"],MYRANKS[team1]["pace"],MYRANKS[team2]["AdjOE"],MYRANKS[team2]["AdjDE"],MYRANKS[team2]["pace"],LeagueTempo,LeagueOE)    
    win_prob =scipy.stats.norm(0,10.5).cdf(PHomeTeamSpread)
    #win_prob = boltzmann_factor/(1+boltzmann_factor) if boltzmann_factor < inf else 1
    # So, prob of team 1 winning is then boltzmann_factor/(1+boltzmann_factor)
    if random() >= win_prob:
        return (team1,team2)
    else:
        return (team2,team1)
def bracket_to_string(all_winners):
    """ Cute version that prints out brackets for 2, 4, 8, 16, 32, 64, etc. """
    result = ''
    aw = all_winners # save some typing later
    nrounds = len(all_winners) #int(np.log2(len(teams)))
    # We'll keep the results in a big array it turns out that arrays
    # of strings have to know the max string size, otherwise things
    # will just get truncated.
    maxlen = max([len(s) for s in all_winners[0]])
    dt = np.dtype([('name', np.str_, maxlen)])
    results = array([['' for i in range(len(all_winners[0]))] for j in 
                     range(nrounds)], dtype=dt['name'])
    # First round, all of the spots are filled
    results[0] = all_winners[0]
    # all other rounds, we split the row in half and fill from the middle out.
    for i in range(1, nrounds): # we've done the 1st and last already
        # round 1 skips two, round 2 skips 4, etc.
        these_winners = all_winners[i]
        # Fill top half
        idx = int(len(all_winners[0])/2 - 1)
        for team in reversed(all_winners[i][:int(len(all_winners[i])/2)]):
            results[i][idx] = team
            idx -= 2**i
        # Fill bottom half
        idx = int(len(all_winners[0])/2)
        for team in all_winners[i][int(len(all_winners[i])/2):]:
            results[i][idx] = team
            idx += 2**i

    def tr(i,include_rank=False,maxlen=None):
        """ Print out the team and ranking """
        if maxlen is not None:
            team = i[:maxlen]
        else:
            team = i
        if include_rank:
            try:
                region = regions[i]
                result = '%s (%s)'%(team,int(regional_rankings[i]))
            except KeyError:
                result = '%s'%(team)
        return result
    stub = '%-25s ' + ' '.join(['%-8s']*(nrounds-1))
    for i in range(len(all_winners[0])):
        these = results[:,i]
        these = [tr(these[0], include_rank=True)] + \
            [tr(i, maxlen=3, include_rank=True) for i in these[1:]]
        result += stub % tuple(these)
        result += '\n'
    result += "Total bracket energy: %s"%bracket_energy(all_winners)
    result += '\n'
    return result

def print_bracket(bracket):
    print( bracket_to_string(bracket))


def makehtmltable(tabledata,headers):
    result = '<table>\n'
    result += '<tr>'
    for header in headers:
        result += '<th>{h}</th>'.format(h=header)
    result += '</tr>\n'
    for row in tabledata:
        result += '<tr>'
        for col in row:
            result += '<td>{c}</td>'.format(c=col)
        result += '</tr>\n'
    result += '</table>'
    return result

class Stats:
    @staticmethod
    def gather_uniquestats(brackets):
        lb = Bracket(brackets[0].teams, T=0.0000001) # low bracket
        low_hash = hash(lb)
        cnt = Counter()
        unique_brackets = []
        lowest_sightings = []
        brackets_by_hash = {}
        for b in brackets:
            h = hash(b)
            cnt[h] += 1
            unique_brackets.append(len(cnt))
            lowest_sightings.append(cnt[low_hash])
            brackets_by_hash[h] = b
        h,c = cnt.most_common(1)[0]
        mcb = brackets_by_hash[h] # most comon bracket
        return lb, mcb, c, unique_brackets, lowest_sightings

    @staticmethod
    def maketable(results):
        counts = defaultdict(Counter)
        allrounds = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
        rounds1 = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']
        rounds2 = ['Championship','Win']
        if 'all' not in results:
            for region in 'south midwest east west'.split():
                for bracket in results[region].brackets:
                    for (name,num) in zip(rounds1,[0,1,2,3,4]):
                        for team in bracket.bracket[num]:
                            counts[team][name] += 1
            for bracket in results['final four'].brackets:
                for (name,num) in zip(rounds2,[1,2]):
                    for team in bracket.bracket[num]:
                        counts[team][name] += 1
            nt1 = len(results['south'].brackets)
            nt2 = len(results['final four'].brackets)
        else:
            for bracket in results['all'].brackets:
                for (name,num) in zip(allrounds,[0,1,2,3,4,5,6]):
                    for team in bracket.bracket[num]:
                        counts[team][name] += 1
            nt1 = nt2 = len(results['all'].brackets)
        # Now turn that into percentages
        pct = {}
        for team in counts:
            pct[team] = {}
            for r in ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']:
                pct[team][r] = 100*counts[team][r]/nt1
            for r in ['Championship','Win']:
                pct[team][r] = 100*counts[team][r]/nt2
            pct[team]['Rank'] = int(regional_rankings[team])
        def tablekey(d):
            # gets teamname, pct
            return [d[1][i] for i in reversed(allrounds)]
        items = reversed(sorted(pct.items(),key=tablekey))
        # make a table
        headers = ['Team'] + ['Region','Rank'] + allrounds
        tabledata = []
        for (team,pcts) in items:
            tabledata.append([team] + [pcts[i] for i in ['Region','Rank'] + allrounds])
        #return tabulate(tabledata, headers=headers, tablefmt="html" )
        return makehtmltable(tabledata, headers=headers)


def tablekeytest(allrounds):
        # gets teamname, pct
    return [d[1][i] for i in reversed(allrounds)]


def maketabletest(results):
    counts = defaultdict(Counter)
    allrounds = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
    rounds1 = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']
    rounds2 = ['Championship','Win']
    if 'all' not in results:
        for region in 'south midwest east west'.split():
            for bracket in results[region].brackets:
                for (name,num) in zip(rounds1,[0,1,2,3,4]):
                    for team in bracket.bracket[num]:
                        counts[team][name] += 1
        for bracket in results['final four'].brackets:
            for (name,num) in zip(rounds2,[1,2]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = len(results['south'].brackets)
        nt2 = len(results['final four'].brackets)
    else:
        for bracket in results['all'].brackets:
            for (name,num) in zip(allrounds,[0,1,2,3,4,5,6]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = nt2 = len(results['all'].brackets)
        # Now turn that into percentages
    pct = {}
    for team in counts:
        pct[team] = {}
        for r in ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']:
            pct[team][r] = 100*counts[team][r]/nt1
        for r in ['Championship','Win']:
            pct[team][r] = 100*counts[team][r]/nt2
        pct[team]['Region'] = regions[team]
        pct[team]['Rank'] = int(regional_rankings[team])
    def tablekey(d):
            # gets teamname, pct
        return [d[1][i] for i in reversed(allrounds)]
    items = reversed(sorted(pct.items(),key=tablekey))
    # make a table
    headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    tabledata = []
    for (team,pcts) in items:
        tabledata.append([team] + [pcts[i] for i in ['Region','Rank'] + allrounds])
    for i in range(len(tabledata)):
        if tabledata[i][9] == 0:
            tabledata[i].append(0)
        else:
            tabledata[i].append((100-tabledata[i][9])/tabledata[i][9])
   
        #return tabulate(tabledata, headers=headers, tablefmt="html" )
    return (tabledata)


def maketabletestSweet16(results):
    counts = defaultdict(Counter)
    allrounds = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
    allrounds = ['Sweet 16','Elite 8','Final 4','Championship','Win']
    
    rounds1 = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']
    rounds2 = ['Championship','Win']
    if 'all' not in results:
        for region in 'south midwest east west'.split():
            for bracket in results[region].brackets:
                for (name,num) in zip(rounds1,[0,1,2,3,4]):
                    for team in bracket.bracket[num]:
                        counts[team][name] += 1
        for bracket in results['final four'].brackets:
            for (name,num) in zip(rounds2,[1,2]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = len(results['south'].brackets)
        nt2 = len(results['final four'].brackets)
    else:
        for bracket in results['all'].brackets:
            for (name,num) in zip(allrounds,[0,1,2,3,4]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = nt2 = len(results['all'].brackets)
        # Now turn that into percentages
    pct = {}
    for team in counts:
        pct[team] = {}
        for r in ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']:
            pct[team][r] = 100*counts[team][r]/nt1
        for r in ['Championship','Win']:
            pct[team][r] = 100*counts[team][r]/nt2
        pct[team]['Region'] = regions[team]
        pct[team]['Rank'] = int(regional_rankings[team])
    def tablekey(d):
            # gets teamname, pct
        return [d[1][i] for i in reversed(allrounds)]
    items = reversed(sorted(pct.items(),key=tablekey))
    # make a table
    headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    tabledata = []
    for (team,pcts) in items:
        tabledata.append([team] + [pcts[i] for i in ['Region','Rank'] + allrounds])
    for i in range(len(tabledata)):
        if tabledata[i][7] == 0:
            tabledata[i].append(0)
        else:
            tabledata[i].append((100-tabledata[i][7])/tabledata[i][7])
   
        #return tabulate(tabledata, headers=headers, tablefmt="html" )
    return (tabledata)

def maketabletestElite8(results):
    counts = defaultdict(Counter)
    allrounds = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
    allrounds = ['Elite 8','Final 4','Championship','Win']
    
    rounds1 = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']
    rounds2 = ['Championship','Win']
    if 'all' not in results:
        for region in 'south midwest east west'.split():
            for bracket in results[region].brackets:
                for (name,num) in zip(rounds1,[0,1,2,3,4]):
                    for team in bracket.bracket[num]:
                        counts[team][name] += 1
        for bracket in results['final four'].brackets:
            for (name,num) in zip(rounds2,[1,2]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = len(results['south'].brackets)
        nt2 = len(results['final four'].brackets)
    else:
        for bracket in results['all'].brackets:
            for (name,num) in zip(allrounds,[0,1,2,3]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = nt2 = len(results['all'].brackets)
        # Now turn that into percentages
    pct = {}
    for team in counts:
        pct[team] = {}
        for r in ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']:
            pct[team][r] = 100*counts[team][r]/nt1
        for r in ['Championship','Win']:
            pct[team][r] = 100*counts[team][r]/nt2
        pct[team]['Region'] = regions[team]
        pct[team]['Rank'] = int(regional_rankings[team])
    def tablekey(d):
            # gets teamname, pct
        return [d[1][i] for i in reversed(allrounds)]
    items = reversed(sorted(pct.items(),key=tablekey))
    # make a table
    headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    tabledata = []
    for (team,pcts) in items:
        tabledata.append([team] + [pcts[i] for i in ['Region','Rank'] + allrounds])
    for i in range(len(tabledata)):
        if tabledata[i][6] == 0:
            tabledata[i].append(0)
        else:
            tabledata[i].append((100-tabledata[i][6])/tabledata[i][6])
   
        #return tabulate(tabledata, headers=headers, tablefmt="html" )
    return (tabledata)

def maketabletestFinal4(results):
    counts = defaultdict(Counter)
    allrounds = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
    allrounds = ['Final 4','Championship','Win']
    
    rounds1 = ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']
    rounds2 = ['Championship','Win']
    if 'all' not in results:
        for region in 'south midwest east west'.split():
            for bracket in results[region].brackets:
                for (name,num) in zip(rounds1,[0,1,2,3,4]):
                    for team in bracket.bracket[num]:
                        counts[team][name] += 1
        for bracket in results['final four'].brackets:
            for (name,num) in zip(rounds2,[1,2]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = len(results['south'].brackets)
        nt2 = len(results['final four'].brackets)
    else:
        for bracket in results['all'].brackets:
            for (name,num) in zip(allrounds,[0,1,2,3]):
                for team in bracket.bracket[num]:
                    counts[team][name] += 1
        nt1 = nt2 = len(results['all'].brackets)
        # Now turn that into percentages
    pct = {}
    for team in counts:
        pct[team] = {}
        for r in ['2nd Round','3rd Round','Sweet 16','Elite 8','Final 4']:
            pct[team][r] = 100*counts[team][r]/nt1
        for r in ['Championship','Win']:
            pct[team][r] = 100*counts[team][r]/nt2
        pct[team]['Region'] = regions[team]
        pct[team]['Rank'] = int(regional_rankings[team])
    def tablekey(d):
            # gets teamname, pct
        return [d[1][i] for i in reversed(allrounds)]
    items = reversed(sorted(pct.items(),key=tablekey))
    # make a table
    headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    tabledata = []
    for (team,pcts) in items:
        tabledata.append([team] + [pcts[i] for i in ['Region','Rank'] + allrounds])
    for i in range(len(tabledata)):
        if tabledata[i][5] == 0:
            tabledata[i].append(0)
        else:
            tabledata[i].append((100-tabledata[i][5])/tabledata[i][5])
   
        #return tabulate(tabledata, headers=headers, tablefmt="html" )
    return (tabledata)


def alex_energy_game(winner, loser):
    """def energy(A,B):
     get AdjEM ("adjusted efficiency margin," not the other AdjEMs) for A and B from kenpom.com
     energy = ln(AdjEM[A]/AdjEM[B])
    """
    result = kpomdata[winner]["AdjEM"]/kpomdata[loser]["AdjEM"]
    #result = -(strength[winner] - strength[loser])
    #np.log(adjem[loser]/adjem[winner])
    return -result


def getSweetSixteensOddsTable(BracketName,EnergyNumber):
    r=simulate(10000,BracketName,EnergyNumber)
    lb, mcb, mcb_count, unique_brackets, lowest_sightings = Stats.gather_uniquestats(r.brackets)
    sr = SimulationResults(r.brackets,unique_brackets,lb,lowest_sightings,mcb,mcb_count)
    trueresults = {'all':sr}
    j=maketabletestSweet16(trueresults)
    #allrounds = ['Sweet 16','Elite 8','Final 4','Championship','Win']
    #headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    #l=HTML(makehtmltable(j, headers=headers))
    return(j)


def getEliteEightOddsTable(BracketName,EnergyNumber):
    r=simulate(10000,BracketName,EnergyNumber)
    lb, mcb, mcb_count, unique_brackets, lowest_sightings = Stats.gather_uniquestats(r.brackets)
    sr = SimulationResults(r.brackets,unique_brackets,lb,lowest_sightings,mcb,mcb_count)
    trueresults = {'all':sr}
    j=maketabletestElite8(trueresults)
    #allrounds = ['Sweet 16','Elite 8','Final 4','Championship','Win']
    #headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    #l=HTML(makehtmltable(j, headers=headers))
    return(j)
def getFinalFourOddsTable(BracketName,EnergyNumber):
    r=simulate(10000,BracketName,EnergyNumber)
    lb, mcb, mcb_count, unique_brackets, lowest_sightings = Stats.gather_uniquestats(r.brackets)
    sr = SimulationResults(r.brackets,unique_brackets,lb,lowest_sightings,mcb,mcb_count)
    trueresults = {'all':sr}
    j=maketabletestFinal4(trueresults)
    #allrounds = ['Sweet 16','Elite 8','Final 4','Championship','Win']
    #headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']
    #l=HTML(makehtmltable(j, headers=headers))
    return(j)
def showPlayersTable(player_data,team_selected):
    cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256)
    colors = [(0.6, 0.76, 0.98), (0, 0.21, 0.46)] # Experiment with this
    cm1 = LinearSegmentedColormap.from_list('test', colors, N=256)
    df = player_data[['Player','Number','Year','Height','Position','Games','ORTG','BPM','OBPM','DBPM', 'PRPG','Points','EFG','3PT%','FT%','Min%','USAGE','Team']]

    #df = player_data1[player_data1['Team']=='Purdue']   
    df =df.sort_values('PRPG',ascending=False)
    df = df[df['ORTG'] != 0]
    df['EFG'] =df['EFG']/100
    df['Min%'] =df['Min%']/100
    df['USAGE'] =df['USAGE']/100
    df = df[df['Games']>2]
    df1 = df[df['Team']==team_selected]
    df1 = df1.set_index('Player')
    df1 = df1[['Number','Year','Height','Position','Games','ORTG','BPM','OBPM','DBPM', 'PRPG','Points','EFG','3PT%','FT%','Min%','USAGE']]
    team_rating_cols = ['ORTG','BPM','OBPM','DBPM','PRPG']
    depth_rating_cols = ['Min%','USAGE']
    shooting_cols = ['Points','EFG','3PT%','FT%',]
    col_defs = (
    [
        
        ColumnDefinition(
            name="Player",
            textprops={"ha": "left", "weight": "bold"},
            width=3,
        ),

        ColumnDefinition(
            name="Number",
            group="Basic Info",
            textprops={"ha": "center"},
            width=1,
        ),
        ColumnDefinition(
            name="Position",
            group="Basic Info",
            textprops={"ha": "center"},
            width=1.2,
        ),
        ColumnDefinition(
            name="Height",
            group="Basic Info",
            textprops={"ha": "center"},
            width=.9,
        ),
         ColumnDefinition(
            name="Games",
            group="Basic Info",
            textprops={"ha": "center"},
            width=.9,
        ),
         ColumnDefinition(
            name="Year",
            group="Basic Info",
            textprops={"ha": "center"},
            width=.9,
        ),
        ColumnDefinition(
            name="ORTG",
            group="Advanced Stats",
            textprops={"ha": "center"},
            cmap=normed_cmap(df["ORTG"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            width=1,
        ),
        ColumnDefinition(
            name="BPM",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["BPM"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Advanced Stats",
        ),
        ColumnDefinition(
            name="OBPM",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["OBPM"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Advanced Stats",
        ),
        ColumnDefinition(
            name="DBPM",
            width=1,
            textprops={
                "ha": "center",
               # "bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["DBPM"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Advanced Stats",
        ),
        ColumnDefinition(
            name="PRPG",
            width=1,
            textprops={
                "ha": "center",
                #"bbox": {"boxstyle": "circle", "pad": 0.35},
            },
            formatter= "{:.1f}",
            cmap=normed_cmap(df["PRPG"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            #group="Advanced Stats",
        ),
    ]
    + [
        ColumnDefinition(
            name=depth_rating_cols[0],
            title=depth_rating_cols[0].replace(" ", "\n", 1),
            #formatter=decimal_to_percent,
           formatter= "{:.2%}",
            cmap=cm1,
            width=1.2,
            group="Depth ",
            #border="center",
        )
    ]
    + [
        ColumnDefinition(
            name=col,
            title=col.replace(" ", "\n", 1),
            formatter= "{:.2%}",
            cmap=cm1,
            width=1.2,
            #formatter=decimal_to_percent,
            group="Depth ",
        )
        for col in depth_rating_cols[1:]
    ]
    + [
        ColumnDefinition(
            name=shooting_cols[0],
            title=shooting_cols[0].replace(" ", "\n", 1),
            formatter= " {:.2f}",
            width=1.2,
            #formatter=decimal_to_percent,
            cmap=normed_cmap(df["Points"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            group="Shooting Stats",
           # border="center",
        )
    ]
    + [
        ColumnDefinition(
            name=col,
            title=col.replace(" ", "\n", 1),
            formatter= " {:.2%}",
            width=1.2,
            #formatter=decimal_to_percent,
            cmap=cmap,
            group="Shooting Stats",
        )
        for col in shooting_cols[1:]
    ]
)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"
    fig, ax = plt.subplots(figsize=(20, 8))
 
    table = Table(
    df1,
    column_definitions=col_defs,
    row_dividers=True,
    footer_divider=True,
    ax=ax,
    textprops={"fontsize": 14},
    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
    column_border_kw={"linewidth": 1, "linestyle": "-"},
    ).autoset_fontcolors(colnames=["OBPM", "DBPM"])
    st.pyplot(fig)
    
def GetBracketMatrix():
    BracketLookup="http://bracketmatrix.com"
    res = requests.get(BracketLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))[0]
    df1=df.iloc[3:, 0:4]
    return(df1)
def getMGWinRecord(s):
    if (s['MG_SpreadWinATS'] == 1):
        return 3
    else:
        return -3
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



agContextMenuItemsDeluxe = JsCode(
    """
    function getContextMenuItems(params) {
      const result = [
        'copy',
        'copyWithHeaders',
        'paste',
        'separator',
        'autoSizeAll',
        'expandAll',
        'contractAll',
        'resetColumns',
        'separator',
        'export',
      ];
      
      return result;
    }
    """
)

agContextMenuItemsBasic = JsCode(
    """
    function getContextMenuItems(params) {
      const result = [
        'copy',
        'copyWithHeaders',
        'paste',
        'separator',
        'autoSizeAll',
        'resetColumns',
        'separator',
        'export',
      ];
      
      return result;
    }
    """
)


_type_mapper = {
    "b": ["textColumn"],
    "i": ["numericColumn", "numberColumnFilter"],
    "u": ["numericColumn", "numberColumnFilter"],
    "f": ["numericColumn", "numberColumnFilter"],
    "c": [],
    "m": ['timedeltaFormat'],
    "M": ["dateColumnFilter", "customDateTimeFormat"],
    "O": [],
    "S": [],
    "U": [],
    "V": [],
}



DEFAULT_COL_PARAMS = dict(
    filterParams=dict(buttons=['apply', 'reset'], closeOnApply=True),
    groupable=False,
    enableValue=True,
    enableRowGroup=True,
    enablePivot=False,
    editable=False
)


DEFAULT_STATUS_BAR = {
    'statusPanels': [dict(statusPanel='agTotalAndFilteredRowCountComponent', align='left')]
}


def gridOptionsFromDataFrame(df: pd.DataFrame, **default_column_parameters) -> GridOptionsBuilder():

    gb = GridOptionsBuilder()

    params = {**DEFAULT_COL_PARAMS, **default_column_parameters}
    gb.configure_default_column(**params)

    if any('.' in col for col in df.columns):
        gb.configure_grid_options(suppressFieldDotNotation=True)

    for col, col_type in zip(df.columns, df.dtypes):
        if col in FIELD_CONFIG:
            conf = FIELD_CONFIG.get(col)
            if 'type' not in conf:
                gb.configure_column(field=col, type=_type_mapper.get(col_type.kind, []), **conf)
            else:
                gb.configure_column(field=col, **conf)
        else:
            gb.configure_column(field=col, type=_type_mapper.get(col_type.kind, []), custom_format_string='yyyy-MM-dd HH:mm')

    return gb





DEFAULT_GRID_OPTIONS = dict(
    domLayout='normal',
    # rowGroupPanelShow='always',
    statusBar=DEFAULT_STATUS_BAR,
    autoGroupColumnDef=dict(pinned='left'),
    getContextMenuItems=agContextMenuItemsBasic,
    pivotPanelShow='onlyWhenPivoting',
    # pivotMode=False,
    # pivotPanelShow='always',
    # pivotColumnGroupTotals='before',
    # rowSelection='multiple',
    enableRangeSelection=True,
    suppressMultiRangeSelection=True,
    # defaultCsvExportParams=dict(fileName='testExport.csv'),
    suppressExcelExport=True,
)

def displayGrid(df: pd.DataFrame, key: str, reloadData: bool, updateMode: GridUpdateMode=GridUpdateMode.VALUE_CHANGED):

    gb = gridOptionsFromDataFrame(df)
    gb.configure_side_bar()
    gb.configure_grid_options(**DEFAULT_GRID_OPTIONS)
    gridOptions = gb.build()

    # updateMode = GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.VALUE_CHANGED
    dataReturnMode = DataReturnMode.FILTERED_AND_SORTED

    g = AgGrid(df, gridOptions=gridOptions, height=700, key=key, editable=True,
               enable_enterprise_modules=True,
               allow_unsafe_jscode=True,
               fit_columns_on_grid_load=False,
               reload_data=reloadData,
               # theme='streamlit',
               update_mode=updateMode,
               data_return_mode=dataReturnMode,
               )

    return g

def numberSort():
    jscript = """
        function(num1, num2) {
          return num1 - num2;
        };
        """

    return JsCode(jscript)

def numberFormat(precision: int, comma: bool = True):
    if comma:
        jscript = """
        function(params) {
          if (params.value === null || isNaN(params.value)) {
            return "";
          }
          return params.value.toFixed(%d).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        };
        """ % precision
    else:
        jscript = """
        function(params) {
          if (params.value === null || isNaN(params.value)) {
            return "";
          }
          return params.value.toFixed(%d);
        };
        """ % precision

    return JsCode(jscript)


def cellStyleFromDict(d):
    dd = json.dumps(d)
    return JsCode("""
    function(params) {
      return %s;
    };
    """ % dd)

def cellStyleBrown(bold: bool = False):
    if not bold:
        j = """
        function(params) {
          return {'color': 'black', 'backgroundColor': '#e4d2ba'};
        };
        """
    else:
        j = """
        function(params) {
          return {'color': 'black', 'backgroundColor': '#e4d2ba', 'fontWeight': 'bold'};
        };
        """
    return JsCode(j)


def cellStyleGrey():
    j = """
    function(params) {
      return {'color': 'black', 'backgroundColor': '#ececec'};
    };
    """
    return JsCode(j)


def cellStyleBig():
    cellsytle_jscode105 = JsCode("""
      function(params) {
      switch (true ) {
        case (params.value < -100000): return {'color': 'white', 'backgroundColor': '#ff0000'};
        case (params.value < -50000 && params.value >= -100000): return {'color': 'white', 'backgroundColor': '#ff4c4c'};
        case (params.value < -5000 && params.value>= -50000): return {'color': 'white', 'backgroundColor': '#ff9999'};
        case (params.value < 0 && params.value >= -5000): return {'color': 'white', 'backgroundColor': '#ffb2b2'};
        case (params.value > 100000): return {'color': 'white', 'backgroundColor': '#2753a1'};
        case (params.value < 100000 && params.value >= 50000): return {'color': 'white', 'backgroundColor': '#3c64aa'};
        case (params.value < 50000 && params.value >= 5000): return {'color': 'white', 'backgroundColor': '#7d97c6'};
        case (params.value < 5000 && params.value > 0): return {'color': 'white', 'backgroundColor': '#a8bad9'};
         }
      };
         """)
    return cellsytle_jscode105


def getBigSeriesFromColumns(df: pd.DataFrame, columns: list = None):
    l = []
    cols = columns if columns is not None else df.columns
    for col in cols:
        if is_numeric_dtype(df[col]):
            l.extend(df[col].values.tolist())

    return pd.Series(l)


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

def _displayGrid(df: pd.DataFrame, gb: GridOptionsBuilder,
                 key: str, reloadData: bool=False, updateMode: GridUpdateMode=GridUpdateMode.VALUE_CHANGED,
                 height=700, fit_columns_on_grid_load=True):

    gridOptions = gb.build()

    # updateMode = GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.VALUE_CHANGED
    dataReturnMode = DataReturnMode.FILTERED_AND_SORTED

    g = AgGrid(df, gridOptions=gridOptions, height=height, key=key, editable=True,
               enable_enterprise_modules=True,
               allow_unsafe_jscode=True,
               fit_columns_on_grid_load=fit_columns_on_grid_load,
               reload_data=reloadData,
               # theme='streamlit',
               update_mode=updateMode,
               data_return_mode=dataReturnMode,
               )

    return g


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
    HomeTeamInfo["EMOver"]=HomeTeamInfo['PlayingOverRating'].rolling(3).mean()
    AwayTeamInfo["EM3"]=AwayTeamInfo['AdjO3ExpMA']-AwayTeamInfo['AdjD3ExpMA']
    AwayTeamInfo["EM5"]=AwayTeamInfo['AdjO5ExpMA']-AwayTeamInfo['AdjD5ExpMA']
    AwayTeamInfo["EM10"]=AwayTeamInfo['AdjO10ExpMA']-AwayTeamInfo['AdjD10ExpMA']
    AwayTeamInfo["EMOver"]=AwayTeamInfo['PlayingOverRating'].rolling(3).mean()
    
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+SecondStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    #ax1.scatter(AwayTeamInfo.index,AwayTeamInfo[SecondStat],color='black')
    ax1.plot(AwayTeamInfo[PomStatAway],color='green')
    ax1.plot(AwayTeamInfo["EMOver"],color='red')
    #ax1.plot(AwayTeamInfo["EM5"],color='black')
    ax1.plot(AwayTeamInfo["EM10"],color='purple')
    
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='dodgerblue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    #ax2.scatter(HomeTeamInfo.index,HomeTeamInfo[FirstStat],color='black')
    ax2.plot(HomeTeamInfo[PomStatHome],color='green')
    ax2.plot(HomeTeamInfo["EMOver"],color='red')
    #ax2.plot(HomeTeamInfo["EM5"],color='black')
    ax2.plot(HomeTeamInfo["EM10"],color='purple')

    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat],color='dodgerblue')
    st.pyplot(f)
    #plt.show()
def GetTwoChartsTogether_EMA_2024(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,SecondStat,PomStatHome,PomStatAway,VegasStat):
    #HomeTeamInfo["EM3"]=HomeTeamInfo['AdjO3ExpMA']-HomeTeamInfo['AdjD3ExpMA']
    #HomeTeamInfo["EM5"]=HomeTeamInfo['AdjO5ExpMA']-HomeTeamInfo['AdjD5ExpMA']
    #HomeTeamInfo["EM10"]=HomeTeamInfo['AdjO10ExpMA']-HomeTeamInfo['AdjD10ExpMA']
    HomeTeamInfo["EMOver"]=HomeTeamInfo['PlayingOverRating'].rolling(5).mean()
    #AwayTeamInfo["EM3"]=AwayTeamInfo['AdjO3ExpMA']-AwayTeamInfo['AdjD3ExpMA']
    #AwayTeamInfo["EM5"]=AwayTeamInfo['AdjO5ExpMA']-AwayTeamInfo['AdjD5ExpMA']
    #AwayTeamInfo["EM10"]=AwayTeamInfo['AdjO10ExpMA']-AwayTeamInfo['AdjD10ExpMA']
    AwayTeamInfo["EMOver"]=AwayTeamInfo['PlayingOverRating'].rolling(5).mean()
    #st.dataframe(HomeTeamInfo)
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
    ChartTitleName=AwayTeam+" "+SecondStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.plot(AwayTeamInfo.index,AwayTeamInfo[SecondStat],color='black')
    ax1.plot(AwayTeamInfo[PomStatAway],color='green')
    ax1.plot(AwayTeamInfo["EMOver"],color='red')
    #ax1.plot(AwayTeamInfo["EMRating5GameExpMA"],color='black')
    ax1.plot(AwayTeamInfo["EMRating10GameExpMA"],color='purple')
    #st.dataframe(AwayTeamInfo)
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='dodgerblue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.plot(HomeTeamInfo.index,HomeTeamInfo[FirstStat],color='black')
    ax2.plot(HomeTeamInfo[PomStatHome],color='green')
    ax2.plot(HomeTeamInfo["EMOver"],color='red')
    #ax2.plot(HomeTeamInfo["EMRating5GameExpMA"],color='black')
    ax2.plot(HomeTeamInfo["EMRating10GameExpMA"],color='purple')

    ax2.bar(HomeTeamInfo.index,HomeTeamInfo[VegasStat],color='dodgerblue')
    st.pyplot(f)
def GetTwoChartsTogether_EMA_2023(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,SecondStat,PomStatHome,PomStatAway,VegasStat):
    HomeTeamInfo["EM3"]=HomeTeamInfo['AdjO3ExpMA']-HomeTeamInfo['AdjD3ExpMA']
    HomeTeamInfo["EM5"]=HomeTeamInfo['AdjO5ExpMA']-HomeTeamInfo['AdjD5ExpMA']
    HomeTeamInfo["EM10"]=HomeTeamInfo['AdjO10ExpMA']-HomeTeamInfo['AdjD10ExpMA']
    HomeTeamInfo["EMOver"]=HomeTeamInfo['PlayingOverRating'].rolling(5).mean()
    AwayTeamInfo["EM3"]=AwayTeamInfo['AdjO3ExpMA']-AwayTeamInfo['AdjD3ExpMA']
    AwayTeamInfo["EM5"]=AwayTeamInfo['AdjO5ExpMA']-AwayTeamInfo['AdjD5ExpMA']
    AwayTeamInfo["EM10"]=AwayTeamInfo['AdjO10ExpMA']-AwayTeamInfo['AdjD10ExpMA']
    AwayTeamInfo["EMOver"]=AwayTeamInfo['PlayingOverRating'].rolling(5).mean()
    
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ChartTitleName=AwayTeam+" "+SecondStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.plot(AwayTeamInfo.index,AwayTeamInfo[SecondStat],color='black')
    ax1.plot(AwayTeamInfo[PomStatAway],color='green')
    ax1.plot(AwayTeamInfo["EMOver"],color='red')
    #ax1.plot(AwayTeamInfo["EM5"],color='black')
    ax1.plot(AwayTeamInfo["EM10"],color='purple')
    
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='dodgerblue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.plot(HomeTeamInfo.index,HomeTeamInfo[FirstStat],color='black')
    ax2.plot(HomeTeamInfo[PomStatHome],color='green')
    ax2.plot(HomeTeamInfo["EMOver"],color='red')
    #ax2.plot(HomeTeamInfo["EM5"],color='black')
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
def GetTwoTeamChartsTogether2024(AwayTeamInfo,HomeTeamInfo,AwayTeam,HomeTeam,FirstStat,VegasStat):
    HomeTeamInfo["First 3 Game"]=HomeTeamInfo[FirstStat].rolling(3).mean()
    HomeTeamInfo["First 5 Game"]=HomeTeamInfo[FirstStat].rolling(5).mean()
    HomeTeamInfo["First 10 Game"]=HomeTeamInfo[FirstStat].rolling(10).mean()
    
    AwayTeamInfo["Second 3 Game"]=AwayTeamInfo[FirstStat].rolling(3).mean()
    AwayTeamInfo["Second 5 Game"]=AwayTeamInfo[FirstStat].rolling(5).mean()
    AwayTeamInfo["Second 10 Game"]=AwayTeamInfo[FirstStat].rolling(10).mean()
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ChartTitleName=AwayTeam+" "+FirstStat+ " and "+VegasStat
    ax1.set_title(ChartTitleName)
    ax1.scatter(AwayTeamInfo.index,AwayTeamInfo[FirstStat])
    ax1.plot(AwayTeamInfo["Pomeroy_Tm_AdjEM"])
    ax1.plot(AwayTeamInfo["Second 10 Game"],color='green')
    ax1.plot(AwayTeamInfo["Second 3 Game"],color='red')
    ax1.bar(AwayTeamInfo.index,AwayTeamInfo[VegasStat],color='blue')
    ChartTitleName=HomeTeam+" "+FirstStat+ " and "+VegasStat
    ax2.set_title(ChartTitleName)
    ax2.scatter(HomeTeamInfo.index,HomeTeamInfo[FirstStat])
    ax2.plot(HomeTeamInfo["Pomeroy_Tm_AdjEM"])
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

def getDistributionMatchupCharts2024(AwayTeam,HomeTeam,test1,test2):
    
    teamname1=AwayTeam
    #test1=GetThisTeamInfoFromCsv(teamname1,"TeamDataFilesStarter2022")
    teamname2=HomeTeam
    #test2=GetThisTeamInfoFromCsv(teamname2,"TeamDataFilesStarter2022")

    test2EFG=test2['Tm_O_EFG']
    test2TO=test2['Tm_O_TO']
    test2OR=test2['Tm_O_OR']
    test2FTR=test2['Tm_O_FTR']
    test2ADJO=test2['Tm_AdjO']
    test2ADJD=test2['Tm_AdjD']
    test2ADJEM=test2['EMRating']


    test1EFG=random.choices(list(test1['Tm_O_EFG']), k=len(test2['Tm_O_EFG']))
    test1TO=random.choices(list(test1['Tm_O_TO']), k=len(test2['Tm_O_TO']))
    test1OR=random.choices(list(test1['Tm_O_OR']), k=len(test2['Tm_O_OR']))
    test1FTR=random.choices(list(test1['Tm_O_FTR']), k=len(test2['Tm_O_FTR']))
    test1ADJO=random.choices(list(test1['Tm_AdjO']), k=len(test2['Tm_AdjO']))
    test1ADJD=random.choices(list(test1['Tm_AdjD']), k=len(test2['Tm_AdjD']))
    test1ADJEM=random.choices(list(test1['EMRating']), k=len(test2['EMRating']))

    data = pd.DataFrame({teamname1:test1EFG,teamname2:test2EFG})

    data['Stat']='Tm_O_EFG'

    data2 = pd.DataFrame({teamname1:test1TO,teamname2:test2TO})
    data2['Stat']='Tm_O_TO'
    #data2['Base']=1
    data3 = pd.DataFrame({teamname1:test1OR,teamname2:test2OR})
    data3['Stat']='Tm_O_OR'
    #data2['Base']=1
    data5 = pd.DataFrame({teamname1:test1FTR,teamname2:test2FTR})
    data5['Stat']='Tm_O_FTR'
    data6 = pd.DataFrame({teamname1:test1ADJO,teamname2:test2ADJO})
    data6['Stat']='Tm_AdjO'
    data7 = pd.DataFrame({teamname1:test1ADJD,teamname2:test2ADJD})
    data7['Stat']='Tm_AdjD'
    data8 = pd.DataFrame({teamname1:test1ADJEM,teamname2:test2ADJEM})
    data8['Stat']='AdjEM'


    dataOne=pd.concat([data6,data7])
    data4=pd.concat([data,data2,data3,data5])
    st.dataframe(dataOne)
    st.dataframe(data4)
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

def getTeamDFTable2024(team1,teamname):
    colsM=['DateNew','Opp','Result_x','ATS','EMRating','PlayingOverRating','pace','ATSVegas','OverUnderVegas']
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
    gb.configure_column('ATS',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_column('EMRating',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()

    AgGrid(team1, gridOptions=gridOptions, enable_enterprise_modules=True,height=1000,allow_unsafe_jscode=True)

    
def getTodaysGamesData(Dailyschedule,TeamDatabase,PomeroyDF1,BartDF1,MG_DF1):
    
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
    for x in range(len(Dailyschedule.index)):
    
        AwayTeam=Dailyschedule['AWAY'].iloc[x]
        HomeTeam=Dailyschedule['HOME'].iloc[x]
        TimeGame=Dailyschedule['Time'].iloc[x]
        whereisGame=Dailyschedule['Court'].iloc[x]
        test1=GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles2022")
        test2=GetThisTeamInfoFromCsv(HomeTeam,"TeamDataFiles2022")

        AwayTeamB=TeamDatabase.loc[AwayTeam,"UpdatedTRankName"]
        HomeTeamB=TeamDatabase.loc[HomeTeam,"UpdatedTRankName"]

        AwayTeamP=TeamDatabase.loc[AwayTeam,"UpdatedTRankName"]
        HomeTeamP=TeamDatabase.loc[HomeTeam,"UpdatedTRankName"]
    
        AwayTeamM=TeamDatabase.loc[AwayTeam,"SportsReference"]
        HomeTeamM=TeamDatabase.loc[HomeTeam,"SportsReference"]
        test1['AdjEM_MG']=test1['AdjOE_MG']-test1['AdjDE_MG']
        test2['AdjEM_MG']=test2['AdjOE_MG']-test2['AdjDE_MG']

        team2Signals=list(test2["SignalSum"])
        team1Signals=list(test1["SignalSum"])

        if len(test2)>0:
        
            HomeTeamSignalScore=team2Signals[-1]
            team2Play=list(test2["PlayingOverRating"])
            HomeTeamSignalPlay=team2Play[-1]
            team2PlaySum=list(test2["DifCumSum"])
            HomeTeamSignalPlayOver=team2PlaySum[-1]

            OverplayingList.append([HomeTeamB,HomeTeamSignalPlay,HomeTeamSignalPlayOver])
        else:
            HomeTeamSignalPlay=0
            HomeTeamSignalPlayOver=0
            test2=test1

        if len(test1)>0:

            AwayTeamSignalScore=team1Signals[-1]   
            team1Play=list(test1["PlayingOverRating"])
            AwayTeamSignalPlay=team1Play[-1]
            team1PlayOver=list(test1["DifCumSum"])
            AwayTeamSignalPlayOver=team1PlayOver[-1]
        
            OverplayingList.append([AwayTeamB,AwayTeamSignalPlay,AwayTeamSignalPlayOver])

        else:
        
            AwayTeamSignalPlayOver=0
            AwayTeamSignalPlay=0  
            test1=test2

    
        if whereisGame =="N":

            PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NF.SetPomeroyDataNeutral2020(PomeroyDF1,AwayTeamP,HomeTeamP)
            BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NF.SetBartDataNeutral(BartDF1,AwayTeamB,HomeTeamB,thePGameTempo)
            MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NF.Set_MG_Data_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)

            B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NF.NewgetGamePredictionBartNeutralCourt(test1.iloc[len(test1.index)-1]["AdjO3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjD3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjO3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjD3ExpMAS"],thePGameTempo,LeagueOE)

            r,r1,theEstimatedTotal,theEstimatedSpread=NF.getMonteCarloGameScoreNeutralCourt(test1,test2,MonteCarloNumberofGames,10000,thePGameTempo)
            Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=NF.getMonteCarloGameScoreNeutralCourt(test1,test2,SecondMonteCarloNumberofGames,10000,thePGameTempo)
            MG_Rank_Score_Dif=NF.get_MG_Margin_Dif_Neutral(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
        else:
            if whereisGame =="A":
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NF.SetPomeroyData2020(PomeroyDF1,HomeTeamP,AwayTeamP)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NF.SetBartData(BartDF1,HomeTeamB,AwayTeamB,thePGameTempo)
        
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NF.Set_MG_Data(MG_DF1,HomeTeamM,AwayTeamM,thePGameTempo)

                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NF.NewgetGamePredictionBart(test1.iloc[len(test1.index)-1]["AdjO3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjD3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjO3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjD3ExpMAS"],thePGameTempo,LeagueOE)

                r,r1,theEstimatedTotal,theEstimatedSpread=NF.getMonteCarloGameScore(test2,test1,MonteCarloNumberofGames,10000,thePGameTempo)
                Total10G,Spread10G,theEstimatedTotal10G,theEstimatedSpread10G=NF.getMonteCarloGameScore(test2,test1,SecondMonteCarloNumberofGames,10000,thePGameTempo)
                MG_Rank_Scoree_Dif=NF.get_MG_Margin_Dif_Ratio(MG_DF1,HomeTeamM,AwayTeamM,thePGameTempo)
            else:
                PAwayTeamScore,PHomeTeamScore,PTotalPoints,PHomeTeamSpread,thePGameTempo=NF.SetPomeroyData2020(PomeroyDF1,AwayTeamP,HomeTeamP)
                BAwayTeamScore,BHomeTeamScore,BTotalPoints,BHomeTeamSpread,theBGameTempo=NF.SetBartData(BartDF1,AwayTeamB,HomeTeamB,thePGameTempo)
                MAwayTeamScore,MHomeTeamScore,MTotalPoints,MHomeTeamSpread,theMGameTempo=NF.Set_MG_Data(MG_DF1,AwayTeamM,HomeTeamM,thePGameTempo)
                B3GAwayTeamScore,B3GHomeTeamScore,B3GTotalPoints,B3GHomeTeamSpread,theB3GGameTempo=NF.NewgetGamePredictionBart(test2.iloc[len(test2.index)-1]["AdjO3ExpMAS"],test2.iloc[len(test2.index)-1]["AdjD3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjO3ExpMAS"],test1.iloc[len(test1.index)-1]["AdjD3ExpMAS"],thePGameTempo,LeagueOE)

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
    
    
    
    
        AwayVScore,HomeVScore=NF.GetVegasProjectedScore(Dailyschedule['VegasTotal'].iloc[x],Dailyschedule['VegasSpread'].iloc[x])

    
        edgeAgainstVegasTotal=NF.stats.percentileofscore(r, Dailyschedule['VegasTotal'].iloc[x])
        edgeAgainstVegasSpread=NF.stats.percentileofscore(r1, Dailyschedule['VegasSpread'].iloc[x])
        edgeAgainstVegasTotal10G=NF.stats.percentileofscore(Total10G, Dailyschedule['VegasTotal'].iloc[x])
        edgeAgainstVegasSpread10G=NF.stats.percentileofscore(Spread10G, Dailyschedule['VegasSpread'].iloc[x])
    
        theSpreads=[BHomeTeamSpread,MHomeTeamSpread,theEstimatedSpread,theEstimatedSpread10G,PHomeTeamSpread]
        theSpreadsMG=[BHomeTeamSpread,MHomeTeamSpread,MG_Rank_Score_Dif,theEstimatedSpread,theEstimatedSpread10G,PHomeTeamSpread]
        theOUs=[BTotalPoints,MTotalPoints,theEstimatedTotal,theEstimatedTotal10G,PTotalPoints]
    

        theSpreadTeamPicks=NF.getDailyPredictionTeamsAgainstSpreadDec18(theSpreads,Dailyschedule['VegasSpread'].iloc[x],round(MHomeTeamSpread,2),round(theEstimatedSpread,2),HomeTeam,AwayTeam,AwayTeamSignalScore, HomeTeamSignalScore,AwayTeamSignalPlay,HomeTeamSignalPlay)
        the_OU_TeamPicks=NF.getDailyPredictionTeams_OU_Dec18(theOUs,Dailyschedule['VegasTotal'].iloc[x],MTotalPoints,theEstimatedTotal,HomeTeam,TimeGame)
        theSpreadTeamPicksMG=NF.getDailyPredictionTeamsAgainstSpreadDec18(theSpreadsMG,Dailyschedule['VegasSpread'].iloc[x],round(MHomeTeamSpread,2),round(theEstimatedSpread,2),HomeTeam,AwayTeam,AwayTeamSignalScore, HomeTeamSignalScore,AwayTeamSignalPlay,HomeTeamSignalPlay)
   

        theSpreadTeamPicks.append(TimeGame)
        theSpreadTeamPicksMG.append(round(MG_Rank_Score_Dif,2))
        theSpreadTeamPicksMG.append(theEstimatedSpread10G)
        theSpreadTeamPicksMG.append(TimeGame)
    
        appendTeamList.append(theSpreadTeamPicks)
        appendTeamListMG.append(theSpreadTeamPicksMG)

        appendTeam_OU_List.append(the_OU_TeamPicks)

        TestFrame2 = [(AwayTeam, Dailyschedule['VegasTotal'].iloc[x],round(MTotalPoints,2),round(PTotalPoints,2),round(BTotalPoints,2),round(B3GTotalPoints,2),round(theEstimatedTotal,2),round(theEstimatedTotal10G,2)), (HomeTeam, Dailyschedule['VegasSpread'].iloc[x],round(MHomeTeamSpread,2),round(PHomeTeamSpread,2),round(BHomeTeamSpread,2),round(B3GHomeTeamSpread,2),round(theEstimatedSpread,2),round(theEstimatedSpread10G,2))]
        TestFrame=[(AwayTeam, AwayTeamSignalScore,AwayVScore,round(PAwayTeamScore,2),AwayTeamSignalPlay,edgeAgainstVegasTotal,edgeAgainstVegasTotal10G), (HomeTeam, HomeTeamSignalScore,HomeVScore,round(PHomeTeamScore,2),HomeTeamSignalPlay,edgeAgainstVegasSpread,edgeAgainstVegasSpread10G)]
        TestFrameMG = [(AwayTeam, Dailyschedule['VegasTotal'].iloc[x],round(MTotalPoints,2),0,round(theEstimatedTotal,2),round(B3GTotalPoints,2),round(BTotalPoints,2),round(theEstimatedTotal10G,2)), (HomeTeam, Dailyschedule['VegasSpread'].iloc[x],round(MHomeTeamSpread,2),round(MG_Rank_Score_Dif,2),round(theEstimatedSpread,2),round(B3GHomeTeamSpread,2),round(BHomeTeamSpread,2),round(theEstimatedSpread10G,2))]

        j=pd.DataFrame.from_records(TestFrame2, columns=['Teams', 'Vegas','MG','Pom','TR','3G','MC5','MC10'])
        j1=pd.DataFrame.from_records(TestFrame, columns=['Teams', 'Trend','VScore','PomScore','PlayingO','Edge5','Edge10'])
        j3=pd.DataFrame.from_records(TestFrameMG, columns=['Teams', 'Vegas','MG','MG_Margin','MC5','3G','TR','MC10'])

        OverplayingList2.append([HomeTeamB,HomeTeamSignalPlay,HomeTeamSignalPlayOver,HomeTeamSignalScore,AwayTeamB,AwayTeamSignalPlay,AwayTeamSignalPlayOver,AwayTeamSignalScore,TimeGame,round(PHomeTeamSpread,2)])
    
        appended_data1.append(j)
        appended_data1MG.append(j3)
        appended_dataExtra.append(j1)

        TestFrameTime = [(AwayTeam, Dailyschedule['VegasTotal'].iloc[x],round(MTotalPoints,2),round(PTotalPoints,2),round(BTotalPoints,2),round(B3GTotalPoints,2),round(theEstimatedTotal,2),round(theEstimatedTotal10G,2),TimeGame), (HomeTeam, Dailyschedule['VegasSpread'].iloc[x],round(MHomeTeamSpread,2),round(PHomeTeamSpread,2),round(BHomeTeamSpread,2),round(B3GHomeTeamSpread,2),round(theEstimatedSpread,2),round(theEstimatedSpread10G,2),TimeGame)]

        jTime=pd.DataFrame.from_records(TestFrameTime, columns=['Teams', 'Vegas','MG','Pom','TR','3G','MC5','MC10','Time'])
        appended_dataTime.append(jTime)

    appended_data2=pd.concat(appended_data1,axis=0)
    labels=["TRank","TG3","MC5","MC10","Pom","HomeTeam","SignalScoreTotal","OverPlay","Spread","MC10Spread","MCEdge"]
    return(appended_data2,appended_dataTime,appended_dataExtra,appended_data1MG,appendTeam_OU_List,appendTeamListMG,appendTeamList,theSpreadTeamPicksMG,theSpreadTeamPicks)
def plot_line_chart(df, teams):
    # Filter the dataframe for the selected teams
    df = df[df['Tm_'].isin(teams)]

    # Convert the 'Date_zero' column to datetime
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])

    # Sort the dataframe by date
    df = df.sort_values('Date_zero')

    # Set the style of seaborn to fivethirtyeight
    plt.style.use('fivethirtyeight')

    # Create the line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for team in teams:
        df_team = df[df['Tm_'] == team]
        sns.lineplot(x='Date_zero', y='tm_margin_net_eff', data=df_team, ax=ax, label=team)

    # Make x-axis labels bigger and rotate them if necessary
    #ax.xaxis.set_major_locator(mdates.MonthLocator())  # to ensure a tick every month
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # to format the date as 'Month Year'
    plt.xticks(rotation=45, fontsize=10)  # rotate x-axis labels and make them bigger

    # Set the title and labels
    plt.title('Margin Net Over Time', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Margin Net', fontsize=15)

    # Show the plot
    st.pyplot(fig)

def getHotColdTeams(df):
    
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])
    maxdate =df['Date_zero'].max()
    # Create a list to store the dataframes
    dfs = []

    # Iterate over the unique teams
    for team in df['Team'].unique():
        # Filter the rows for the current team
        team_df = df[df['Team'] == team]
        # Sort the dataframe by 'Date_zero'
        team_df = team_df.sort_values('Date_zero')
        # Calculate the change in performance
        team_df['performance_change'] = team_df['ATS_net_eff'].diff(periods=14)
        # Append the dataframe to 'dfs'
        dfs.append(team_df)

    # Concatenate all the dataframes in 'dfs'
    change_df = pd.concat(dfs)
    hotteams = change_df[change_df['Date_zero']==maxdate].sort_values('performance_change',ascending=False)
    coldteams = change_df[change_df['Date_zero']==maxdate].sort_values('performance_change',ascending=True)
    return(hotteams,coldteams)
def plot_line_chartLetsPlotHot(df, teams):
    # Filter the dataframe for the selected teams
    df = df[df['Team'].isin(teams)]

    # Convert the 'Date_zero' column to datetime
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])

    # Sort the dataframe by date
    df = df.sort_values('Date_zero')

    # Create the line chart
    for team in teams:
        df_team = df[df['Team'] == team]
        p = ggplot(df_team, aes(x='Date_zero', y='ATS_net_eff')) + \
            geom_line(color='red', size=1.5) + \
            ggtitle('Margin Net Over Time') + \
            xlab('Date') + \
            ylab('Margin Net') + \
            theme(axis_text_x=element_text(angle=45, hjust=1))
        #st.write(p)
        #st_letsplot(p)
        #st.pyplot(p)
    p = ggplot(df, aes(x='Date_zero', y='ATS_net_eff', group='Team')) + \
    geom_line(aes(color='Team'), size=1, alpha=0.5)+ggtitle("ATS Net Rating") + \
    ggsize(800, 600)
    st_letsplot(p)

def plot_line_chartLetsPlot(df, teams):
    # Filter the dataframe for the selected teams
    df = df[df['Tm_'].isin(teams)]

    # Convert the 'Date_zero' column to datetime
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])

    # Sort the dataframe by date
    df = df.sort_values('Date_zero')

    # Create the line chart
    for team in teams:
        df_team = df[df['Tm_'] == team]
        p = ggplot(df_team, aes(x='Date_zero', y='tm_margin_net_eff')) + \
            geom_line(color='red', size=1.5) + \
            ggtitle('Margin Net Over Time') + \
            xlab('Date') + \
            ylab('Margin Net') + \
            theme(axis_text_x=element_text(angle=45, hjust=1))
        #st.write(p)
        #st_letsplot(p)
        #st.pyplot(p)
    p = ggplot(df, aes(x='Date_zero', y='tm_margin_net_eff', group='Tm_')) + \
    geom_line(aes(color='Tm_'), size=1, alpha=0.5)+ggtitle("ATS Net Rating") + \
    ggsize(800, 800)
    st_letsplot(p)
def displayTeamDistributions(Gamesdf,myteam):
    import streamlit.components.v1 as components
    dff1 = Gamesdf[Gamesdf['Tm']==myteam][['Tm','Opp','Tm_AdjO','Tm_AdjD','Tm_O_PPP','Tm_O_EFG','Tm_O_TO','Tm_O_OR','Tm_O_FTR','Tm_D_PPP','Tm_D_EFG','Tm_D_TO','Tm_D_OR','Tm_D_FTR','Tempo','EMRating']]
    col = ['Tm','Tm_AdjO','Tm_AdjD','Tm_O_PPP','Tm_O_EFG','Tm_O_TO','Tm_O_OR','Tm_O_FTR','Tm_D_PPP','Tm_D_EFG','Tm_D_TO','Tm_D_OR','Tm_D_FTR','Tempo','EMRating']

    dff1 = dff1.rename(columns={
        'Tm': 'Team',
        'Tm_AdjO': 'AdjO',
        'Tm_AdjD': 'AdjD',
    'Tm_O_EFG':'O_EFG%',
    'Tm_D_EFG':'D_EFG%',
        'Tm_O_PPP': 'O_PPP',
        'Tm_O_TO': 'O_TO%',
        'Tm_O_OR': 'O_OR%',
        'Tm_O_FTR': 'O_FTR',
     'Tm_D_PPP': 'D_PPP',
        'Tm_D_TO': 'D_TO%',
        'Tm_D_OR': 'D_OR%',
        'Tm_D_FTR': 'D_FTR',
        
    })
    np.random.seed(12)
    data = dict(
    cond=np.repeat(['A','B'], 200),
    rating=np.concatenate((np.random.normal(0, 1, 200), np.random.normal(1, 1.5, 200))))

    a = ggplot(data, aes(x='rating', fill='cond')) + ggsize(500, 250) + geom_density(color='dark_green', alpha=.7) + scale_fill_brewer(type='seq')+ theme(axis_line_y='blank')

    # plots any Let's Plot visualization object
    #st_letsplot(a)

    density1  = ggplot(dff1, aes(x='O_EFG%', color='Team')) + geom_density(aes(fill='Team'), alpha=.3,color='dark_green') + scale_fill_brewer(type='seq')+ ggtitle("Offensive EFG%")+ ggsize(1000, 2000)
    #st.write(density1)
    #st_letsplot(density1)
    density2  = ggplot(dff1, aes(x='O_TO%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green')+ scale_fill_brewer(type='seq')+ ggtitle("Offensive TO%")
    density3  = ggplot(dff1, aes(x='O_OR%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green')+ scale_fill_brewer(type='seq')+ ggtitle("Offensive OR%")
    density4 = ggplot(dff1, aes(x='O_FTR', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green')+ scale_fill_brewer(type='seq')+ ggtitle("Offensive FTR")
    #gggrid([density1,density2,density3,density4], ncol=4)
    density12  = ggplot(dff1, aes(x='D_EFG%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green') + scale_fill_brewer(type='seq')+ ggtitle("Defensive EFG%")
    density22  = ggplot(dff1, aes(x='D_TO%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green')+ scale_fill_brewer(type='seq')+ ggtitle("Defensive TO%")
    density32  = ggplot(dff1, aes(x='D_OR%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green')+ scale_fill_brewer(type='seq')+ ggtitle("Defensive OR%")
    density42 = ggplot(dff1, aes(x='D_FTR', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_green')+ scale_fill_brewer(type='seq')+ ggtitle("DEfensive FTR")

    density11  = ggplot(dff1, aes(x='AdjO', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_blue') + scale_fill_brewer(type='seq')+ ggtitle("Adjusted Offensive Rating")
    density21  = ggplot(dff1, aes(x='AdjD', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_blue')+ scale_fill_brewer(type='seq')+ ggtitle("Adjusted Defensive Rating")
    density31  = ggplot(dff1, aes(x='EMRating', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_blue')+ scale_fill_brewer(type='seq')+ ggtitle("Adjusted Efficiency Rating")
    density41 = ggplot(dff1, aes(x='Tempo', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3,color='dark_blue')+ scale_fill_brewer(type='seq')+ ggtitle("Pace/Tempo")
    p2 = gggrid([density1,density2,density3,density4,density12,density22,density32,density42,density11,density21,density31,density41], ncol=4)+ ggsize(1500, 800)
    st.subheader(' Distribution Charts')
    #st_letsplot(p2)
    plot_dict = p2.as_dict()
    components.html(_as_html(plot_dict), height=1500 + 20,width=1500 + 20,scrolling=True,)
def displayTeamDistributionsMatchup(Gamesdf,myteam,team2):
    import streamlit.components.v1 as components
    df = Gamesdf[Gamesdf['Tm']==myteam][['Tm','Opp','Tm_AdjO','Tm_AdjD','Tm_O_PPP','Tm_O_EFG','Tm_O_TO','Tm_O_OR','Tm_O_FTR','Tm_D_PPP','Tm_D_EFG','Tm_D_TO','Tm_D_OR','Tm_D_FTR','Tempo','EMRating']]
    df1 = Gamesdf[Gamesdf['Tm']==team2][['Tm','Opp','Tm_AdjO','Tm_AdjD','Tm_O_PPP','Tm_O_EFG','Tm_O_TO','Tm_O_OR','Tm_O_FTR','Tm_D_PPP','Tm_D_EFG','Tm_D_TO','Tm_D_OR','Tm_D_FTR','Tempo','EMRating']]
    dff1 = pd.concat([df,df1])
    col = ['Tm','Tm_AdjO','Tm_AdjD','Tm_O_PPP','Tm_O_EFG','Tm_O_TO','Tm_O_OR','Tm_O_FTR','Tm_D_PPP','Tm_D_EFG','Tm_D_TO','Tm_D_OR','Tm_D_FTR','Tempo','EMRating']

    dff1 = dff1.rename(columns={
        'Tm': 'Team',
        'Tm_AdjO': 'AdjO',
        'Tm_AdjD': 'AdjD',
    'Tm_O_EFG':'O_EFG%',
    'Tm_D_EFG':'D_EFG%',
        'Tm_O_PPP': 'O_PPP',
        'Tm_O_TO': 'O_TO%',
        'Tm_O_OR': 'O_OR%',
        'Tm_O_FTR': 'O_FTR',
     'Tm_D_PPP': 'D_PPP',
        'Tm_D_TO': 'D_TO%',
        'Tm_D_OR': 'D_OR%',
        'Tm_D_FTR': 'D_FTR',
        
    })
    

    density1  = ggplot(dff1, aes(x='O_EFG%', color='Team')) + geom_density(aes(fill='Team'), alpha=.6) + ggtitle("Offensive EFG%")+ ggsize(1000, 2000)
    density2  = ggplot(dff1, aes(x='O_TO%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.6,)+ ggtitle("Offensive TO%")
    density3  = ggplot(dff1, aes(x='O_OR%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.6)+ ggtitle("Offensive OR%")
    density4 = ggplot(dff1, aes(x='O_FTR', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.6)+ ggtitle("Offensive FTR")
    
    density12  = ggplot(dff1, aes(x='D_EFG%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3) + scale_fill_brewer(type='seq')+ ggtitle("Defensive EFG%")
    density22  = ggplot(dff1, aes(x='D_TO%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3)+ scale_fill_brewer(type='seq')+ ggtitle("Defensive TO%")
    density32  = ggplot(dff1, aes(x='D_OR%', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3)+ scale_fill_brewer(type='seq')+ ggtitle("Defensive OR%")
    density42 = ggplot(dff1, aes(x='D_FTR', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3)+ scale_fill_brewer(type='seq')+ ggtitle("DEfensive FTR")

    density11  = ggplot(dff1, aes(x='AdjO', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3) + scale_fill_brewer(type='seq')+ ggtitle("Adjusted Offensive Rating")
    density21  = ggplot(dff1, aes(x='AdjD', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3)+ scale_fill_brewer(type='seq')+ ggtitle("Adjusted Defensive Rating")
    density31  = ggplot(dff1, aes(x='EMRating', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3)+ scale_fill_brewer(type='seq')+ ggtitle("Adjusted Efficiency Rating")
    density41 = ggplot(dff1, aes(x='Tempo', color='Team')) + ggsize(800, 550)+ geom_density(aes(fill='Team'), alpha=.3)+ scale_fill_brewer(type='seq')+ ggtitle("Pace/Tempo")
    p2 = gggrid([density1,density2,density3,density4,density12,density22,density32,density42,density11,density21,density31,density41], ncol=4)+ ggsize(2000, 1000)
    st.subheader(' Distribution Charts')
    #st_letsplot(p2)
    plot_dict = p2.as_dict()
    components.html(_as_html(plot_dict), height=1500 + 20,width=3000 + 20,scrolling=True,)

def get_team_info_from_gamesdf(df,Team):
    AF = df[df['Tm']==Team].sort_values('DateNew')
    


    AF["EMRating3GameExpMA"]=AF["EMRating"].ewm(span=3,adjust=False).mean()
    AF["EMRating5GameExpMA"]=AF["EMRating"].ewm(span=5,adjust=False).mean()
    AF["EMRating10GameExpMA"]=AF["EMRating"].ewm(span=10,adjust=False).mean()

    AF["AdjO3GameExpMA"]=AF["Tm_AdjO"].ewm(span=3,adjust=False).mean()
    AF["AdjO5GameExpMA"]=AF["Tm_AdjO"].ewm(span=5,adjust=False).mean()
    AF["AdjO10GameExpMA"]=AF["Tm_AdjO"].ewm(span=10,adjust=False).mean()

    AF["AdjD3GameExpMA"]=AF["Tm_AdjD"].ewm(span=3,adjust=False).mean()
    AF["AdjD5GameExpMA"]=AF["Tm_AdjD"].ewm(span=5,adjust=False).mean()
    AF["AdjD10GameExpMA"]=AF["Tm_AdjD"].ewm(span=10,adjust=False).mean()
    
    AF["PlayingOverRating"]=AF["EMRating5GameExpMA"] - AF["Pomeroy_Tm_AdjEM"]
    AF["GameDifRating"]=AF["EMRating"]-AF["Pomeroy_Tm_AdjEM"]
    AF["DifCumSum"]=AF["GameDifRating"].cumsum()
    AF["DifCumSumEMA"]=AF["DifCumSum"].ewm(span=5,adjust=False).mean()
    AF["DifCumSum"]=AF["DifCumSum"].shift(1).fillna(0)
    AF["DifCumSumEMA"]=AF["DifCumSumEMA"].shift(1).fillna(0)
    AF["EMOver"]=AF['PlayingOverRating'].rolling(5).mean()
    AF["PPP_3GameExpMA"]=AF["Tm_O_PPP"].ewm(span=3,adjust=False).mean()    
    AF["PPP_10GameExpMA"]=AF["Tm_O_PPP"].ewm(span=10,adjust=False).mean()
    AF["PPP_D_3GameExpMA"]=AF["Tm_D_PPP"].ewm(span=3,adjust=False).mean()    
    AF["PPP_D_10GameExpMA"]=AF["Tm_D_PPP"].ewm(span=10,adjust=False).mean()    
    
    #AF.reset.index()
    return(AF)
def getTodaysDateFormat():
    
    from datetime import datetime
    import pytz

    # Create a timezone object for the Central Time Zone
    central = pytz.timezone('US/Central')

    # Get the current time in the Central Time Zone
    central_time = datetime.now(central)
    one_day_before = central_time - timedelta(days=1)
    # Change the format to 'YYYYMMDD'
    formatted_time = central_time.strftime('%Y%m%d')
    formatted_time_before = one_day_before.strftime('%Y%m%d')
    return(formatted_time,formatted_time_before)

def keep_first_four(df):
    # Group the dataframe by 'Seed' and keep only the first 4 rows of each group
    df = df.groupby('Seed').head(4)
    return df
def get_next_region(seed):
    
    region_index = seed_region[seed]
    region = regions[region_index]
    seed_region[seed] = (region_index + 1) % len(regions)
    return region
def getBracketMatrixDataframe():
    dfe = GetBracketMatrix()
    dfe = dfe[:68]
    dfe.columns = ['Seed', 'Team', 'Conference', 'Avg Seed']
    regions = ['south', 'east', 'midwest', 'west']

    # Create a dictionary to store the current region for each seed
    seed_region = {i: 0 for i in range(1, 17)}

    # Function to get the next region for a seed


    # Apply the function to the 'Seed' column to create the 'region' column
    dfe['region'] = dfe['Seed'].apply(get_next_region)

    dfe = keep_first_four(dfe)
    #st.dataframe(dfe)
    dfp = dfe.sort_values('Seed')
    #st.dataframe(dfp)
    BracketProjections= dfp.pivot(index='Seed', columns='region', values='Team')

    # Reset the index
    BracketProjections.reset_index(inplace=True)

    BracketProjections["seedings"]=(0,14,10,6,4,8,12,2,3,13,9,5,7,11,15,1)
    thisBracketProjection=BracketProjections.sort_values(by=['seedings'])
    return(thisBracketProjection)



# working TR
def getTRankBracket():
    
    response = requests.get('https://barttorvik.com/now_seeding.json#')
    #response = requests.get('https://barttorvik.com/tranketology.php?&json=1')

    text = response.text
    start_index = text.find('[[')
    end_index = text.rfind(']]') + 2
    json_text = text[start_index:end_index]
    #data = json.loads(json_text)
    #df = pd.DataFrame(data)

    json.loads(text)
    data_dict = json.loads(text)

    # Convert Python dictionary to pandas dataframe
    df = pd.DataFrame(list(data_dict.items()), columns=['Team', 'Seed'])
    seeds_to_drop = {11: 2, 16: 2}

    for seed, drop_count in seeds_to_drop.items():
        # Get the indices of the rows to drop
        indices_to_drop = df[df['Seed'] == seed].index[:drop_count]
        # Drop the rows
        df = df.drop(indices_to_drop)
    regions = ['south', 'east', 'midwest', 'west']

    # Create a dictionary to store the current region for each seed
    seed_region = {i: 0 for i in range(1, 17)}
    # Apply the function to the 'Seed' column to create the 'region' column
    df['region'] = df['Seed'].apply(get_next_region)

    dfp = df.sort_values('Seed')
    BracketProjections= dfp.pivot(index='Seed', columns='region', values='Team')

    # Reset the index
    BracketProjections.reset_index(inplace=True)

    BracketProjections["seedings"]=(0,14,10,6,4,8,12,2,3,13,9,5,7,11,15,1)
    thisBracketProjection=BracketProjections.sort_values(by=['seedings'])
    return(thisBracketProjection)
def get2023Display(Dailyschedule,dateToday,d2,season):
    TeamDatabase2=pd.read_csv("Data/TeamDatabase2023.csv")
    AllGames=pd.read_csv("Data/Season_GamesAll.csv")
    AwayTeamAll=list(TeamDatabase2['OldTRankName'])
    HomeTeamAll=list(TeamDatabase2['OldTRankName'])
    Tables_Selection=st.sidebar.selectbox('Any or Scheduled ',['Any', 'Todays Games','All Games'])
    if 'All Games' in  Tables_Selection:
        allcols=AllGames.columns
        gb = GridOptionsBuilder.from_dataframe(AllGames,groupable=True)
        gb.configure_columns(allcols, cellStyle=cellStyle)
        csTotal=cellStyleDynamic(Dailyschedule.Reg_dif)
        #gb.configure_column('Reg_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
        #gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        #gridOptions = gb.build()
        opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
        gb.configure_grid_options(**opts)
        keyname='Test All'
        g = _displayGrid(AllGames, gb, key=keyname, height=1200)
    if 'Any' in  Tables_Selection:
        AwayTeam = st.sidebar.selectbox('Away Team',AwayTeamAll)
        HomeTeam = st.sidebar.selectbox('Home Team',HomeTeamAll)
    if 'Todays Games' in  Tables_Selection:
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

    if st.button('Run'):
        dateforRankings=dateToday
        dateforRankings5=d2
        #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)
        MG_DF1=pd.read_csv("Data/MGRankings"+season+"/tm_seasons_stats_ranks"+dateforRankings5+" .csv")
        MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
        MG_DF1.set_index("updated", inplace=True)
        from matplotlib.backends.backend_pdf import PdfPages
        WhichFile='TeamDataFiles'+season
        pp= PdfPages("Daily_Team_Charts_"+dateToday+".pdf")
        if 'Todays Games' in  Tables_Selection:
            st.header('Sortable NCAA Game Schedule')
            st.text('Games can be sorted by columns. Click on column header to sort')
            st.text('To sort by game time click the Time column.  ')
            st.text('Low Negative values in the Reg Dif and Overplaying column mean the Home team is the pick  ')  
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
            gb = GridOptionsBuilder.from_dataframe(Dailyschedule,groupable=True)
            gb.configure_columns(allcols, cellStyle=cellStyle)
            csTotal=cellStyleDynamic(Dailyschedule.Reg_dif)
            gb.configure_column('Reg_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))

            #gb.configure_pagination()
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                #gridOptions = gb.build()
            opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
            gb.configure_grid_options(**opts)
            keyname='Test'
            g = _displayGrid(Dailyschedule, gb, key=keyname, height=800)
            #AgGrid(Dailyschedule, gridOptions=gridOptions, enable_enterprise_modules=True,allow_unsafe_jscode=True,height=800)

        TeamDatabase=pd.read_csv("Data/TeamDatabase"+season+".csv")
        TeamDatabase.set_index("OldTRankName", inplace=True)
        Dailyschedule['VegasSpread'] = Dailyschedule.VegasSpread.apply(calculate_to_numeric)
        Dailyschedule['Total'] = Dailyschedule.VegasTotal.apply(calculate_to_numeric)   
        PomeroyDF1=pd.read_csv("Data/PomeroyDailyRankings"+season+"/PomeroyRankings"+dateforRankings+".csv")
        PomeroyDF1["updated"]=update_type(PomeroyDF1.Team,TeamDatabase.set_index('PomeroyName').UpdatedTRankName)
        PomeroyDF1["updated"]=PomeroyDF1["updated"].str.rstrip()
   
        PomeroyDF1.set_index("updated", inplace=True)

        BartDF1=pd.read_csv("Data/TRankDailyRankings"+season+"/"+dateforRankings+".csv")
        BartDF1["updated"]=update_type(BartDF1.Team,TeamDatabase.set_index('TRankName').UpdatedTRankName)
        BartDF1.set_index("updated", inplace=True)
        import seaborn
        st.header('Team Matchup')

        #### Sidebar Creation #######

    

        test1=GetThisTeamInfoFromCsv(AwayTeam,"TeamDataFiles"+season)
        test2=GetThisTeamInfoFromCsv(HomeTeam,"TeamDataFiles"+season)
        AwayTeamB=TeamDatabase.loc[AwayTeam,"UpdatedTRankName"]
        HomeTeamB=TeamDatabase.loc[HomeTeam,"UpdatedTRankName"]

        AwayTeamP=TeamDatabase.loc[AwayTeam,"UpdatedTRankName"]
        HomeTeamP=TeamDatabase.loc[HomeTeam,"UpdatedTRankName"]
    
        AwayTeamM=TeamDatabase2.loc[AwayTeam,"SportsReference"]
        HomeTeamM=TeamDatabase2.loc[HomeTeam,"SportsReference"]
        test1['AdjEM_MG']=(test1['AdjOE_MG']-test1['AdjDE_MG'])
        test2['AdjEM_MG']=(test2['AdjOE_MG']-test2['AdjDE_MG'])

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
        try:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=2, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test1,order=2, ax=ax1, color = 'green')
        except:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=1, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test1,order=1, ax=ax1, color = 'green')
        try: 
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2, color = 'blue')
            fig4=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test2,order=2, ax=ax2, color = 'green')
        except:
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=1, ax=ax2, color = 'blue')
            fig4=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test2,order=1, ax=ax2, color = 'green')        
    
        st.subheader('Polynomial Regression Charts')
        st.text('Daily Pomeroy Rankings line in green for each game')
        st.text('Polynomial Regression of actual game performance in blue for each game ')
        st.text('If the blue line is above the green then the team is playing better than its ranking ')
        st.pyplot(fig)
#############################################
        test1['New_ID'] = range(0, 0+len(test1))
        test2['New_ID'] = range(0, 0+len(test2))
        fig_dims = (15,10)
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=fig_dims)
        #fig, axs = plt.subplots(ncols=2,figsize=fig_dims)
        plt.figure(figsize=(20, 12))
        ax1.set_title(AwayTeam)
        ax2.set_title(HomeTeam)
        try:     
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=2, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test1,order=2, ax=ax1, color = 'green')
        except:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=1, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test1,order=1, ax=ax1, color = 'green')
        try:
            fig5=sns.regplot(x='New_ID', y='AdjEM_MG', data=test1,order=2, ax=ax1, color = 'red')
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2)
        except:
            fig5=sns.regplot(x='New_ID', y='AdjEM_MG', data=test1,order=1, ax=ax1, color = 'red')
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=1, ax=ax2)
        try:
            fig4=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test2,order=2, ax=ax2, color = 'green')
            fig6=sns.regplot(x='New_ID', y='AdjEM_MG', data=test2,order=2, ax=ax2, color = 'red')
        except:
            fig4=sns.regplot(x='New_ID', y='PomAdjEMCurrent', data=test2,order=1, ax=ax2, color = 'green')
            fig6=sns.regplot(x='New_ID', y='AdjEM_MG', data=test2,order=1, ax=ax2, color = 'red')    
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
        GetTwoChartsTogether_EMA_2023(test1,test2,AwayTeam,HomeTeam,"PlayingOverRating","PlayingOverRating","PomAdjEMCurrent","PomAdjEMCurrent","ATS")
        st.subheader('MG Rankings and ATS spread')
        st.text('MG Rankings by game is the Blue Line')
        test1['MG_SpreadWinATSResult'] = test1.apply(getMGWinRecord, axis=1)
        test2['MG_SpreadWinATSResult'] = test2.apply(getMGWinRecord, axis=1)
        col1, col2 = st.columns(2)
        with col1:
            st.write('MG Rankings ATS W-L',test1['MG_SpreadWinATS'].sum(),'-',test1['MG_SpreadLossATS'].sum())
            st.write('MG Rankings Win %',test1['MG_SpreadWinATS'].sum()/test1['MG_SpreadLossATS'].count())
        with col2:
            st.write('MG Rankings ATS W-L',test2['MG_SpreadWinATS'].sum(),'-',test2['MG_SpreadLossATS'].sum())
            st.write('MG Rankings Win %',test2['MG_SpreadWinATS'].sum()/test2['MG_SpreadLossATS'].count())
        #GetTwoChartsTogether_EMA(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating",'AdjEM_MG','AdjEM_MG',"ATS")
        GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"EMRating","Adj_Margin_EM_MG","MG_SpreadWinATSResult")
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

def Historical_Rankings_Page(data):
    MG_Rank =data['MG_Rank']
    selected_teams = st.multiselect('Select teams:', teams)
    st.header('NCAA ATS Net Rating Comp')
    plot_line_chartLetsPlot(MG_Rank, selected_teams)
def set_energy_function(ef):
    global default_energy_function
    default_energy_function = ef
def default_energy_game(winner, loser):
    """This is where you'll input your own energy functions. Here are
    some of the things we talked about in class. Remember that you
    want the energy of an "expected" outcome to be lower than that of
    an upset.
    """
    #result = -(strength[winner] - strength[loser])
    #result = regional_rankings[winner] - regional_rankings[loser]
    #result = regional_rankings[winner]/regional_rankings[loser]
    #result = -(strength[winner]/strength[loser])
    result = -(strength[winner]-strength[loser])/200.0
    #result = random()
    #result = color of team 1 jersey better than color of team 2 jersey
    #print "energy_game(",winner,loser,")",result
    return result
def Bracketology_Page(data):
    #bracket_selected = st.selectbox('Select a Bracketology',['TRank','Bracket Matrix']) 
    #ranking_selected = st.selectbox('Select a Ranking for Sim',['TRank','Mg Rankings','Pomeroy'])
    BM1 = data['BM1']
    TBracket1 = data['TBracket']
    
    st.subheader('Bracket Matrix Bracketology Projection')
    
    gb = GridOptionsBuilder.from_dataframe(BM1,groupable=True)
    #csTotal=cellStyleDynamic(hot2.performance_change)
    #gb.configure_column('performance_change',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_column('Seed',valueFormatter=numberFormat(0))
    #csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
    #gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
    gb.configure_grid_options(**opts)
    keyname='Test BM'
    g = _displayGrid(BM1, gb, key=keyname, height=600)
    #st.dataframe(BM)
    
    st.subheader('TRank Bracketology Projection')

    gb = GridOptionsBuilder.from_dataframe(BM1,groupable=True)
    #csTotal=cellStyleDynamic(hot2.performance_change)
    #gb.configure_column('performance_change',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_column('Seed',valueFormatter=numberFormat(0))
    #csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
    #gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
    gb.configure_grid_options(**opts)
    keyname='Test TBracket'
    g = _displayGrid(TBracket1, gb, key=keyname, height=600)
    #st.dataframe(BM)
    #st.dataframe(TBracket)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Bracket Sim USing TRank Rankings')
        df = data['TSim']
        showBracketTable(df)
    with col2:
        st.subheader('Bracket Sim USing MG Rankings')
        df = data['MSim']
        showBracketTable(df)

def MG_Rankings(data):
    hot = data['hot']
    cold = data['cold']
    MG_Rank2 = data['MG_Rank2']
    coldlist = data['coldlist']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Who is Hot?')
        st.write('These teams have improved the most over the last two weeks in my rankings')
        st.write('The ATS premium gives you credit for outperfoming the ATS spread against a tough team')
        hot2 = hot.head(10)[['Team','ATS_net_eff','performance_change']]
        gb = GridOptionsBuilder.from_dataframe(hot2,groupable=True)
        #gb.configure_columns(allcols, cellStyle=cellStyle)
        csTotal=cellStyleDynamic(hot2.performance_change)
        gb.configure_column('performance_change',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_column('ATS_net_eff',valueFormatter=numberFormat(2))
        #csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
        #gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
        gb.configure_grid_options(**opts)
        keyname='Test 1'
        g = _displayGrid(hot2, gb, key=keyname, height=500)
        #st.image('Data/hot-icon.jpg')
        st.write('Time Series Chart of the teams playing the hottest')
        plot_line_chartLetsPlotHot(MG_Rank2, hotlist)
    with col2:
        st.subheader('Who is Not?')
        st.write('These teams have fallen the most over the last two weeks in my rankings')
        st.write('The ATS premium or penalty means my rankings are more reactive than Pomeroy')
        cold2 = cold.head(10)[['Team','ATS_net_eff','performance_change']]
        gb = GridOptionsBuilder.from_dataframe(cold2,groupable=True)
        #gb.configure_columns(allcols, cellStyle=cellStyle)
        csTotal=cellStyleDynamic(cold2.performance_change)
        #gb.configure_column('performance_change',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_column('performance_change',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_column('ATS_net_eff',valueFormatter=numberFormat(2))
        #csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
        #gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        #gridOptions = gb.build()
        opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
        gb.configure_grid_options(**opts)
        keyname='Test '
        g = _displayGrid(cold2, gb, key=keyname, height=500)
        #st.image('Data/cold-icon-3.jpg')
        st.write('Time Series Chart of the teams underperforming the last two weeks')
        plot_line_chartLetsPlotHot(MG_Rank2,coldlist)
    import streamlit.components.v1 as components
    add_selectbox_start =st.date_input('Pick date for Rankings')
    st.write('Clicking on the headers will sort that column in ascending or descending order')
    st.write('MG_NET_EFF is my rankings with no early season weighting and solely counting games played this year. This leads a reactive ranking for hot and cold teams')
    #st.write('This leads a reactive ranking for hot and cold teams')
    st.write('ATS_NET_EFF adds a premium for beating the spread against a weighted opponent ranking')
    st.write('ATS_PREMIUM highlights that premium or deficit. It should correspond to teams ATS record')
    dateString=str(add_selectbox_start)
    dateToday=dateString.replace('-', '')
    files = os.listdir('Data/MGRankings2024')

    # Filter the list to include only files that start with 'MG'
    files = [file for file in files if file.startswith('MG')]

    # Create a dictionary with the last 8 characters in the filename as the key and the filename as the value
    file_dict = {file[-8:]: file for file in files}
    #st.header("test html import")
    if dateToday in file_dict:
        # If the date is in the dictionary, select the corresponding filename
        myfilestring = file_dict[dateToday]
    else:
        # If the date is not in the dictionary, select the filename with the latest date
        latest_date = max(file_dict.keys())
        myfilestring = file_dict[latest_date]
    myfile = "Data/MGRankings2024/"+myfilestring
    HtmlFile = open(myfile, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    #print(source_code)
    components.html(source_code, height = 3000)
    if st.button('Run'):
        components.html(source_code, height = 3000)
        #col1, col2 = st.columns(2)
        #with col1:
        #    components.html(source_code, height = 3000)
        #with col2:
            #plot_line_chart(MG_Rank, selected_teams)
def Todays_Games(data):
    today_date_format = data['today_date_format']
    #Dailyschedule = data['Dailyschedule']
    #try:
    Gamesdf = pd.read_csv("Data/DailySchedules2024/Gamesdf"+today_date_format+".csv")
    Gamesdf = Gamesdf.reset_index(drop=True)
    Gamesdf.drop(columns=Gamesdf.columns[0], axis=1,  inplace=True)
    Gamesdf = Gamesdf.drop_duplicates()
    
    Tables_Choice=st.selectbox('Sort Games By',['Alphabetical', 'Time','Regression_Difference','OverPlaying'],index=0)
    Dailyschedule=pd.read_csv("Data/DailySchedules2024/"+today_date_format+"Schedule.csv")
    if 'Alphabetical'in  Tables_Choice:
        Dailyschedule=Dailyschedule.sort_values(by=['AWAY'])
    if 'Time' in Tables_Choice:
        Dailyschedule=Dailyschedule.sort_values(by=['commence_time'])   
    if 'Regression_Difference' in Tables_Choice: 
        Dailyschedule=Dailyschedule.sort_values(by=['Reg_dif'])
    if 'OverPlaying' in Tables_Choice: 
        Dailyschedule=Dailyschedule.sort_values(by=['Over_dif'])
    AwayList=[''] + Dailyschedule['AWAY'].tolist()
    HomeList=[''] + Dailyschedule['HOME'].tolist()
    AwayTeam = st.selectbox('Away Team',AwayList,index=0)
    HomeTeam = st.selectbox('Home Team',HomeList,index=0)
    st.header('Sortable NCAA Game Schedule')
    st.text('Games can be sorted by columns. Click on column header to sort')
    st.text('To sort by game time click the Time column.  ')
    st.text('Low Negative values in the Reg Dif and Overplaying column mean the Home team is the pick  ') 
    Dailyschedule = Dailyschedule[['AWAY','HOME','HomeAway','FanDuel','MG_ATS_PointDiff','commence_time','Reg_dif','Over_dif','Dif_from_Vegas','Pomeroy_PointDiff','TRank_PointDiff','MG_PointDiff','Daily_Reg_PointDiff','DraftKings','BetMGM spreads','Caesars spreads','BetRivers spreads','VegasTotal']]
    Dailyschedule.DraftKings = Dailyschedule.DraftKings.astype(float).round(1)
    Dailyschedule.VegasTotal = Dailyschedule.VegasTotal.astype(float).round(1)
    Dailyschedule['commence_time'] = pd.to_datetime(Dailyschedule['commence_time'])
    # Convert to US Central time
    Dailyschedule['commence_time'] = Dailyschedule['commence_time'].dt.tz_convert('US/Central')
    # Format time to display like 11:00AM, 2:00PM, etc.
    Dailyschedule['commence_time'] = Dailyschedule['commence_time'].dt.strftime('%I:%M%p')
    allcols=Dailyschedule.columns
    gb = GridOptionsBuilder.from_dataframe(Dailyschedule,groupable=True)
    gb.configure_columns(allcols, cellStyle=cellStyle)
    csTotal=cellStyleDynamic(Dailyschedule.Reg_dif)
    gb.configure_column('Reg_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
    csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
    gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_column('DraftKings',valueFormatter=numberFormat(1))
    gb.configure_column('VegasTotal',valueFormatter=numberFormat(1))
    gb.configure_column('Pomeroy_PointDiff',valueFormatter=numberFormat(1))
    gb.configure_column('TRank_PointDiff',valueFormatter=numberFormat(1))
    gb.configure_column('MG_PointDiff',valueFormatter=numberFormat(1))
    gb.configure_column('MG_ATS_PointDiff',valueFormatter=numberFormat(1))
    gb.configure_column('Daily_Reg_PointDiff',valueFormatter=numberFormat(1))
    gb.configure_column('Dif_from_Vegas',cellStyle=csTotal,valueFormatter=numberFormat(2))
    #gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    #gridOptions = gb.build()
    opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
    gb.configure_grid_options(**opts)
    keyname='Test'
    g = _displayGrid(Dailyschedule, gb, key=keyname, height=800)
    #AgGrid(Dailyschedule, gridOptions=gridOptions, enable_enterprise_modules=True,allow_unsafe_jscode=True,height=800)
    st.text('MG_ATS_PointDif is the point spread using the ATS model')
    st.text('Reg_dif is the differnce between both teams using a polynomial regression of current rankings')
    st.text('Over_dif is the cumulative total of how both teams having played compared to their rankings')
    st.text('A negative Over_dif means the Home teal has been overplaying relative to the away team')
    st.text('Dif_from_Vegas is the difference between the ATS model and the current market. A large value indicates a divergence')
    if st.button('Run'): 
        dateforRankings=today_date_format
         #dateforRankings5=d2
        #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)
        #MG_DF1=pd.read_csv("Data/MGRankings"+season+"/tm_seasons_stats_ranks"+dateforRankings5+" .csv")
        #MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
        #MG_DF1.set_index("updated", inplace=True)
        from matplotlib.backends.backend_pdf import PdfPages
        #WhichFile='TeamDataFiles'+season
        pp= PdfPages("Daily_Team_Charts_"+dateforRankings+".pdf")     
        st.header('Team Matchup')
        plt.style.use('seaborn')
        fig_dims = (12,10)
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=fig_dims)
        plt.figure(figsize=(16, 10))
        ax1.set_title(AwayTeam)
        ax2.set_title(HomeTeam)  
        test1=get_team_info_from_gamesdf(Gamesdf,AwayTeam)
        #st.dataframe(test1)
        test1 = test1.reset_index(drop=True)
        #test1.drop(columns=test1.columns[0], axis=1,  inplace=True)
         #test1 = test1.drop_duplicates()
        test2=get_team_info_from_gamesdf(Gamesdf,HomeTeam)
        test2 = test2.reset_index(drop=True)
        #test2.drop(columns=test2.columns[0], axis=1,  inplace=True)
        #test2 = test2.drop_duplicates()
        test1['New_ID'] = range(0, 0+len(test1))
        test2['New_ID'] = range(0, 0+len(test2))
        myteams = [AwayTeam,HomeTeam]
        plot_line_chartLetsPlotHot(MG_Rank2, myteams)
        dfI =getIndividualPlayerData()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(AwayTeam + ' Rankings')
            displayRankingHistory(data,AwayTeam)
            getTeamDFTable2024(test1,AwayTeam)
        
        with col2:
            st.subheader(HomeTeam + ' Rankings')
            displayRankingHistory(data,HomeTeam)
            getTeamDFTable2024(test2,HomeTeam)
        col1, col2 = st.columns(2)
        with col1:
            team_players = data['Players']
            #team_players = team_players[team_players['Team']==AwayTeam]
            st.subheader(AwayTeam + ' Player Data')
            showPlayersTable(team_players,AwayTeam)
            dfI_Team = dfI[dfI['Team'] == AwayTeam]
            tp = team_players[team_players['Team'] == AwayTeam].sort_values('PRPG', ascending=False)
            player1 = tp['Player'].head(8).to_list()

            for player in player1:
                with st.expander(player):
                    st.subheader(player+' Game Stats')
                    showPlayerStatTables(dfI_Team, player)
                    showIndividualPlayerCharts(dfI_Team, player)
        with col2:
            team_players = data['Players']
            #team_players = team_players[team_players['Team']==HomeTeam]
            st.subheader(HomeTeam + ' Player Data')
            showPlayersTable(team_players,HomeTeam)
            dfI_Team = dfI[dfI['Team'] == HomeTeam]
            tp = team_players[team_players['Team'] == HomeTeam].sort_values('PRPG', ascending=False)
            player1 = tp['Player'].head(8).to_list()

            for player in player1:
                with st.expander(player):
                    st.subheader(player+' Game Stats')
                    showPlayerStatTables(dfI_Team, player)
                    showIndividualPlayerCharts(dfI_Team, player)
            
        try:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=2, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test1,order=2, ax=ax1, color = 'green')
        except:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=1, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test1,order=1, ax=ax1, color = 'green')
        try: 
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2, color = 'blue')
            fig4=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test2,order=2, ax=ax2, color = 'green')
        except:
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=1, ax=ax2, color = 'blue')
            fig4=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test2,order=1, ax=ax2, color = 'green')
        #plt.show(fig)
        #st.pyplot(fig)
        st.subheader('Polynomial Regression Charts')
        st.text('Daily Pomeroy Rankings line in green for each game')
        st.text('Polynomial Regression of actual game performance in blue for each game ')
        st.text('If the blue line is above the green then the team is playing better than its ranking ')
        st.pyplot(fig)
        st.subheader('Pomeroy Ranking and ATS Record')
        st.text('Pomeroy Rankings by game Line in Green')
        st.text('Blue bars are positive if the team won against the spread')
        GetTwoChartsTogether_EMA_2024(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","Pomeroy_Tm_AdjEM","Pomeroy_Tm_AdjEM","ATS")
        GetTwoChartsTogether_EMA_2024(test1,test2,AwayTeam,HomeTeam,"PlayingOverRating","PlayingOverRating","Pomeroy_Tm_AdjEM","Pomeroy_Tm_AdjEM","ATS")
        st.subheader('Team Playing Over its Ranking')
        st.text('Blue bars are positive if the team played over its rating')
        st.text('The green and blue lines are cumulative moving averages')
        #st.dataframe(test1)
        getOverplayingChartBothTeamsDec4(pp,test1,test2,AwayTeam,HomeTeam)
        st.subheader('Adjusted Offense and the ATS spread')
        GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Tm_AdjO","Pomeroy_Tm_AdjEM","ATS")
        st.subheader('Adjusted Defense against the Over/Under')
        GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Tm_AdjD","Pomeroy_Tm_AdjEM","OverUnder")
        st.subheader('Estimated Pace against the Over/Under')
        #GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Pace","PomTempo","OverUnder")
    
        st.subheader('Points per Possesion against the ATS')
        GetTwoTeamChartsTogether2024(test1,test2,AwayTeam,HomeTeam,"Tm_O_PPP","ATS")
        st.subheader('Defensive Points per Possesion against the Over/Under')
        GetTwoTeamChartsTogether2024(test1,test2,AwayTeam,HomeTeam,"Tm_D_PPP","OverUnder")
        #getDistributionMatchupChartsNew(AwayTeam,HomeTeam)
        #getDistributionMatchupCharts2024(AwayTeam,HomeTeam,test1,test2)
        displayTeamDistributionsMatchup(Gamesdf,AwayTeam,HomeTeam)

    #except:
        #st.write(' No games today')
    
    
def Team_Matchup(data):
    AwayTeamAll = [''] + data['AwayTeamAll']
    HomeTeamAll = [''] + data['HomeTeamAll']
    st.title('NCAA Head to Head Matchup')
    #AwayList=[''] + Dailyschedule['AWAY'].tolist()
    #HomeList=[''] + Dailyschedule['HOME'].tolist()
    AwayTeam = st.selectbox('Away Team',AwayTeamAll)
    HomeTeam = st.selectbox('Home Team',HomeTeamAll)
    Dailyschedule=pd.read_csv("Data/DailySchedules2024/SkedHistory.csv")
    Gamesdf = pd.read_csv("Data/DailySchedules2024/Gamesdf"+today_date_format+".csv")
    Gamesdf = Gamesdf.reset_index(drop=True)
    Gamesdf.drop(columns=Gamesdf.columns[0], axis=1,  inplace=True)
    Gamesdf = Gamesdf.drop_duplicates()
    if st.button('Run'):
        #dateforRankings=dateToday
        #dateforRankings5=d2
        #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
        TeamDatabase2.set_index("OldTRankName", inplace=True)
        from matplotlib.backends.backend_pdf import PdfPages
        season ='2024'
        WhichFile='TeamDataFiles'+season
        pp= PdfPages("Daily_Team_Charts_"+today_date_format+".pdf")

                
        st.header('Team Matchup')
        plt.style.use('seaborn')
        fig_dims = (15,10)
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=fig_dims)
        plt.figure(figsize=(8, 8))
        ax1.set_title(AwayTeam)
        ax2.set_title(HomeTeam)
            
        test1=get_team_info_from_gamesdf(Gamesdf,AwayTeam)
        #st.dataframe(test1)
        test1 = test1.reset_index(drop=True)
        #test1.drop(columns=test1.columns[0], axis=1,  inplace=True)
        #test1 = test1.drop_duplicates()
        test2=get_team_info_from_gamesdf(Gamesdf,HomeTeam)
        test2 = test2.reset_index(drop=True)
        #test2.drop(columns=test2.columns[0], axis=1,  inplace=True)
        #test2 = test2.drop_duplicates()
        test1['New_ID'] = range(0, 0+len(test1))
        test2['New_ID'] = range(0, 0+len(test2))
        myteams = [AwayTeam,HomeTeam]
        plot_line_chartLetsPlotHot(MG_Rank2, myteams)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(AwayTeam + ' Rankings')
            displayRankingHistory(data,AwayTeam)
            getTeamDFTable2024(test1,AwayTeam)
        
        with col2:
            st.subheader(HomeTeam + ' Rankings')
            displayRankingHistory(data,HomeTeam)
            getTeamDFTable2024(test2,HomeTeam)
        col1, col2 = st.columns(2)
        dfI =getIndividualPlayerData()
        
        with col1:
            team_players = data['Players']
            st.subheader(AwayTeam + ' Player Data')
            showPlayersTable(team_players, AwayTeam)
            dfI_Team = dfI[dfI['Team'] == AwayTeam]
            tp = team_players[team_players['Team'] == AwayTeam].sort_values('PRPG', ascending=False)
            player1 = tp['Player'].head(8).to_list()

            for player in player1:
                with st.expander(player):
                    st.subheader(player+' Game Stats')
                    showPlayerStatTables(dfI_Team, player)
                    showIndividualPlayerCharts(dfI_Team, player)
            showTeamLetsPlotMultiCharts2024(test1,'ATSvalue',"EMRating10GameExpMA", "EMRating3GameExpMA","Pomeroy_Tm_AdjEM","EMRating",'EMRating vs ATS',AwayTeam)  
            
            showTeamLetsPlotCharts2024(test1,'ATSvalue','AdjO3GameExpMA','AdjO10GameExpMA','Tm_AdjO','Adj Offense vs ATS',AwayTeam)
            showTeamLetsPlotCharts2024(test1,'OverUnder','AdjD3GameExpMA','AdjD10GameExpMA','Tm_AdjD','Adj Defense vs OverUnder',AwayTeam)
            showTeamLetsPlotCharts2024(test1,'ATSvalue','PPP_3GameExpMA','PPP_10GameExpMA','Tm_O_PPP','PPP  vs ATS',AwayTeam)
            showTeamLetsPlotCharts2024(test1,'OverUnder','PPP_D_3GameExpMA','PPP_D_10GameExpMA','Tm_D_PPP','PPP Defense vs OverUnder',AwayTeam)
            showTeamLetsPlotOverplayingCharts2024(test1,'ATSvalue',"DifCumSum", "DifCumSumEMA",'Overplaying vs ATS',AwayTeam)

        
            #team_players = data['Players']
            #team_players = team_players[team_players['Team']==AwayTeam]
            #st.subheader(AwayTeam + ' Player Data')
            #showPlayersTable(team_players,AwayTeam)
            #dfI_Team = dfI[dfI['Team']==AwayTeam]
            #tp = team_players[team_players['Team']==AwayTeam].sort_values('PRPG',ascending=False)
            #player1 = tp['Player'].head(5).to_list()
            #showIndividualPlayerCharts(dfI_Team,player1[0])
            #showPlayerStatTables(dfI_Team,player1[0])
        with col2:
            team_players = data['Players']
            st.subheader(HomeTeam + ' Player Data')
            showPlayersTable(team_players, HomeTeam)
            dfI_Team = dfI[dfI['Team'] == HomeTeam]
            tp = team_players[team_players['Team'] == HomeTeam].sort_values('PRPG', ascending=False)
            player1 = tp['Player'].head(8).to_list()

            for player in player1:
                with st.expander(player):
                    st.subheader(player+' Game Stats')
                    showPlayerStatTables(dfI_Team, player)
                    showIndividualPlayerCharts(dfI_Team, player)
            showTeamLetsPlotMultiCharts2024(test2,'ATSvalue',"EMRating10GameExpMA", "EMRating3GameExpMA","Pomeroy_Tm_AdjEM","EMRating",'EMRating vs ATS',HomeTeam)
            showTeamLetsPlotCharts2024(test2,'ATSvalue','AdjO3GameExpMA','AdjO10GameExpMA','Tm_AdjO','Adj Offense vs ATS',HomeTeam)
            showTeamLetsPlotCharts2024(test2,'OverUnder','AdjD3GameExpMA','AdjD10GameExpMA','Tm_AdjD','Adj Defense vs OverUnder',HomeTeam) 
            showTeamLetsPlotCharts2024(test2,'ATSvalue','PPP_3GameExpMA','PPP_10GameExpMA','Tm_O_PPP','PPP  vs ATS',HomeTeam)
            showTeamLetsPlotCharts2024(test2,'OverUnder','PPP_D_3GameExpMA','PPP_D_10GameExpMA','Tm_D_PPP','PPP Defense vs OverUnder',HomeTeam)
            showTeamLetsPlotOverplayingCharts2024(test2,'ATSvalue',"DifCumSum", "DifCumSumEMA",'Overplaying vs ATS',HomeTeam)
            #team_players = data['Players']
            #team_players = team_players[team_players['Team']==HomeTeam]
            #st.subheader(HomeTeam + ' Player Data')
            #showPlayersTable(team_players,HomeTeam)
            #dfI_Team = dfI[dfI['Team']==HomeTeam]
            #tp = team_players[team_players['Team']==HomeTeam].sort_values('PRPG',ascending=False)
            #player1 = tp['Player'].head(5).to_list()
            #showIndividualPlayerCharts(dfI_Team,player1[0])
            #showPlayerStatTables(dfI_Team,player1[0])
        try:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=2, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test1,order=2, ax=ax1, color = 'green')
        except:
            fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=1, ax=ax1, color = 'blue')
            fig2=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test1,order=1, ax=ax1, color = 'green')
        try: 
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2, color = 'blue')
            fig4=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test2,order=2, ax=ax2, color = 'green')
        except:
            fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=1, ax=ax2, color = 'blue')
            fig4=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test2,order=1, ax=ax2, color = 'green')
        #plt.show(fig)
        #st.pyplot(fig)
        st.subheader('Polynomial Regression Charts')
        st.text('Daily Pomeroy Rankings line in green for each game')
        st.text('Polynomial Regression of actual game performance in blue for each game ')
        st.text('If the blue line is above the green then the team is playing better than its ranking ')
        st.pyplot(fig)
        st.subheader('Pomeroy Ranking and ATS Record')
        st.text('Pomeroy Rankings by game Line in Green')
        st.text('Blue bars are positive if the team won against the spread')
        GetTwoChartsTogether_EMA_2024(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","Pomeroy_Tm_AdjEM","Pomeroy_Tm_AdjEM","ATS")
        GetTwoChartsTogether_EMA_2024(test1,test2,AwayTeam,HomeTeam,"PlayingOverRating","PlayingOverRating","Pomeroy_Tm_AdjEM","Pomeroy_Tm_AdjEM","ATS")
        st.subheader('Team Playing Over its Ranking')
        st.text('Blue bars are positive if the team played over its rating')
        st.text('The green and blue lines are cumulative moving averages')
        #st.dataframe(test1)
        getOverplayingChartBothTeamsDec4(pp,test1,test2,AwayTeam,HomeTeam)
        st.subheader('Adjusted Offense and the ATS spread')
        GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Tm_AdjO","Pomeroy_Tm_AdjEM","ATS")
        st.subheader('Adjusted Defense against the Over/Under')
        GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Tm_AdjD","Pomeroy_Tm_AdjEM","OverUnder")
        st.subheader('Estimated Pace against the Over/Under')
        #GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Pace","PomTempo","OverUnder")
    
        st.subheader('Points per Possesion against the ATS')
        GetTwoTeamChartsTogether2024(test1,test2,AwayTeam,HomeTeam,"Tm_O_PPP","ATS")
        st.subheader('Defensive Points per Possesion against the Over/Under')
        GetTwoTeamChartsTogether2024(test1,test2,AwayTeam,HomeTeam,"Tm_D_PPP","OverUnder")
        
        #showTeamLetsPlotCharts2024(test1,'ATSvalue','AdjD3GameExpMA','AdjD10GameExpMA','Tm_AdjD','Adj Defense vs ATS',AwayTeam)
        #getDistributionMatchupChartsNew(AwayTeam,HomeTeam)
        #getDistributionMatchupCharts2024(AwayTeam,HomeTeam,test1,test2)
        
def Past_Games(data):
    st.title('NCAA Head to Head Matchup')
    season = st.selectbox('Season Selection',['2024','2023'])
    if season == '2024':
        #st.write('2024')
        add_selectbox = st.header("Select Todays Date")
        add_selectbox_start =st.date_input('Pick date')
        dateString=str(add_selectbox_start)
        dateToday=dateString.replace('-', '')
        #st.write(dateToday)
        Gamesdf = pd.read_csv("Data/DailySchedules2024/Gamesdf"+dateToday+".csv")
        Gamesdf = Gamesdf.reset_index(drop=True)
        Gamesdf.drop(columns=Gamesdf.columns[0], axis=1,  inplace=True)
        Gamesdf = Gamesdf.drop_duplicates()
        Tables_Choice=st.selectbox('Sort Games By',['Alphabetical', 'Time','Regression_Difference','OverPlaying'],index=0)
        Dailyschedule=pd.read_csv("Data/DailySchedules2024/"+dateToday+"Schedule.csv")
        if 'Alphabetical'in  Tables_Choice:
            Dailyschedule=Dailyschedule.sort_values(by=['AWAY'])
        if 'Time' in Tables_Choice:
            Dailyschedule=Dailyschedule.sort_values(by=['commence_time'])   
        if 'Regression_Difference' in Tables_Choice: 
            Dailyschedule=Dailyschedule.sort_values(by=['Reg_dif'])
        if 'OverPlaying' in Tables_Choice: 
            Dailyschedule=Dailyschedule.sort_values(by=['Over_dif'])
        AwayList=[''] + Dailyschedule['AWAY'].tolist()
        HomeList=[''] + Dailyschedule['HOME'].tolist()
        AwayTeam = st.selectbox('Away Team',AwayList,index=0)
        HomeTeam = st.selectbox('Home Team',HomeList,index=0)
        st.header('Sortable NCAA Game Schedule')
        st.text('Games can be sorted by columns. Click on column header to sort')
        st.text('To sort by game time click the Time column.  ')
        st.text('Low Negative values in the Reg Dif and Overplaying column mean the Home team is the pick  ') 
        Dailyschedule = Dailyschedule[['AWAY','HOME','HomeAway','FanDuel','MG_ATS_PointDiff','commence_time','Reg_dif','Over_dif','Dif_from_Vegas','Pomeroy_PointDiff','TRank_PointDiff','MG_PointDiff','Daily_Reg_PointDiff','DraftKings','BetMGM spreads','Caesars spreads','BetRivers spreads','VegasTotal']]
        Dailyschedule.DraftKings = Dailyschedule.DraftKings.astype(float).round(1)
        Dailyschedule.VegasTotal = Dailyschedule.VegasTotal.astype(float).round(1)
        Dailyschedule['commence_time'] = pd.to_datetime(Dailyschedule['commence_time'])
        # Convert to US Central time
        Dailyschedule['commence_time'] = Dailyschedule['commence_time'].dt.tz_convert('US/Central')
        # Format time to display like 11:00AM, 2:00PM, etc.
        Dailyschedule['commence_time'] = Dailyschedule['commence_time'].dt.strftime('%I:%M%p')
        allcols=Dailyschedule.columns
        gb = GridOptionsBuilder.from_dataframe(Dailyschedule,groupable=True)
        gb.configure_columns(allcols, cellStyle=cellStyle)
        csTotal=cellStyleDynamic(Dailyschedule.Reg_dif)
        gb.configure_column('Reg_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
        csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
        gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_column('DraftKings',valueFormatter=numberFormat(1))
        gb.configure_column('VegasTotal',valueFormatter=numberFormat(1))
        gb.configure_column('Pomeroy_PointDiff',valueFormatter=numberFormat(1))
        gb.configure_column('TRank_PointDiff',valueFormatter=numberFormat(1))
        gb.configure_column('MG_PointDiff',valueFormatter=numberFormat(1))
        gb.configure_column('MG_ATS_PointDiff',valueFormatter=numberFormat(1))
        gb.configure_column('Daily_Reg_PointDiff',valueFormatter=numberFormat(1))
        gb.configure_column('Dif_from_Vegas',cellStyle=csTotal,valueFormatter=numberFormat(2))
        #gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        #gridOptions = gb.build()
        opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
        gb.configure_grid_options(**opts)
        keyname='Test     '
        g = _displayGrid(Dailyschedule, gb, key=keyname, height=800)
        #AgGrid(Dailyschedule, gridOptions=gridOptions, enable_enterprise_modules=True,allow_unsafe_jscode=True,height=800)
    
        if st.button('Run'):
        

        
            dateforRankings=today_date_format
            #dateforRankings5=d2
            #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
            TeamDatabase2.set_index("OldTRankName", inplace=True)
            #MG_DF1=pd.read_csv("Data/MGRankings"+season+"/tm_seasons_stats_ranks"+dateforRankings5+" .csv")
            #MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
            #MG_DF1.set_index("updated", inplace=True)
            from matplotlib.backends.backend_pdf import PdfPages
            #WhichFile='TeamDataFiles'+season
            pp= PdfPages("Daily_Team_Charts_"+dateforRankings+".pdf")     
            st.header('Team Matchup')
            plt.style.use('seaborn')
            fig_dims = (15,10)
            plt.figure(figsize=(10,8))
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=fig_dims)
            plt.figure(figsize=(20, 12))
            ax1.set_title(AwayTeam)
            ax2.set_title(HomeTeam)  
            test1=get_team_info_from_gamesdf(Gamesdf,AwayTeam)
            #st.dataframe(test1)
            test1 = test1.reset_index(drop=True)
            #test1.drop(columns=test1.columns[0], axis=1,  inplace=True)
            #test1 = test1.drop_duplicates()
            test2=get_team_info_from_gamesdf(Gamesdf,HomeTeam)
            test2 = test2.reset_index(drop=True)
            #test2.drop(columns=test2.columns[0], axis=1,  inplace=True)
            #test2 = test2.drop_duplicates()
            test1['New_ID'] = range(0, 0+len(test1))
            test2['New_ID'] = range(0, 0+len(test2))
            myteams = [AwayTeam,HomeTeam]
            plot_line_chartLetsPlotHot(MG_Rank2, myteams)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(AwayTeam + ' Rankings')
                displayRankingHistory(data,AwayTeam)
            with col2:
                st.subheader(HomeTeam + ' Rankings')
                displayRankingHistory(data,HomeTeam)
            col1, col2 = st.columns(2)
            with col1:
                team_players = data['Players']
                #team_players = team_players[team_players['Team']==AwayTeam]
                st.subheader(AwayTeam + ' Player Data')
                showPlayersTable(team_players,AwayTeam)
            with col2:
                team_players = data['Players']
                #team_players = team_players[team_players['Team']==HomeTeam]
                st.subheader(HomeTeam + ' Player Data')
                showPlayersTable(team_players,HomeTeam)
            
            try:
                fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=2, ax=ax1, color = 'blue')
                fig2=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test1,order=2, ax=ax1, color = 'green')
            except:
                fig1=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test1,order=1, ax=ax1, color = 'blue')
                fig2=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test1,order=1, ax=ax1, color = 'green')
            try: 
                fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=2, ax=ax2, color = 'blue')
                fig4=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test2,order=2, ax=ax2, color = 'green')
            except:
                fig3=sns.regplot(x="New_ID", y="EMRating5GameExpMA", data=test2,order=1, ax=ax2, color = 'blue')
                fig4=sns.regplot(x='New_ID', y='Pomeroy_Tm_AdjEM', data=test2,order=1, ax=ax2, color = 'green')
            #plt.show(fig)
            st.pyplot(fig)
            st.subheader('Polynomial Regression Charts')
            st.text('Daily Pomeroy Rankings line in green for each game')
            st.text('Polynomial Regression of actual game performance in blue for each game ')
            st.text('If the blue line is above the green then the team is playing better than its ranking ')
            st.pyplot(fig)
            st.subheader('Pomeroy Ranking and ATS Record')
            st.text('Pomeroy Rankings by game Line in Green')
            st.text('Blue bars are positive if the team won against the spread')
            GetTwoChartsTogether_EMA_2024(test1,test2,AwayTeam,HomeTeam,"EMRating","EMRating","Pomeroy_Tm_AdjEM","Pomeroy_Tm_AdjEM","ATS")
            GetTwoChartsTogether_EMA_2024(test1,test2,AwayTeam,HomeTeam,"PlayingOverRating","PlayingOverRating","Pomeroy_Tm_AdjEM","Pomeroy_Tm_AdjEM","ATS")
            st.subheader('Team Playing Over its Ranking')
            st.text('Blue bars are positive if the team played over its rating')
            st.text('The green and blue lines are cumulative moving averages')
            #st.dataframe(test1)
            getOverplayingChartBothTeamsDec4(pp,test1,test2,AwayTeam,HomeTeam)
            st.subheader('Adjusted Offense and the ATS spread')
            GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Tm_AdjO","Pomeroy_Tm_AdjEM","ATS")
            st.subheader('Adjusted Defense against the Over/Under')
            GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Tm_AdjD","Pomeroy_Tm_AdjEM","OverUnder")
            st.subheader('Estimated Pace against the Over/Under')
            #GetTwoTeamChartsTogetherDec6(pp,test1,test2,AwayTeam,HomeTeam,"Pace","PomTempo","OverUnder")
    
            st.subheader('Points per Possesion against the ATS')
            GetTwoTeamChartsTogether2024(test1,test2,AwayTeam,HomeTeam,"Tm_O_PPP","ATS")
            st.subheader('Defensive Points per Possesion against the Over/Under')
            GetTwoTeamChartsTogether2024(test1,test2,AwayTeam,HomeTeam,"Tm_D_PPP","OverUnder")
            #getDistributionMatchupChartsNew(AwayTeam,HomeTeam)
            #getDistributionMatchupCharts2024(AwayTeam,HomeTeam,test1,test2)
            displayTeamDistributionsMatchup(Gamesdf,AwayTeam,HomeTeam)
            getTeamDFTable2024(test1,AwayTeam)
            getTeamDFTable2024(test2,HomeTeam)
        
        
def Team_Page(data):
    st.subheader('NCAA Mens Basketball Team Pages')
    team_selected = st.selectbox('Select a Team',data['teams']) 
    test1=get_team_info_from_gamesdf(data['Gamesdf'],team_selected)
    test1 = test1.reset_index(drop=True)
    col1, col2 = st.columns(2)
    with col1:
        displayRankingHistory(data,team_selected)
    with col2:
        team_players = data['Players']
        #team_players = team_players[team_players['Team']==team_selected]
        st.subheader(team_selected + ' Player Data')
        showPlayersTable(team_players,team_selected)
    dfI =getIndividualPlayerData()
    dfI_Team = dfI[dfI['Team'] == team_selected]
    tp = team_players[team_players['Team'] == team_selected].sort_values('PRPG', ascending=False)
    player1 = tp['Player'].head(8).to_list()

    for player in player1:
        with st.expander(player):
            st.subheader(player+' Game Stats')
            showPlayerStatTables(dfI_Team, player)
            showIndividualPlayerCharts(dfI_Team, player)
    st.subheader(team_selected + ' Schedule/Results Data')
    test1 = test1[['Date','Tm','Opp','HomeAway','Result_x','EMRating','PlayingOverRating'	,'ATSvalue','Tempo','Lead','AvgLead','Tm_AdjO','Tm_AdjD','G-Score']]

    allcols=test1.columns
    gb = GridOptionsBuilder.from_dataframe(test1,groupable=True)
    gb.configure_columns(allcols, cellStyle=cellStyle)
    csTotal=cellStyleDynamic(test1.EMRating)
    gb.configure_column('EMRating',cellStyle=csTotal,valueFormatter=numberFormat(2))
    csTotal=cellStyleDynamic(test1.ATSvalue)
    gb.configure_column('ATSvalue',cellStyle=csTotal,valueFormatter=numberFormat(2))
    gb.configure_column('PlayingOverRating',valueFormatter=numberFormat(2))
    #gb.configure_column('OBPM',valueFormatter=numberFormat(2))
    #gb.configure_column('DBPM',valueFormatter=numberFormat(2))
    #gb.configure_column('USAGE',valueFormatter=numberFormat(1))
    #gb.configure_column('Points',valueFormatter=numberFormat(2))
    #gb.configure_column('EFG',valueFormatter=numberFormat(2))
    #gb.configure_column('OR',valueFormatter=numberFormat(2))
    #gb.configure_column('3PT%',cellStyle=csTotal,valueFormatter=numberFormat(2))
    #gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    #gridOptions = gb.build()
    opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
    gb.configure_grid_options(**opts)
    keyname='Team D'+team_selected
    g = _displayGrid(test1, gb, key=keyname, height=600)
    displayTeamDistributions(data['Gamesdf'],team_selected)

    

    allcols=team_players.columns
    gb = GridOptionsBuilder.from_dataframe(team_players,groupable=True)
    gb.configure_columns(allcols, cellStyle=cellStyle)
    csTotal=cellStyleDynamic(team_players.PRPG)
    gb.configure_column('PRPG',cellStyle=csTotal,valueFormatter=numberFormat(2))
    csTotal=cellStyleDynamic(team_players.ORTG)
    gb.configure_column('ORTG',cellStyle=csTotal,valueFormatter=numberFormat(2))
    gb.configure_column('BPM',valueFormatter=numberFormat(2))
    gb.configure_column('OBPM',valueFormatter=numberFormat(2))
    gb.configure_column('DBPM',valueFormatter=numberFormat(2))
    gb.configure_column('USAGE',valueFormatter=numberFormat(1))
    gb.configure_column('Points',valueFormatter=numberFormat(2))
    gb.configure_column('EFG',valueFormatter=numberFormat(2))
    gb.configure_column('OR',valueFormatter=numberFormat(2))
    gb.configure_column('3PT%',cellStyle=csTotal,valueFormatter=numberFormat(2))
    #gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    #gridOptions = gb.build()
    opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
    gb.configure_grid_options(**opts)
    keyname='Team P'+team_selected
    g = _displayGrid(team_players, gb, key=keyname, height=600)
    dfI =getIndividualPlayerData()
    
    #showIndividualPlayerCharts(dfI,'Zach Edey')
    #showPlayerStatTables(dfI,'Zach Edey')
def Betting_Performance_Page(data):
    st.title('Betting Performance')
    df = data['SkedBetting']
    bcols = ['Tm','AWAY','HOME','Date_zero','Reg_dif','Over_dif','Daily_Reg_Tm_net_eff','Daily_Reg_Opp_net_eff','MG_ATS_PointDiff','ATSVegas','Pomeroy_PointDiffWinATS',
     'Pomeroy_PointDiffLossATS','Pomeroy_OverUnderWinTotal','Pomeroy_OverUnderLossTotal','TRank_PointDiffWinATS',
     'TRank_PointDiffLossATS','TRank_OverUnderWinTotal','TRank_OverUnderLossTotal','MG_PointDiffWinATS','MG_PointDiffLossATS',
     'MG_OverUnderWinTotal','MG_OverUnderLossTotal','MG_ATS_PointDiffWinATS','MG_ATS_PointDiffLossATS','Daily_Reg_PointDiffWinATS',
     'Daily_Reg_PointDiffLossATS','VegasImpliedWinPercent','MG_ATS_ImpliedWinPercent','Dif_from_Vegas','MG_Reg_ATS_ImpliedWinPercent','Reg_Dif_Abs','Over_Dif_Abs']
    df = df[bcols]
    gb = GridOptionsBuilder.from_dataframe(df,groupable=True)
    #csTotal=cellStyleDynamic(hot2.performance_change)
    #gb.configure_column('performance_change',cellStyle=csTotal,valueFormatter=numberFormat(1))
    #gb.configure_column('Seed',valueFormatter=numberFormat(0))
    #csTotal=cellStyleDynamic(Dailyschedule.Over_dif)
    #gb.configure_column('Over_dif',cellStyle=csTotal,valueFormatter=numberFormat(1))
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
    gb.configure_grid_options(**opts)
    keyname='Test bet'
    g = _displayGrid(df, gb, key=keyname, height=600)
    card("This works", "I can insert text inside a card")

    br(2)
    #st.markdown("<div class="alert alert-success">Example text highlighted in green background.</div>")
    #else:
    #    add_selectbox = st.sidebar.header("Select Todays Date")
    #    add_selectbox_start =st.sidebar.date_input('Pick date')
    #    dateString=str(add_selectbox_start)
    #    dateToday=dateString.replace('-', '')
    #    #Dailyschedule=pd.read_csv("DailySchedules2023/"+dateToday+"Schedule.csv")
    #    Dailyschedule=pd.read_csv("Data/DailySchedules2023/"+dateToday+"Schedule.csv")
    #    d2=dateString.split('-')[1]+'_'+dateString.split('-')[2]+'_'+dateString.split('-')[0]
    #    themonth=int(dateString.split('-')[1])
    #    theday=int(dateString.split('-')[2])
    #    theyear=dateString.split('-')[0]
    #    get2023Display(Dailyschedule,dateToday,d2,season)
def read_csv_from_url(url):
    df = pd.read_csv(url,sep=',',  header=None)
    return df
def displayRankingHistory(data,myteam):
    MGG = data['MG_Rank2']
    TR1 = data['TRank']
    MGG['Date_zero'] = pd.to_datetime(MGG['Date_zero'])
    TR1['Date_zero'] = pd.to_datetime(TR1['Date_zero'])
    MGG = MGG[['Team', 'ATS_net_eff', 'MG_net_eff', 'Date_zero']]
    TR1 = TR1[['Team', 'AdjEM', 'Date_zero']].rename(columns={'AdjEM': 'TRank_AdjEM'})
    df = pd.merge(MGG, TR1, on=['Team', 'Date_zero'], how='outer')
    df1=df[df['Team']==myteam].sort_values('Date_zero')[2:]
    melted_df = df1.melt(id_vars=['Team', 'Date_zero'], value_vars=['ATS_net_eff', 'MG_net_eff', 'TRank_AdjEM'], var_name='Ranking Type', value_name='Rankings')

    p = ggplot(melted_df, aes(x='Date_zero', y='Rankings', group='Ranking Type')) + geom_line(aes(color='Ranking Type'), size=1, alpha=0.5)+ggtitle("Ranking Comparison") + ggsize(800, 600)
    st_letsplot(p)
def getPomeroyDict():
    
    df = pd.read_csv('Data/Pomeroy_2024_DB.csv')
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])
    df = df[['Rk','Team','AdjEM','AdjO','AdjD','AdjT','Date_zero']]
    df= df.rename(columns={
        'AdjO': 'AdjOE',
        'AdjD': 'AdjDE',
        'AdjT': 'pace' 

    })
    # Filter the dataframe for the latest date
    latest_df = df[df['Date_zero'] == df['Date_zero'].max()]
    latest_df = latest_df.drop_duplicates()
    grouped_df = latest_df.groupby('Team').mean()
    # Create a nested dictionary from the dataframe
    PomeroyDict= grouped_df.to_dict('index')
    return(PomeroyDict)

def getTRankDict():
    
    df = pd.read_csv('Data/TRank_2024_DB.csv')
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])
    df = df[['Team','AdjOE','AdjDE','BARTHAG','AdjEM','Date_zero','ADJ. T']]
    df= df.rename(columns={
        'ADJ. T': 'pace'  
    })
    
    latest_df = df[df['Date_zero'] == df['Date_zero'].max()]
    latest_df = latest_df.drop_duplicates()
    grouped_df = latest_df.groupby('Team').mean()
    # Create a nested dictionary from the dataframe
    TRankDict= grouped_df.to_dict('index')

    return(TRankDict)

def getMGRatingsDict():
    
    df = pd.read_csv('Data/MGRatings2024_Daily_New_DB.csv')
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])

    df =df[['Team','ATS_net_eff','MG_net_eff','mod_AdjO','mod_AdjD','Date_zero','pace']]
    df= df.rename(columns={
        'mod_AdjO': 'AdjOE',
        'mod_AdjD': 'AdjDE',
        'MG_net_eff': 'AdjEM'

    })
    latest_df = df[df['Date_zero'] == df['Date_zero'].max()]
    latest_df = latest_df.drop_duplicates()
    grouped_df = latest_df.groupby('Team').mean()
# Create a nested dictionary from the dataframe
    MGDict= grouped_df.to_dict('index')
    return(MGDict)

def setStrength(latest_df):
    lf = latest_df[['Team','AdjEM']]
    lf.set_index('Team', inplace=True)
    strength=lf['AdjEM'].to_dict()
    return(strength)

st.set_page_config(page_title="MG Rankings",layout="wide")




_MENU_STYLE = {
    'container': {
        'padding': '4px!important', 
        'background-color': '#fafafa"',
    },
    'nav-link': {
        '--hover-color': '#dfdfdf',
        
    },
    'nav-link-selected': {
        'background-color': '#131414',
        'font-weight': '600'
    },
}

_CHOICES = {
    'MG Rankings': dict(func=MG_Rankings, icon='play-fill'),
    'Todays Games': dict(func=Todays_Games, icon='play-fill'),
    'Team Matchup': dict(func=Team_Matchup, icon='play-fill'),
    'Past Games': dict(func=Past_Games, icon='play-fill'),
    'Rankings Historical Charts': dict(func=Historical_Rankings_Page, icon='play-fill'),
    'Bracketology Page': dict(func=Bracketology_Page, icon='play-fill'),
    'Team Pages': dict(func=Team_Page, icon='play-fill'),
    'Betting Performance': dict(func=Betting_Performance_Page, icon='play-fill'),
   
}

_MENU_ITEMS = list(_CHOICES.keys())
_ICONS = [d.get('icon', 'database') for d in _CHOICES.values()]
maketable = Stats.maketable
#deltaU = energy_of_flipping
LeagueTempo=69.1
LeagueOE=104.6
SimulationResults = namedtuple('SimulationResults','brackets unique_brackets lowest_bracket lowest_bracket_count most_common_bracket most_common_bracket_count')
regions = ['south', 'east', 'midwest', 'west']
seed_region = {i: 0 for i in range(1, 17)}
default_energy_function = None
MoneyLine=pd.read_csv("Data/MoneyLineConversion.csv")
PomDict = {}
TRDict = {}
MGDict = {}
PomDict = getPomeroyDict()
TRDict = getTRankDict()
MGDict = getMGRatingsDict()

dfT = pd.DataFrame.from_dict(TRDict, orient='index')
dfT.reset_index(inplace=True)
# Rename the index column to 'Team'
dfT.rename(columns={'index': 'Team'}, inplace=True)
strength = setStrength(dfT)

set_energy_function(default_energy_game)
#set_energy_function = set_energy_function
#set_energy_function(My_energy_game)
kenpom = {}
deltaU = energy_of_flipping

    
BM = getBracketMatrixDataframe()
TBracket = getTRankBracket()
TBracket1 = TBracket[['Seed','east','midwest','south','west']]
BM1 = BM[['Seed','east','midwest','south','west']]
newsouth=list(TBracket1["south"])
neweast=list(TBracket1["east"])
newmidwest=list(TBracket1["midwest"])
newwest=list(TBracket1["west"])

lineparts = ["Rank","Team","Conf","W-L","AdjEM","AdjO","AdjO-Rank","AdjD","AdjD-Rank","AdjT","AdjT-Rank","Luck","Luck-Rank",
             "SOSPyth","SOSPyth-Rank","SOSOppO","SOSOppO-Rank","SOSOppD","SOSOppD-Rank","NCOSPyth","NCOSPyth-Rank"]
textparts = ["Team","Conf","W-L"]
kpomdata = {}
teamsdict = {}
teams={}
teams['midwest'] =newmidwest
teams['south'] = newsouth
teams['east'] = neweast
teams['west'] = newwest
teamsdict['midwest'] = newmidwest
teamsdict['south'] =newsouth
teamsdict['east'] = neweast
teamsdict['west'] = newwest    
teamsdict['SweetSixteen']=list(BM1["west"])[0:4]+list(BM1["east"])[0:4]+list(BM1["midwest"])[0:4]+list(BM1["south"])[0:4]
teamsdict['EliteEight']=list(BM1["west"])[0:2]+list(BM1["east"])[0:2]+list(BM1["midwest"])[0:2]+list(BM1["south"])[0:2]
teamsdict['FinalFour']=list(BM1["west"])[0:1]+list(BM1["east"])[0:1]+list(BM1["midwest"])[0:1]+list(BM1["south"])[0:1]

# These are all listed in the same order:
_rankings = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
regional_rankings = {}
#regional_rankings = regional_rankings
for region in teamsdict:
    for (team,rank) in zip(teamsdict[region],_rankings):
    # We use a random number here so that the south's number 2
    # seed won't come out exactly the same rank as the west's.
        regional_rankings[team] = rank + random()/10

regions = {}
for region in teamsdict:
    for team in teamsdict[region]:
        regions[team] = region

all_teams = teamsdict['midwest'] + teamsdict['south'] + teamsdict['west'] + teamsdict['east']
teamsdict['all'] = all_teams
regional_rankings = regional_rankings 
SimulationResults = namedtuple('SimulationResults','brackets unique_brackets lowest_bracket lowest_bracket_count most_common_bracket most_common_bracket_count')
    
    

teams['SweetSixteen'] = list(BM1["west"])[0:4]+list(BM1["east"])[0:4]+list(BM1["midwest"])[0:4]+list(BM1["south"])[0:4]
teams['EliteEight'] = ['San Diego St.','Creighton','FAU','Kansas St.','Miami FL','Texas','Connecticut','Gonzaga']
teams['FinalFour'] = list(BM1["west"])[0:1]+list(BM1["east"])[0:1]+list(BM1["midwest"])[0:1]+list(BM1["south"])[0:1]
#all_teams = teams['midwest'] + teams['south'] + teams['west'] + teams['east']+teams['SweetSixteen']
all_teams = teams['midwest'] + teams['south'] + teams['west'] + teams['east']
#MoneyLine=pd.read_csv("C:/Users/mpgen/MoneyLineConversion.csv")
    
        
#st.write(myranks)
MYRANKS = TRDict
results = runbracket1(teamsdict,ntrials=1000,T=.2)
#st.write(str(results['all'][0][0]))
j=maketabletest(results)
allrounds = ['1st Round','2nd Round','3rd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
allrounds = ['Make','2nd Round','Sweet 16','Elite 8','Final 4','Championship','Win']
headers = ['Team'] + ['Region','Rank'] + allrounds+['Odds']

#st.dataframe(pd.DataFrame(j, columns=headers))
#l = makehtmltable(j, headers=headers)
#l=HTML(makehtmltable(j, headers=headers))
#st.write(l)
data={}
MYRANKS = MGDict
results = runbracket1(teamsdict,ntrials=1000,T=.2)
#st.write(str(results['all'][0][0]))
j1=maketabletest(results)
#st.dataframe(pd.DataFrame(j1, columns=headers))
data['TSim'] = pd.DataFrame(j, columns=headers)
data['MSim'] = pd.DataFrame(j1, columns=headers)
data['BM1'] = BM1
data['TBracket'] = TBracket1
TeamDatabase2=pd.read_csv("Data/TeamDatabase2023.csv")

player_data = read_csv_from_url('http://barttorvik.com/getadvstats.php?year=2024&csv=1')
myc = ['Player','Team','Conference','Games','Min%','ORTG','USAGE','EFG','TS','OR','DR','Assists','TO','FT made','FT Att','FT%','far 2 made','far 2 att','far 2 pct','3pts made','3pts att','3PT%','Blocks','STL','FTR','Year','Height','Number','PRPG','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','BPM','OBPM','DBPM', '3' ,' 4',' 5', 'Rebounds', 'Assists1' ,' 6 ', '7 ' ,'Points','Position','j']
player_data.columns=myc
player_data1 = player_data[['Number','Player','Team','Games','Min%','ORTG','BPM','OBPM','DBPM', 'PRPG','USAGE','Height','Year','Points','Position','EFG','TS','OR','DR','Assists','TO','FT made','FT Att','FT%','far 2 made','far 2 att','far 2 pct','3pts made','3pts att','3PT%','Blocks','STL','FTR','Rebounds', 'Assists1' ]]

#st.dataframe(player_data)
data['Players'] = player_data1
AllGames=pd.read_csv("Data/Season_GamesAll.csv")
AwayTeamAll=list(TeamDatabase2['OldTRankName'])
HomeTeamAll=list(TeamDatabase2['OldTRankName'])
MG_Rank=pd.read_csv("Data/MGRatings2024_Daily_All_DB.csv")
MG_Rank2=pd.read_csv("Data/MGRatings2024_Daily_New_DB.csv")
SkedBetting = pd.read_csv("Data/SkedBetting.csv")
TR = pd.read_csv('Data/TRank_2024_DB.csv')
hot,cold=  getHotColdTeams(MG_Rank2)
hotlist = hot.head(10)['Team'].to_list()
coldlist = cold.head(10)['Team'].to_list()
teams = MG_Rank['Tm_'].unique()
#st.title('NCAA Head to Head Matchup')
#page = st.sidebar.selectbox('Select page',['MG Rankings','Todays Games','Team Matchup','Past Games','Rankings Historical Charts','Bracketology Futures'])
TeamDatabase2=pd.read_csv("Data/TeamDatabase2024T.csv")
AllGames=pd.read_csv("Data/Season_GamesAll_2024.csv")
AwayTeamAll=list(TeamDatabase2['OldTRankName'])
HomeTeamAll=list(TeamDatabase2['OldTRankName'])
today_date_format,yesterday_date_format = getTodaysDateFormat()
try:
    Gamesdf = pd.read_csv("Data/DailySchedules2024/Gamesdf"+today_date_format+".csv")
except:
    Gamesdf = pd.read_csv("Data/DailySchedules2024/Gamesdf"+yesterday_date_format+".csv")
Gamesdf = Gamesdf.reset_index(drop=True)
Gamesdf.drop(columns=Gamesdf.columns[0], axis=1,  inplace=True)
Gamesdf = Gamesdf.drop_duplicates()

data['TRank'] = TR
data['SkedBetting'] = SkedBetting
data['TeamDatabase2']=TeamDatabase2
data['Gamesdf'] = Gamesdf
data['hot'] = hot
data['cold']= cold
data['hotlist']= hotlist
data['coldlist'] = coldlist
data['teams']= teams
data['MG_Rank'] = MG_Rank
data['MG_Rank2'] = MG_Rank2
data['today_date_format'] = today_date_format
#data['Dailyschedule'] = Dailyschedule
data['AwayTeamAll'] = AwayTeamAll
data['HomeTeamAll'] = HomeTeamAll

with st.sidebar:
    choice = option_menu(
            None,
            _MENU_ITEMS, 
            default_index=0,
            styles=_MENU_STYLE,
            icons=_ICONS,
            key='credit_dash_choice'
        )
if choice in _CHOICES:
    func = _CHOICES[choice]['func']
    func(data)
else:
    st.error(f'Unknown choice: {choice}')
#if page == 'Bracketology Futures':
#   Bracketology_Page() 
#if page == 'Rankings Historical Charts':
#    Historical_Rankings_Page(MG_Rank)
#if page == 'MG Rankings':
#    MG_Rankings(hot,cold,MG_Rank2,coldlist)
#if page == 'Todays Games':
#    Todays_Games(today_date_format,Dailyschedule)
#if page == 'Team Matchup':
#    Team_Matchup(AwayTeamAll,HomeTeamAll)
  
#if page == 'Past Games':
#    Past_Games() 
    
