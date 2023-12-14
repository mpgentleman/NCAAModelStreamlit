from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
#from bqplot import pyplot as plt
from matplotlib import pyplot as plt
from IPython.display import HTML
from numpy.random import random

from collections import OrderedDict

import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

from numpy import exp, array, zeros, ones, convolve
import scipy
from time import sleep
from copy import deepcopy
from collections import Counter, OrderedDict
from bs4 import BeautifulSoup
import requests
import pylab as pl
from random import choice,shuffle
#from urllib import urlopen
import pandas as pd
from numpy import exp,array,zeros,inf

from collections import Counter, OrderedDict, defaultdict

import os, os.path

from functools import wraps
from time import time

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

from collections import namedtuple
import json




class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)
    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__
    def __get__(self, obj, objtype):
        """Support instance methods."""
        fn = functools.partial(self.__call__, obj)
        fn.reset = self._reset
        fn.getcache = self._get_cache
        fn.setcache = self._set_cache
        fn.cachedict = self._get_cache_dict
        return fn
        
    def _reset(self):
        self.cache = {}
    def _get_cache(self,key):
        return self.cache[key]
    def _set_cache(self,key,value):
        self.cache[key] = value
    def _get_cache_dict(self):
        return self.cache



def movingaverage(interval, window_size):
    window= ones(int(window_size))/float(window_size)
    return convolve(interval, window, 'same')

def plotone(brackets, label, subplot1, subplot2, values=None, label2=None, 
            values2=None, description=None, useavg=False):
    """
    Plotting too many points causes lots of trouble for matplotlib. At
    the moment, we deal with that by plotting at most 50000 points,
    skipping evenly through the data if needed.
    """
    maxpts = 50000

    ntrials = len(brackets)
    if values is None:
        try:
            values = [getattr(b,label)() for b in brackets]
        except TypeError:
            values = [getattr(b,label) for b in brackets]

    if len(values) >= 50000:
        step = divmod(len(values),maxpts)[0]
    else:
        step = 1
    plt.subplot(subplot1)
    plt.plot(range(0,ntrials,step),values[::step],'.',label=label)
    if useavg:
        # want something like 2000 windows
        if step > 1:
            npts = maxpts
        else:
            npts = len(values)
        avgstep = divmod(len(values),int(npts/25))[0]
        
        plt.plot(range(0,ntrials,step),movingaverage(values[::step],avgstep),'-',label='avg. '+label)
            
    plt.ylabel(label.capitalize())
    plt.xlabel('Game')
    if description is not None:
        plt.title('%s over the trajectory, T=%s, %s'%(label.capitalize(),
                                                     brackets[0].T,description))
    else:
        plt.title('%s over the trajectory, T=%s'%(label.capitalize(),
                                                 brackets[0].T))
    #plt.legend()
    plt.subplot(subplot2)
    if values2 is None:
        if ntrials > 1000:
            nbins = min(int(ntrials/100),200)
        else:
            nbins = 10
        plt.hist(values,bins=nbins)
        plt.title('%s distribution, T=%s'%(label.capitalize(), brackets[0].T))
    else:
        plt.subplot(subplot2)
        plt.plot(range(0,ntrials,step),values2[::step],'.',label=label2)
        plt.ylabel(label2.capitalize())
        plt.xlabel('Game')
        plt.title('%s over the trajectory, T=%s'%(label2.capitalize(),
                                                 brackets[0].T))

    
def showstats(sr,newfig=False,description='MMMC',figsize=(15,8)):
    if newfig:
        plt.figure(figsize=figsize)
    plotone(sr.brackets, 'energy', 231, 234, description=description)
    plotone(sr.brackets, 'upsets', 232, 235, description=description)
    plotone(sr.brackets, 'Unique brackets', 233, 236, values=sr.unique_brackets,
            label2="Lowest Energy Sightings", values2=sr.lowest_bracket_count,
            description=description)
    plt.show()

    
def winpct8(team8,team9,T,numtrials=10000):
    results = [playgame(team8,team9,T)[0] == team8 for i in range(numtrials)]
    return np.average(results)
def plotwins(team8,team9,numtrials=10000):
    Ts = np.linspace(0,.5,100)
    pct = [winpct8(team8,team9,T,numtrials) for T in Ts]
    plt.plot(Ts,pct,label='{t1} vs. {t2}'.format(t1=team8,t2=team9))
    plt.xlabel('T')
    plt.ylabel('winpct')  
def winpct8CDF(team8,team9,T,numtrials=10000):
    results = [playgameCDF(team8,team9,T)[0] == team8 for i in range(numtrials)]
    return np.average(results)
def plotwinsCDF(team8,team9,numtrials=10000):
    Ts = np.linspace(0,3,100)
    pct = [winpct8CDF(team8,team9,T,numtrials) for T in Ts]
    plt.plot(Ts,pct,label='{t1} vs. {t2}'.format(t1=team8,t2=team9))
    plt.xlabel('T')
    plt.ylabel('winpct')  
    



def NewgetGamePredictionNeutralCourt(Team1AdjOff,Team1AdjDef,Team1AdjTempo,Team2AdjOff,Team2AdjDef,Team2AdjTempo,LeagueTempo,LeagueOE):
  
    GameTempo=(Team1AdjTempo/LeagueTempo*Team2AdjTempo/LeagueTempo)*LeagueTempo

    Team1Score=(Team1AdjOff/LeagueOE*Team2AdjDef)*GameTempo/100
    Team2Score=(Team2AdjOff/LeagueOE*Team1AdjDef)*GameTempo/100
    OverUnder=Team1Score+Team2Score
    PointDiff=Team1Score-Team2Score
    return PointDiff

 
def playgameCDF(team1, team2, T):
    """There's a difference between flipping a game in an existing
    bracket, and playing a game from scratch. If we're going to just
    use Boltzmann statistics to play a game from scratch, we can make
    life easy by using the Boltzmann factor to directly pick a
    winner.
    """
    ediff = deltaU(team1, team2)
    boltzmann_factor = exp(-ediff/T)
    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(kpomdata[team1]["AdjO"],kpomdata[team1]["AdjD"],kpomdata[team1]["AdjT"],kpomdata[team2]["AdjO"],kpomdata[team2]["AdjD"],kpomdata[team2]["AdjT"],LeagueTempo,LeagueOE)
    win_prob =scipy.stats.norm(0,10.5).cdf(PHomeTeamSpread)
    #win_prob = boltzmann_factor/(1+boltzmann_factor) if boltzmann_factor < inf else 1
    # So, prob of team 1 winning is then boltzmann_factor/(1+boltzmann_factor)
    if random() >= win_prob:
        return (team1,team2)
    else:
        return (team2,team1)
def playgameCDF2023(team1, team2, T):
    """There's a difference between flipping a game in an existing
    bracket, and playing a game from scratch. If we're going to just
    use Boltzmann statistics to play a game from scratch, we can make
    life easy by using the Boltzmann factor to directly pick a
    winner.
    """
    #print(team1,team2)
    #ediff = deltaU(team1, team2)
    #boltzmann_factor = exp(-ediff/T)
    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(kpomdata2[team1]["AdjO"],kpomdata2[team1]["AdjD"],kpomdata2[team1]["AdjT"],kpomdata2[team2]["AdjO"],kpomdata2[team2]["AdjD"],kpomdata2[team2]["AdjT"],LeagueTempo,LeagueOE)
    win_prob =scipy.stats.norm(0,10.5).cdf(PHomeTeamSpread)
    #win_prob = boltzmann_factor/(1+boltzmann_factor) if boltzmann_factor < inf else 1
    # So, prob of team 1 winning is then boltzmann_factor/(1+boltzmann_factor)
    if random() >= win_prob:
        return (team1,team2)
    else:
        return (team2,team1)
def playgamesfortestingCDF(team1, team2, ntrials, T):
    print("Boltzmann tells that the ratio of team1 winning to team 2"+ 
          "winning should be")
    print(exp(-deltaU(team1,team2)/T))
    wins = {team1:0,team2:0}
    for i in range(ntrials):
        winner,loser = playgameCDF(team1,team2,T)
        wins[winner] = wins[winner] + 1
    print("wins {} {} {} {} {}".format(wins, wins[team1]/wins[team2], 
                                       wins[team2]/wins[team1], 
                                       wins[team1]/ntrials, 
                                       wins[team2]/ntrials))   
def playgamesfortestingCDF2(team1, team2, ntrials, T):
    print("Boltzmann tells that the ratio of team1 winning to team 2"+ 
          "winning should be")
    print(exp(-deltaU(team1,team2)/T))
    wins = {team1:0,team2:0}
    for i in range(ntrials):
        winner,loser = playgameCDF2023(team1,team2,T)
        wins[winner] = wins[winner] + 1
    print("wins {} {} {} {} {}".format(wins, wins[team1]/wins[team2], 
                                       wins[team2]/wins[team1], 
                                       wins[team1]/ntrials, 
                                       wins[team2]/ntrials))     
    
    


#strength = RAS.kenpom['Luck']
#strength = RAS.sagarin['Rating']
#strength = RAS.kenpom['Pyth']
#strength = kenpom['AdjEM']

#T = 0.5 # In units of epsilon/k
#T = 2.5 # In units of epsilon/k


#@memoized
def energy_of_flipping(current_winner, current_loser):
    """Given the current winner and the current loser, this calculates
    the energy of swapping, i.e. having the current winner lose.
    """
    return (default_energy_function(current_loser, current_winner) - 
            default_energy_function(current_winner, current_loser))



# Here are the "magic functions" I mentioned to get pairs of teams.

#from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def pairs(iterable):
    return grouper(2,iterable)



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

    
def runbracket(teams, T):
    # How many rounds do we need?
    nrounds = int(np.log2(len(teams)))
    winners = teams #they won to get here!
    all_winners = [winners]
    for round in range(nrounds):
        winners, losers = playround(winners, T)
        all_winners.append(winners)
    return all_winners


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

#@profile
def simulate(ntrials, region, T, printonswap=False, printbrackets=True):
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
    b = Bracket(teams, T)
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
    if printbrackets:
        print("Lowest energy bracket")
        print(lb)
        print("Most common bracket (%s)"%mcb_count)
        print(mcb)
    return sr

def runbracket1(ntrials, T):
    results = {'all':simulate(ntrials,'all',T)}
    return results

def runbracket2(ntrials1, ntrials2, T):
    results = {}
    regions = 'midwest west south east'.split()
    for (i,region) in enumerate(regions):
        results[region] = simulate(ntrials1, region, T, printbrackets=False)
    # Make a new bracket from our final four
    teams = [results[region].most_common_bracket.bracket[-1][0] for region in regions]
#    ff_lb, ff_mcb, ff_mcb_count = simulate(ntrials2, teams, T, newfig=i+1, 
    ff_sr = simulate(ntrials2, teams, T, printbrackets=False)

    print("YOUR LOWEST ENERGY BRACKETS")
    for region in regions:
        print("LOWEST ENERGY BRACKET FOR REGION", region)
        print(results[region].lowest_bracket)
        print()
    print( "LOWEST ENERGY BRACKET FOR FINAL FOUR")
    print( ff_sr.lowest_bracket)
        
    print( "YOUR MOST COMMON BRACKETS")
    for region in regions:
        print("MOST COMMON BRACKET FOR REGION", region)
        print(results[region].most_common_bracket)
        print("number of times this bracket happened:",results[region].most_common_bracket_count)
        print()
        print()
    print( "MOST COMMON BRACKET FOR FINAL FOUR")
    print( ff_sr.most_common_bracket)
    print( "number of times this bracket happened:",ff_sr.most_common_bracket_count)
    results['final four'] = ff_sr
    return results


class Bracket(object):
    def __init__(self, teams, T, bracket=None):
        """
        
        Arguments:
        - `teams`:
        - `T`:
        """
        self.teams = teams
        self.T = T
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


def GetBracketMatrixRegionals():
    BracketLookup="http://www.gadepool.com/bracketology.html"
    res = requests.get(BracketLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))[0]
    df=df.iloc[1:]
    df.columns=["seed","south","east","midwest","west"]
    df.drop(['seed'], axis = 1, inplace = True)
    #df1=df.iloc[8:, 0:4]
    now = datetime.datetime.now()
    theDateToday=now.strftime("%Y-%m-%d")
    fileNameToPrint="C:/Users/michael/BracketProjection"+theDateToday+".csv"
    #df.to_csv(fileNameToPrint)
    return(df)

def GetBracketMatrix():
    BracketLookup="http://bracketmatrix.com"
    res = requests.get(BracketLookup)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))[0]
    df1=df.iloc[3:, 0:4]
    return(df1)

def set_energy_function(ef):
    global default_energy_function
    default_energy_function = ef

   
    
#strength = kenpom['AdjEM']    
def default_energy_game(winner, loser):
    """This is where you'll input your own energy functions. Here are
    some of the things we talked about in class. Remember that you
    want the energy of an "expected" outcome to be lower than that of
    an upset.
    """
    result = -(strength[winner] - strength[loser])
    result = regional_rankings[winner] - regional_rankings[loser]
    result = regional_rankings[winner]/regional_rankings[loser]
    result = -(strength[winner]/strength[loser])
    result = -(strength[winner]-strength[loser])/200.0
    #result = random()
    #result = color of team 1 jersey better than color of team 2 jersey
    #print "energy_game(",winner,loser,")",result
    return result

def log5_energy_game(winner, loser):
    strength = RAS.kenpom['Pyth']
    A,B = strength[winner],strength[loser]
    # see http://207.56.97.150/articles/playoff2002.htm
    win_pct = (A-A*B)/(A+B-2*A*B)
    return -win_pct
def My_energy_game(winner, loser):
    #strength = RAS.kenpom['Pyth']
    #A,B = strength[winner],strength[loser]
    # see http://207.56.97.150/articles/playoff2002.htm
    #PHomeTeamSpread=NewgetGamePredictionNeutralCourt(kpomdata[winner]["AdjO"],kpomdata[winner]["AdjD"],kpomdata[winner]["AdjT"],kpomdata[loser]["AdjO"],kpomdata[loser]["AdjD"],kpomdata[loser]["AdjT"],LeagueTempo,LeagueOE)
    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(kpomdata2[winner]["AdjO"],kpomdata2[winner]["AdjD"],kpomdata2[winner]["AdjT"],kpomdata2[loser]["AdjO"],kpomdata2[loser]["AdjD"],kpomdata2[loser]["AdjT"],LeagueTempo,LeagueOE)

    win_prob =scipy.stats.norm(0,10.5).cdf(PHomeTeamSpread)

    #win_pct = (A-A*B)/(A+B-2*A*B)
    return (win_prob-.02)

def Zero_energy_game(winner, loser):
    result = 0
    return result


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
def FullDisplay(df):
    with pd.option_context('display.max_rows',7,'display.max_columns',None):
        display(df)
def keep_first_four(df):
    # Group the dataframe by 'Seed' and keep only the first 4 rows of each group
    df = df.groupby('Seed').head(4)
    return df
def get_next_region(seed):
    seed_region = {i: 0 for i in range(1, 17)}
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
    

    # Function to get the next region for a seed


    # Apply the function to the 'Seed' column to create the 'region' column
    dfe['region'] = dfe['Seed'].apply(get_next_region)

    dfe = keep_first_four(dfe)
    dfp = dfe.sort_values('Seed')

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
def playgameCDF2023(team1, team2, T):
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
    PHomeTeamSpread=NewgetGamePredictionNeutralCourt(TRankDict[team1]["AdjOE"],TRankDict[team1]["AdjDE"],TRankDict[team1]["ADJ. T"],TRankDict[team2]["AdjOE"],TRankDict[team2]["AdjDE"],TRankDict[team2]["ADJ. T"],LeagueTempo,LeagueOE)
    
    win_prob =scipy.stats.norm(0,10.5).cdf(PHomeTeamSpread)
    #win_prob = boltzmann_factor/(1+boltzmann_factor) if boltzmann_factor < inf else 1
    # So, prob of team 1 winning is then boltzmann_factor/(1+boltzmann_factor)
    if random() >= win_prob:
        return (team1,team2)
    else:
        return (team2,team1)
