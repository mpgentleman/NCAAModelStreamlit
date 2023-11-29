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

import requests
import io
from pandas.api.types import is_numeric_dtype
import os
from lets_plot import *
LetsPlot.setup_html()

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


def plot_line_chartLetsPlot(df, teams):
    # Filter the dataframe for the selected teams
    df = df[df['Team'].isin(teams)]

    # Convert the 'Date_zero' column to datetime
    df['Date_zero'] = pd.to_datetime(df['Date_zero'])

    # Sort the dataframe by date
    df = df.sort_values('Date_zero')

    # Create the line chart
    for team in teams:
        df_team = df[df['Team'] == team]
        p = ggplot(df_team, aes(x='Date_zero', y='margin_net')) + \
            geom_line(color='red', size=1.5) + \
            ggtitle('Margin Net Over Time') + \
            xlab('Date') + \
            ylab('Margin Net') + \
            theme(axis_text_x=element_text(angle=45, hjust=1))
        st.write(p)
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

        TeamDatabase=pd.read_csv("Data/TeamDatabase"+season+"T.csv")
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
st.set_page_config(page_title="MG Rankings",layout="wide")
#TeamDatabase2=pd.read_csv("TeamDatabase.csv")
TeamDatabase2=pd.read_csv("Data/TeamDatabase2023.csv")
AllGames=pd.read_csv("Data/Season_GamesAll.csv")
AwayTeamAll=list(TeamDatabase2['OldTRankName'])
HomeTeamAll=list(TeamDatabase2['OldTRankName'])
MG_Rank=pd.read_csv("Data/MGRankings_2024_DB.csv")
teams = MG_Rank['Tm_'].unique()
#st.title('NCAA Head to Head Matchup')
page = st.sidebar.selectbox('Select page',['MG Rankings','Todays Games'])

if page == 'MG Rankings':
    #st.write('MG Rankings')
    
    import streamlit.components.v1 as components
    add_selectbox_start =st.sidebar.date_input('Pick date')
    selected_teams = st.multiselect('Select teams:', teams)
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
    if st.button('Run'):
        col1, col2 = st.columns(2)
        with col1:
            components.html(source_code, height = 3000)
        with col2:
            plot_line_chart(MG_Rank, selected_teams)
            plot_line_chartLetsPlot(MG_Rank, selected_teams)
if page == 'Todays Games':
    st.title('NCAA Head to Head Matchup')
    season = st.sidebar.selectbox('Season Selection',['2024','2023'])
    if season == '2024':
        #st.write('2024')
        add_selectbox = st.sidebar.header("Select Todays Date")
        add_selectbox_start =st.sidebar.date_input('Pick date')
        dateString=str(add_selectbox_start)
        dateToday=dateString.replace('-', '')
        #Dailyschedule=pd.read_csv("DailySchedules2023/"+dateToday+"Schedule.csv")
        Dailyschedule=pd.read_csv("Data/DailySchedules2024/"+dateToday+"Schedule.csv")
        d2=dateString.split('-')[1]+'_'+dateString.split('-')[2]+'_'+dateString.split('-')[0]
        themonth=int(dateString.split('-')[1])
        theday=int(dateString.split('-')[2])
        theyear=dateString.split('-')[0]
        
        TeamDatabase2=pd.read_csv("Data/TeamDatabase2024T.csv")
        AllGames=pd.read_csv("Data/Season_GamesAll_2024.csv")
        AwayTeamAll=list(TeamDatabase2['OldTRankName'])
        HomeTeamAll=list(TeamDatabase2['OldTRankName'])
        Tables_Selection=st.sidebar.selectbox('Any or Scheduled ',['Any', 'Todays Games','All Games'])
        if 'All Games' in  Tables_Selection:
            allcols=AllGames.columns
            gb = GridOptionsBuilder.from_dataframe(AllGames,groupable=True)
            gb.configure_columns(allcols, cellStyle=cellStyle)
            csTotal=cellStyleDynamic(Dailyschedule.Reg_dif)
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
            opts= {**DEFAULT_GRID_OPTIONS,
               **dict(rowGroupPanelShow='always',getContextMenuItems=agContextMenuItemsDeluxe,)}
            gb.configure_grid_options(**opts)
            keyname='Test All 2024'
            g = _displayGrid(AllGames, gb, key=keyname, height=1200)
        if 'Any' in  Tables_Selection:
            AwayTeam = st.sidebar.selectbox('Away Team',AwayTeamAll)
            HomeTeam = st.sidebar.selectbox('Home Team',HomeTeamAll)
        if 'Todays Games' in  Tables_Selection:
            Tables_Choice=st.sidebar.selectbox('Sort Games By',['Alphabetical', 'Time','Regression_Difference','OverPlaying'])
            if 'Alphabetical'in  Tables_Choice:
                Dailyschedule=Dailyschedule.sort_values(by=['Away'])
            if 'Time' in Tables_Choice:
                Dailyschedule=Dailyschedule.sort_values(by=['Time'])   
            if 'Regression_Difference' in Tables_Choice: 
                Dailyschedule=Dailyschedule.sort_values(by=['Reg_dif'])
            if 'OverPlaying' in Tables_Choice: 
                Dailyschedule=Dailyschedule.sort_values(by=['Over_dif'])
            AwayList=list(Dailyschedule['Away'])
            HomeList=list(Dailyschedule['Home'])
            AwayTeam = st.sidebar.selectbox('Away Team',AwayList)
            HomeTeam = st.sidebar.selectbox('Home Team',HomeList)

        if st.button('Run'):
            dateforRankings=dateToday
            dateforRankings5=d2
            #TeamDatabase2=pd.read_csv("Data/TeamDatabase.csv")
            TeamDatabase2.set_index("OldTRankName", inplace=True)
            #MG_DF1=pd.read_csv("Data/MGRankings"+season+"/tm_seasons_stats_ranks"+dateforRankings5+" .csv")
            #MG_DF1["updated"]=update_type(MG_DF1.tm,TeamDatabase2.UpdatedTRankName)
            #MG_DF1.set_index("updated", inplace=True)
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
                cells=dict(values=[Dailyschedule.Away, Dailyschedule.Home, Dailyschedule.ATSVegas, Dailyschedule.OverUnderVegas, Dailyschedule.HomeAway,Dailyschedule.Reg_dif],
                fill_color = [[rowOddColor,rowEvenColor]*lengthrows],
                    align='left',
                font_size=12,
                height=30))
                ])
                fig.update_layout(width=1200, height=800)
                #st.plotly_chart(fig)
                Dailyschedule = Dailyschedule[['Away','Home','HomeAway','Result_x','ATS','ATSVegas','OverUnderVegas','Reg_dif','Over_dif']]
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
        
    else:
        add_selectbox = st.sidebar.header("Select Todays Date")
        add_selectbox_start =st.sidebar.date_input('Pick date')
        dateString=str(add_selectbox_start)
        dateToday=dateString.replace('-', '')
        #Dailyschedule=pd.read_csv("DailySchedules2023/"+dateToday+"Schedule.csv")
        Dailyschedule=pd.read_csv("Data/DailySchedules2023/"+dateToday+"Schedule.csv")
        d2=dateString.split('-')[1]+'_'+dateString.split('-')[2]+'_'+dateString.split('-')[0]
        themonth=int(dateString.split('-')[1])
        theday=int(dateString.split('-')[2])
        theyear=dateString.split('-')[0]
        get2023Display(Dailyschedule,dateToday,d2,season)
    
