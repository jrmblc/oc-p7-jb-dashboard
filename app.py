# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import requests
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

#==============================================================================
local_server_api = False

if local_server_api:
    API_URL = "http://127.0.0.1:5000/"
else:
    API_URL = "https://oc-p7-jb-api.herokuapp.com/"

#==============================================================================

req_list = requests.get(API_URL+"id_list")
id_list = json.loads(req_list.content)

def data(id_client):
    req_data = requests.get(API_URL+"api?id="+str(id_client))
    return json.loads(req_data.content)
    
#==============================================================================

app = dash.Dash(__name__)
server = app.server


app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Img(src='assets\hcg-logo.png'),
                html.Div("Customers Credit Acceptance", className="title"),
                html.P(" ",
               ),
                     ],className="header"
                   ),

    html.Div(
        children=[
            html.Div("Client ID...", className="menu-title"),
            dcc.Dropdown(
                    id="id_client",
                    options=[{'label': str(i), 'value': i} for i in id_list],
                    value=id_list[0],style={'width': '40%'})
                 ]
                ),
    html.Br(),
    html.Div("Answer...", className="menu-title"),
    html.Div(id='prediction',className="answer"),
    html.Br(),
    
    html.Div(dcc.Graph(id='graph_rate')),
           
    html.Div(dcc.Graph(id='graph_local_ft'))
            
    ],className="all")

@app.callback(
    [Output(component_id='graph_local_ft', component_property='figure'),
     Output(component_id='prediction', component_property='children'),
     Output(component_id='graph_rate', component_property='figure')
     ],     
    [Input(component_id='id_client', component_property='value')]
            )
def update_figure(id_client):
    data_client = data(id_client)
    #--------------------------------------------------------------------------
    if data_client['prediction'] == 0:
        prediction="Credit is accepted "
    else:
        prediction="Credit is declined "
    #--------------------------------------------------------------------------    
    df_rate=pd.DataFrame()
    df_rate['Rate'] = ['Positive','Negative']
    df_rate['Score'] = data_client['proba']
    df_rate['Color'] = ['green','red']

    fig_rate = px.bar(df_rate, x='Rate', y='Score',width=600, height=300) 
    fig_rate.update_traces(marker_color=df_rate['Color'])
    fig_rate.update_layout(margin=dict(l=25, r=25, t=25, b=25),
                           paper_bgcolor="#f0f0f0")
    #--------------------------------------------------------------------------   
    df_local_ft=pd.DataFrame()
    df_local_ft['Local Feature Importance'] = data_client[str(id_client)].keys()
    df_local_ft['Score'] = data_client[str(id_client)].values()
    df_local_ft['Sort'] = df_local_ft['Score'].abs()
    df_local_ft['Color'] = np.where(df_local_ft['Score']<0, 'red', 'green')
        
    nb=25 #nb of best and worst scores
    df_sorted = df_local_ft.sort_values(by='Sort', ascending=False)[:nb]

    fig_local_ft = px.bar(df_sorted, x='Local Feature Importance', y='Score',
                          width=1500, height=600)
    fig_local_ft.update_traces(marker_color=df_sorted['Color'])
    fig_local_ft.update_layout(margin=dict(l=25, r=25, t=25, b=25),
                               paper_bgcolor="#f0f0f0")
    #--------------------------------------------------------------------------
    return fig_local_ft, prediction, fig_rate

#==============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
