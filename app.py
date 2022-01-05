# -*- coding: utf-8 -*-
import pandas as pd
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
                html.Div("Customers Credit Risk", className="title"),
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
                    value=id_list[0],style={'width': '50%'})
                 ]
                ),
    html.Br(),
    html.Br(),
    html.Div(id='prediction',className="answer"),
    
    html.Div(dcc.Graph(id='graph_rate',style={'width': '40%'})),
           
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
    prediction = 'Answer : '+ prediction 
    #--------------------------------------------------------------------------    
    df_rate=pd.DataFrame()
    df_rate['Rate'] = ['Positive','Negative']
    df_rate['Score'] = data_client['proba']

    fig_rate = px.bar(df_rate, x='Rate', y='Score') 
    #--------------------------------------------------------------------------   
    df_local_ft=pd.DataFrame()
    df_local_ft['Local Feature Importance'] = data_client[str(id_client)].keys()
    df_local_ft['Score'] = data_client[str(id_client)].values()
        
    nb=8 #nb of best and worst scores
    df_plus = df_local_ft.sort_values(by='Score', ascending=False)[:nb]
    df_minus = df_local_ft.sort_values(by='Score', ascending=False)[-nb:]
    df_selected = pd.concat([df_plus,df_minus])

    fig_local_ft = px.bar(df_selected, x='Local Feature Importance', y='Score')
    #--------------------------------------------------------------------------
    return fig_local_ft, prediction, fig_rate

#==============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
