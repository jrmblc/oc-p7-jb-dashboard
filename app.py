# -*- coding: utf-8 -*-
import pandas as pd

import requests
import json

import dash
from dash.dependencies import Input, Output
import plotly.express as px

#==============================================================================

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


app.layout = dash.html.Div([
    
    dash.html.H1("Home Credit - Customers Default Risk"),
    
    dash.html.Div(
        children=[
            dash.html.Div(children="Select a client ID", className="menu-title"),
            dash.dcc.Dropdown(
                    id="id_client",
                    options=[{'label': str(i), 'value': i} for i in id_list],
                    value=id_list[0]
                            )
                    ]
            ),
    
    dash.html.H3(id='prediction'),
    dash.html.Div(id='proba_0'),
    dash.html.Div(id='proba_1'),
    
    dash.html.Div(
        children=[
            dash.html.Div(children="Predominate features in the composition "+
                          "of the answer ", className="menu-title"),
            dash.dcc.Graph(
                        id='graph'
                          )
                 ])
    ])

@app.callback(
    [Output(component_id='graph', component_property='figure'),
     Output(component_id='prediction', component_property='children'),
     Output(component_id='proba_0', component_property='children'),
     Output(component_id='proba_1', component_property='children')],     
    [Input(component_id='id_client', component_property='value')]
    )
def update_figure(id_client):
    data_client = data(id_client)
    #--------------------------------------------------------------------------
    if data_client['prediction'] == 0:
        prediction="Credit approval available for this customer"
    else:
        prediction="Credit approval not available for this customer"
    prediction = 'Answer : '+ prediction 
    #--------------------------------------------------------------------------    
    proba_0 = 'Success rate: {:0.1f}%'.format(data_client['proba_0']*100)
    #--------------------------------------------------------------------------  
    proba_1 = 'Failure rate: {:0.1f}%'.format(data_client['proba_1']*100)
    #--------------------------------------------------------------------------   
    df_importance=pd.DataFrame()
    df_importance['feature'] = data_client[str(id_client)].keys()
    df_importance['score'] = data_client[str(id_client)].values()
        
    nb=8 #nb of best and worst scores
    df_plus = df_importance.sort_values(by='score', ascending=False)[:nb]
    df_minus = df_importance.sort_values(by='score', ascending=False)[-nb:]
    df_selected = pd.concat([df_plus,df_minus])

    fig = px.bar(df_selected, x='feature', y='score')
    #--------------------------------------------------------------------------
    return fig, prediction, proba_0, proba_1

#==============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
