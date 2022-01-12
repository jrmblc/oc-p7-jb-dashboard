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

req_ft = requests.get(API_URL+"gfi") #global feature importance
gfi = json.loads(req_ft.content)

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
                html.Div("Customers Credit Acceptance"),
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
    html.Div("Model Proba", className="body-title"),
    html.Div(dcc.Graph(id='graph_rate')),
    html.Br(),
    html.Div("Shapley Additive exPlanations (SHAP)", className="body-title"),      
    html.Div(dcc.Graph(id='graph_shap')),
    html.Br(),
    html.Div("Shapley Additive exPlanations and Feature Importance",
             className="body-title"),
    html.Div(dcc.Graph(id='graph_merge'))   
    ],className="all")

@app.callback(
    [Output(component_id='graph_merge', component_property='figure'),
     Output(component_id='graph_shap', component_property='figure'),
     Output(component_id='graph_rate', component_property='figure'),
     Output(component_id='prediction', component_property='children')
     ],     
    [Input(component_id='id_client', component_property='value')]
            )
def update_figure_client(id_client):
    data_client = data(id_client)
    #--------------------------------------------------------------------------
    if data_client['prediction'] == 0:
        prediction="Credit is accepted "
    else:
        prediction="Credit is declined "
    #--------------------------------------------------------------------------    
    df_rate=pd.DataFrame()
    df_rate['Rate'] = ['For','Against']
    df_rate['Score'] = data_client['proba']
    df_rate['Color'] = ['green','red']

    fig_rate = px.bar(df_rate, x='Rate', y='Score',width=600, height=300) 
    fig_rate.update_traces(marker_color=df_rate['Color'])
    fig_rate.update_layout(margin=dict(l=25, r=25, t=25, b=25),
                           paper_bgcolor="#f0f0f0")
    #--------------------------------------------------------------------------   
    df_shap=pd.DataFrame()
    df_shap['Feature'] = data_client[str(id_client)].keys()
    df_shap['Score'] = data_client[str(id_client)].values()
    df_shap['Sort'] = df_shap['Score'].abs()
    df_shap['Color'] = np.where(df_shap['Score']<0, 'red', 'green')
        
    nb=30 #nb of features on the graph
    df_shap = df_shap.sort_values(by='Sort', ascending=False)[:nb]

    fig_shap = px.bar(df_shap, x='Feature', y='Score',
                          width=1500, height=600)
    fig_shap.update_traces(marker_color=df_shap['Color'])
    fig_shap.update_layout(margin=dict(l=25, r=25, t=25, b=25),
                               paper_bgcolor="#f0f0f0")
    #--------------------------------------------------------------------------
    df_gfi=pd.DataFrame() #global feature importance
    df_gfi['Feature'] = gfi['global_feature_importance'].keys()
    df_gfi['Score'] = gfi['global_feature_importance'].values()
        
    df_merge = df_shap.merge(df_gfi, on='Feature')
    df_merge = df_merge.rename(columns={'Score_x': 'SHAP Score', 
                                        'Score_y': 'Feature Importance'})
    
    fig_merge = px.scatter(df_merge,x='Feature Importance', y='SHAP Score', 
                           text='Feature', width=1500, height=600)
    fig_merge.update_traces(marker_color=df_merge['Color'],
                            textposition="bottom right", textfont_size=12)
    fig_merge.update_layout(margin=dict(l=25, r=25, t=25, b=25),
                                     paper_bgcolor="#f0f0f0")

    #--------------------------------------------------------------------------
    # df_gfi = df_gfi.sort_values(by='Score', ascending=False)[:nb]
    
    # fig_gfi = px.bar(df_gfi, x='Global Feature Importance', y='Score',
    #                       width=1500, height=600)
    # fig_gfi.update_traces(marker_color='blue')
    # fig_gfi.update_layout(margin=dict(l=25, r=25, t=25, b=25),
    #                            paper_bgcolor="#f0f0f0")
    #--------------------------------------------------------------------------
    return fig_merge, fig_shap, fig_rate, prediction
#==============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
