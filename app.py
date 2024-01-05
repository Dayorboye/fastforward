import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq
import plotly.express as px
import pandas as pd
from dash import dash_table
import dash_bootstrap_components as dbc
from datetime import date, datetime
from dash.exceptions import PreventUpdate
from dash import dash
from dash.dash import no_update
# import pymysql
# from sqlalchemy import create_engine
# from dash_auth import BasicAuth
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Sales Report Engine SRE"
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
app.config["suppress_callback_exceptions"] = True

# APP_PATH = str(pathlib.Path(__file__).parent.resolve())
# Abuja_branch_dataset = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "Abuja_branch_dataset.csv")))


# Create Data Pipeline

gsheetid = '1h-4OQN3FGQwO7d2t4xGrPu3T39uydci9Q77ErwLgK6c'
sheet_name = 'Inventory_Summary_With_Sales'

gsheet_url = 'https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}'.format(gsheetid,sheet_name)
url = gsheet_url

# CSV Extract Function
def extract(url):
    df = pd.read_csv(url)
    return df

# Function to format integers to have 16 digits
def format_to_16_digits(x):
    formatted_x = str(int(x))  # Convert to int and then to string to remove decimal points
    if len(formatted_x) < 16:
        formatted_x = '0' * (16 - len(formatted_x)) + formatted_x
    return formatted_x

# Transform
def transform(df, coName, countryName):
    df.dropna(axis = 1, how = 'all', inplace = True)
    df['name'] = coName
    df['country'] = countryName
    df = df[['name','country','Week','Department','SKU','CATEGORY','SEASON','STYLE NAME/INT. CAT','Attribute','Item Description','Ext Price','Ext Price.1','Qty.1','Qty']]
    cols = ['PARTNER NAME','COUNTRY','WEEK','DEPARTMENT','ITEM CODE(16 DIGITS)','CLASSNAME','SEASON','STYLE NAME','COLOUR NAME','DESCRIPTION','ORIGINAL RRP','SALES VALUE LAST WEEK LOCAL','SALES UNITS LAST WEEK','STORE STOCK UNITS']
    df.columns = cols
    df['SALES VALUE LAST WEEK LOCAL'] = df['SALES VALUE LAST WEEK LOCAL'].str.replace(',','')
    df['ORIGINAL RRP'] = df['ORIGINAL RRP'].str.replace(',','')    
    df['STORE STOCK UNITS'] = df['STORE STOCK UNITS'].str.replace(',','')
    df = df.astype({'ORIGINAL RRP':'float','SALES VALUE LAST WEEK LOCAL':'float','SALES UNITS LAST WEEK':'float','STORE STOCK UNITS':'float'})
    df = df.iloc[:-1]
    df['ITEM CODE(16 DIGITS)'] = df['ITEM CODE(16 DIGITS)'].astype(str)

    problematic_values = []

    for idx, value in df['ITEM CODE(16 DIGITS)'].items():
        try:
            df.at[idx, 'ITEM CODE(16 DIGITS)'] = format_to_16_digits(float(value))
        except ValueError:
            problematic_values.append(value)
            df.at[idx, 'ITEM CODE(16 DIGITS)'] = None

    if problematic_values:
        print(f"Problematic values in 'ITEM CODE(16 DIGITS)' column: {problematic_values}")
    return df

# Load to csv
def load_to_csv(df, csv_path):
    csv_path.to_csv(df)


# Load to sql database
def load_to_db(df, sql_connection, table_name):
    df.to_sql(table_name, sql_connection, if_exists='replace', index=False)  


# Logging
# def log(message):
#     timestamp_format = '%Y-%h-%d-%H:%M:%S' # Year-Monthname-Day-Hour-Minute-Second
#     now = datetime.now() # get current timestamp
#     timestamp = now.strftime(timestamp_format)
#     with open("logfile.txt","a") as f:
#         f.write(timestamp + ',' + message + '\n')
    
# log("ETL Job Started")

# log("Extract phase Started")
extracted_data = extract(url)
# log("Extract phase Ended")

# log("Transform phase Started")
coName = 'SMARTMARTLTD'
countryName = 'NIGERIA'
transformed_data = transform(extracted_data,coName,countryName)
# log("Transform phase Ended")

# log("Load phase Started")
df = "transformed_sales_report.csv" 
load_to_csv(df, transformed_data)
# log("Load phase Ended")

# # create aqlalchemy engine for mysql
# sql_connection = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
#                        .format(user="root",
#                                pw="giveme123",
#                                db="dictionary"))
# log('SQL Connection initiated.')

# df = transformed_data 
# table_name = "transformed_sales_report"
# load_to_db(df, sql_connection, table_name)

# log('Data loaded to Database as table. Running the query')


# # create aqlalchemy engine for mysql
# engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
#                        .format(user="root",
#                                pw="giveme123",
#                                db="dictionary"))
# # pymssql


df = pd.read_csv("transformed_sales_report.csv")


Revenue = df['ORIGINAL RRP'].sum().round(2)
TR = 'Revenue  '
StuckLocal = df['SALES VALUE LAST WEEK LOCAL'].sum().round(2)
SL = 'Stuck Local'
StuckUnit = df['STORE STOCK UNITS'].sum().round(2)
SU = 'Stuck Unit'


color = "white"

def drawText(name, val):
    return html.Div(
        dbc.Card(
            html.Div(
                [
                    html.H4(
                        [
                            html.P(name, style={'font-size': '18px'}),
                        ]
                    ),
                    html.H6(f"{val:,}", style={'color': '#91dfd2', 'font-size': '16px'}),
                ],
            ),
            style={
                'width': '150px',
                'padding': '0 0 2px 0',
                'box-shadow': '0px 0px 17px 0px rgba(186, 218, 212, .5)',
                'text-align': 'center',
                'margin': 'auto',
                'border-left': '#FAA831 solid 16px',  # Border to the left
                'border-right': '#FAA831 solid 4px',  # Border to the right
                'border-top': '#91dfd2 solid 8px',  # Border to the left
            },
        ),
    )





params = list(df)
max_length = len(df)



import numpy as np
import matplotlib as mpl
colors={}
def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
c1='#FAA831' 
c2='#9A4800' 
n=9
for x in range(n+1):
    colors['level'+ str(n-x+1)] = colorFader(c1,c2,x/n) 
colors['background'] = '#232425'
colors['text'] = '#fff'

agg_dept_season = df.groupby(['DEPARTMENT','SEASON']).agg({"ORIGINAL RRP" : "sum"}).reset_index()
agg_dept_season = agg_dept_season[agg_dept_season['ORIGINAL RRP']>0]
def drawSun_bst():
    return  html.Div([
                dcc.Graph(
                    figure=px.sunburst(agg_dept_season, path=['DEPARTMENT','SEASON'], values='ORIGINAL RRP',
                                       color='ORIGINAL RRP',
                    color_continuous_scale=[colors['level2'], colors['level10']],

                    ).update_layout(
                        title_text='Department & Season',
                        paper_bgcolor='#161a28',
                        plot_bgcolor='#161a28',
                        font=dict(size=10, color=colors['text']),
                        height=400
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
    ])






agg_Dept_Rev = df.groupby('DEPARTMENT').agg({"SALES VALUE LAST WEEK LOCAL" : "sum"}).reset_index().sort_values(by='SALES VALUE LAST WEEK LOCAL', ascending=False)
agg_Dept_Rev['color'] = colors['level10']
agg_Dept_Rev['color'][:1] = colors['level1']
agg_Dept_Rev['color'][1:2] = colors['level2']
agg_Dept_Rev['color'][2:3] = colors['level3']
agg_Dept_Rev['color'][3:4] = colors['level4']
agg_Dept_Rev['color'][4:5] = colors['level5']
def drawBar_Eng():
    return  html.Div([
                dcc.Graph(
                    figure=go.Figure(data=[go.Bar(x=agg_Dept_Rev['SALES VALUE LAST WEEK LOCAL'],
                                                y=agg_Dept_Rev['DEPARTMENT'], 
                                                marker=dict(color= '#FAA831'),
                                                name='DEPARTMENT', orientation='h',
                                                text=agg_Dept_Rev['SALES VALUE LAST WEEK LOCAL'].astype(int),
                                                textposition='auto',
                                                hoverinfo='text',
                                                hovertext=
                                                '<b>DEPARTMENT</b>:'+ agg_Dept_Rev['DEPARTMENT'] +'<br>' +
                                                '<b>Sales</b>:'+ agg_Dept_Rev['SALES VALUE LAST WEEK LOCAL'].astype(int).astype(str) +'<br>' ,
                                                # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                                                )]

                    ).update_layout(
                        title_text='Best-Selling Department ',
                        paper_bgcolor='#161a28',
                        plot_bgcolor='#161a28',
                        font=dict(size=10, color='white'),
                        height=400
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
    ])


avrg_week_unit = df.groupby('WEEK').agg({"STORE STOCK UNITS" : "sum"}).reset_index()
def drawLine_RavrgT():
    return  html.Div([
                dcc.Graph(style={'overflow':'hidden'},
                    figure=go.Figure(data=[go.Scatter(x=avrg_week_unit['WEEK'], 
                        y=avrg_week_unit['STORE STOCK UNITS'], 
                        fill='tozeroy', fillcolor='#FAA831', 
                        line_color='#91dfd2' )]
                    ).update_layout(
                        title_text='Weekly Stock Unit',
                        height=400, width= 3500,
                        paper_bgcolor='#161a28',
                        plot_bgcolor='#161a28',
                        font=dict(size=10,color='white')
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
    ])









def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Sales Report Engine SRE"),
                    html.H6("Data Extration,Transformation & Report"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.A(
                        html.Button(children="REPORT MANAGEMENT"),
                        href="#",
                    ),
                    html.A(
                        html.Button(children="UPLOAD NEW DATA"),
                        href="https://docs.google.com/spreadsheets/d/1e7ZZBRt37kEElHnyFF_VV_4bVRqjy3oXNw8cY1OhIcw/edit#gid=30863933",
                    ),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Query Engine",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Charts Engine",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )

   


def build_tab_1():
    return [
        # Manually select metrics
        html.Div(
            id="set-specs-intro-container",
            # className='twelve columns',
            children=html.P(
                "Historical streaming data to establish a benchmark, or Report."
            ),
        ),
        html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Div(
                                    [
                                        html.Button(
                                            id="download-button",
                                            children="DOWNLOAD",
                                            n_clicks=0,
                                            style={
                                                "float": "right",
                                                "color": "#92e0d3",
                                                "margin": "0 35px 35px 0",
                                            },
                                        ),

                                        dbc.Input(id='week-input', type='number',
                                            step=1, value=31,
                                        style={
                                            'margin-left': '30px', 
                                            'background-color': '#161a28', 
                                            'color': '#92e0d3'}
                                            
                                        ),

                                    ]
                                ),
                            ]
                        ),
                        
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        dash_table.DataTable(
                                            id="table",
                                            columns=[
                                                {"name": i, "id": i} for i in df.columns
                                            ],
                                            data=df.to_dict("records"),
                                            style_cell=dict(textAlign="left"),
                                            style_header=dict(backgroundColor="#161a28", 
                                                              fontWeight='bold',
                                                              fontSize='16px',
                                                              padding = '18px 10px 18px 10px'),
                                            style_data=dict(backgroundColor="black"),
                                        ),

                                        dcc.Download(id="download-data"),
                                    ],
                                    style={"maxHeight": "80vh", "width": "100%", "overflow": "scroll","margin": "40px 0 40px 0px"},
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        ),
    ]



def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            dbc.Input(
                id='week-input', 
                type='number', 
                step=1, 
                value= 31, 
                style={
                    'margin-left': '30px', 
                    'background-color': '#161a28', 
                    'color': '#92e0d3',
                    'margin-top':'-60px'}
                    ),

            html.Div(
                id="card-1",
                children=[
                    # html.P("Total Revenue"),
                    drawText(TR,Revenue)
                ],
            ),
            html.Div(
                id="card-2",
                children=[
                    # html.P("Stuck Local"),
                    drawText(SL,StuckLocal)
                ],
            ),
            html.Div(
                id="card-3",
                children=[
                    # html.P("Stuck Unit"),
                    drawText(SU,StuckUnit)
                ],
            ),
            
            # html.Div(
            #     id="utility-card",
            #     children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
            # ),
        ],
    )


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def build_top_panel(stopped_interval):
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="seven columns",
                children=[
                    # bar_fig()
                    drawSun_bst()
                    

                ],
            ),
            # Piechart
            html.Div(
                id="ooc-piechart-outer",
                className="five columns",
                children=[
                        
                      drawBar_Eng()
                        
                        # drawPie(Gross_revenue, gender)
                        # bar_fig()
                        
                ],
            ),
        ],
    )






def build_chart_panel():
    return html.Div(
        id="control-chart-container",
        className="twelve columns",
        children=[
            drawLine_RavrgT()
            
            # generate_section_banner("Live Chart"),
            # drawPie(Gross_revenue, gender)
            
        ],
    )




app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=2 * 60 * 1000,  # update every 2 minutes
            n_intervals=50,  # start at batch 50
            disabled=True,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
        dcc.Store(id="value-setter-store", ),
        dcc.Store(id="n-interval-stage", data=50),
    ],
)



@app.callback(
    [
        Output("card-1", "children"),
        Output("card-2", "children"),
        Output("card-3", "children"),
    ],
    [Input("week-input", "value")],
)
def update_text_elements(selected_week):
    # Assuming you have the necessary data, update the values for the text elements
    filtered_data = df[df['WEEK'] == selected_week]

    updated_revenue = filtered_data['ORIGINAL RRP'].sum().round(2)
    updated_stuck_local = filtered_data['SALES VALUE LAST WEEK LOCAL'].sum().round(2)
    updated_stuck_unit = filtered_data['STORE STOCK UNITS'].sum().round(2)

    card_1 = drawText(TR, updated_revenue)
    card_2 = drawText(SL, updated_stuck_local)
    card_3 = drawText(SU, updated_stuck_unit)

    return card_1, card_2, card_3







@app.callback(
    Output('table', 'data'),  # Assuming 'table' is the ID of your DataTable component
    Input('week-input', 'value'),
#     Input('week-range-slider', 'value')
)
def update_table(week_input):
    # Use the selected week or week range to filter your dataset
    filtered_data = df[(df['WEEK'] == week_input)]

    # Return the filtered data to update the DataTable
    return filtered_data.to_dict('records')


@app.callback(
    Output('table_out', 'children'), 
    Input('table', 'active_cell'))
def update_graphs(active_cell):
    if active_cell:
        cell_data = df.iloc[active_cell['row']][active_cell['column_id']]
        return f"Data: \"{cell_data}\" from table cell: {active_cell}"
    return "Click the table"



@app.callback(
    Output("download-data", "data"),
    [Input("download-button", "n_clicks")],
    prevent_initial_call=True,
)
def download_data(n_clicks):
    if n_clicks == 0:
        return no_update

    # Create a CSV string from the filtered DataFrame
    csv_string = df.to_csv(index=False, encoding="utf-8")

    # Return the download data
    return dict(content=csv_string, filename="downloaded_data.csv")        


@app.callback(
    [Output("app-content", "children"), Output("interval-component", "n_intervals")],
    [Input("app-tabs", "value")],
    [State("n-interval-stage", "data")],
)
def render_tab_content(tab_switch, stopped_interval):
    if tab_switch == "tab1":
        return build_tab_1(), stopped_interval
    return (
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=[build_top_panel(stopped_interval), build_chart_panel()],
                ),
            ],
        ),
        stopped_interval,
    )


# ======= update progress gauge =========




# Running the server
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)



























