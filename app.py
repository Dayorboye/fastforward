import os
import pathlib

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq
import plotly.express as px
import pandas as pd
import numpy as np
from dash import dash_table
import dash_bootstrap_components as dbc
from datetime import date, datetime
from dash.exceptions import PreventUpdate
from dash import dash
from dash.dash import no_update
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pymysql
from sqlalchemy import create_engine
import socket
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


# Create Data Pipeline


def extract_and_append_all_tables(sheet_key):
    # Use Google Sheets API credentials
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name("dogwood-courier-406511-c8a6ccfe04e0.json", scope)
    gc = gspread.authorize(credentials)

    # Open the Google Sheet by key
    workbook = gc.open_by_key(sheet_key)

    # Initialize an empty DataFrame to store the appended tables
    appended_table = pd.DataFrame()

    # Iterate through each sheet in the workbook
    for sheet in workbook.worksheets():
        # Get all values from the sheet
        data = sheet.get_all_values()

        # Convert the data to a DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        # Append the DataFrame to the main table
        appended_table = appended_table.append(df, ignore_index=True)
    return appended_table


# Function to format integers to have 16 digits
def format_to_16_digits(x):
    formatted_x = str(int(x))  # Convert to int and then to string to remove decimal points
    if len(formatted_x) < 16:
        formatted_x = '0' * (16 - len(formatted_x)) + formatted_x
    return formatted_x


def transform(appended_table,companyName,countryName):
    
    appended_table.drop(index=appended_table.index[:5],axis=0, inplace=True)
    appended_table.columns = appended_table.iloc[0]
    appended_table = appended_table[1:]
    appended_table.dropna(axis = 1, how = 'all', inplace = True)
    cols = ['', 'Department', 'Week', '', 'Vendor', '', 'Item Name', '',
       'Item Description', '', 'Attribute', '', 'Size', '', 'Item #', '',
       'UPC', '', 'Alternate Lookup', '', 'Dept Code', '', 'Vendor Code', '',
       'CATEGORY', '', 'GENDER', '', 'SEASON', '', 'STYLE NAME/INT. CAT', '',
       'SKU', '', 'Quick Pick Group', '', 'Ext Price(Inventory)', '', 'Qty(Inventory)', '',
       'Ext Cost(Inventory', '', 'Ext Price(Sold)', '', 'Qty(Sold)', '', 'Ext Cost(Sold)']
    appended_table.columns = cols
    appended_table['name'] = companyName
    appended_table['country'] = countryName
    appended_table = appended_table[['name','country','Week','Department','SKU','CATEGORY','SEASON',
                                     'STYLE NAME/INT. CAT','Attribute','Item Description','Ext Price(Inventory)',
                                     'Ext Price(Sold)','Qty(Sold)','Qty(Inventory)']]
    
    colss = ['PARTNER NAME','COUNTRY','WEEK','DEPARTMENT','ITEM CODE(16 DIGITS)','CLASSNAME','SEASON',
            'STYLE NAME','COLOUR NAME','DESCRIPTION','ORIGINAL RRP','SALES VALUE LAST WEEK LOCAL',
            'SALES UNITS LAST WEEK','STORE STOCK UNITS']
    appended_table.columns = colss
    appended_table['SALES VALUE LAST WEEK LOCAL'] = appended_table['SALES VALUE LAST WEEK LOCAL'].str.replace(',','')
    appended_table['ORIGINAL RRP'] = appended_table['ORIGINAL RRP'].str.replace(',','')    
    appended_table['STORE STOCK UNITS'] = appended_table['STORE STOCK UNITS'].str.replace(',','')
#     appended_table = appended_table.astype({'WEEK':'float','ORIGINAL RRP':'float','SALES VALUE LAST WEEK LOCAL':'float','SALES UNITS LAST WEEK':'float','STORE STOCK UNITS':'float'})
    numeric_cols = ['WEEK', 'ORIGINAL RRP', 'SALES VALUE LAST WEEK LOCAL', 'SALES UNITS LAST WEEK', 'STORE STOCK UNITS']

    for col in numeric_cols:
        appended_table[col] = pd.to_numeric(appended_table[col], errors='coerce')

    appended_table = appended_table.dropna(subset=numeric_cols)  # Drop rows with NaN values after conversion
    appended_table = appended_table.iloc[:-1]
        # Applying the function to the DataFrame column
    appended_table['ITEM CODE(16 DIGITS)'] = appended_table['ITEM CODE(16 DIGITS)'].astype(str)

    problematic_values = []

    for idx, value in appended_table['ITEM CODE(16 DIGITS)'].items():
        try:
            appended_table.at[idx, 'ITEM CODE(16 DIGITS)'] = format_to_16_digits(float(value))
        except ValueError:
            problematic_values.append(value)
            appended_table.at[idx, 'ITEM CODE(16 DIGITS)'] = None

    if problematic_values:
        print(f"Problematic values in 'ITEM CODE(16 DIGITS)' column: {problematic_values}")
    
    appended_table['ID'] = np.arange(1, len(appended_table)+1)
    return appended_table

  

    
def load_to_sql(appended_table, output_file):
    
    # Write the final appended table to an Excel file
    appended_table.to_sql(output_file, sql_connection, if_exists='replace', index=False)
    print(f"All tables successfully appended and saved to {output_file}")
    

# Replace the placeholders with your actual database credentials
host = 'sql3.freesqldatabase.com'
database_name = 'sql3675260'
user = 'sql3675260'
password = 'tnqjLxZuFv'
port = '3306'

# Create an SQLAlchemy engine to connect to the MySQL database
sql_connection = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}")  

sheet_key = "1e7ZZBRt37kEElHnyFF_VV_4bVRqjy3oXNw8cY1OhIcw"  # Replace with your Google Sheet key
output_file = "extracted_tables"  # Replace with your desired output file name
extracted_data = extract_and_append_all_tables(sheet_key)
coName = 'SMARTMARTLTD'
countryName = 'NIGERIA'
transformed_data = transform(extracted_data,coName,countryName)
load_to_sql(transformed_data, output_file)

# Optimize loading by using a SQL query instead of directly loading the entire table
# Modify the query as per your requirements to load specific columns or apply filters
query = "SELECT * FROM extracted_tables"

# Use the read_sql() function with the SQL query to load data into a DataFrame
# Use the read_sql() function with the SQL query to load data into a DataFrame
try:
    # Optimize loading by using a SQL query instead of directly loading the entire table
    # Modify the query as per your requirements to load specific columns or apply filters
    query = "SELECT * FROM extracted_tables"

    # Use the read_sql() function with the SQL query to load data into a DataFrame
    # Adjust chunksize as needed; it specifies the number of rows fetched at a time
    chunksize = 10000  # Experiment with different values for optimal performance
    df_chunks = pd.read_sql(query, con=sql_connection , chunksize=chunksize)
    
    # Initialize an empty DataFrame to concatenate chunks
    df = pd.concat(df_chunks)
    
    # Now df contains the entire table data
    print("Data loaded successfully.")
except Exception as e:
    print("Error occurred while loading data:", str(e))
    df = pd.DataFrame()  # Define an empty DataFrame to handle the case of failure to load data

# Display the DataFrame directly in the cell output
df = df


# Assuming the columns are named as in your previous example
columns_to_select = ['PARTNER NAME','COUNTRY','WEEK','DEPARTMENT','ITEM CODE(16 DIGITS)',
                     'CLASSNAME','SEASON','STYLE NAME','COLOUR NAME','DESCRIPTION',
                     'ORIGINAL RRP','SALES VALUE LAST WEEK LOCAL','SALES UNITS LAST WEEK',
                     'STORE STOCK UNITS']

# Create a new DataFrame with selected columns using .loc[]
df = df.loc[:, columns_to_select]

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
                'border-radius': '4px',
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
                                            style_data=dict(backgroundColor="black",
                                                              fontSize='14px',
                                                              padding = '0 10px 0 10px'),
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
                    'margin': '0 0 20px 0px', 
                    'background-color': '#161a28', 
                    'color': '#92e0d3',
                    'width': '150px'}
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
            
            html.Div(
                id="utility-card",
                children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
            ),
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
    app.run_server(debug=True, port=8050)

