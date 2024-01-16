import os
from datetime import date, datetime

import dash
from dash import dash_table
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq
import gspread
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from dash.exceptions import PreventUpdate

# Disable pandas warning for assignment on copy
pd.options.mode.chained_assignment = None

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Sales Report Engine SRE"
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
app.config["suppress_callback_exceptions"] = True



# Create Data Pipeline


# Google Sheets API credentials
def extract_and_append_all_tables(sheet_key):
    # Use Google Sheets API credentials
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name("dogwood-courier-406511-c8a6ccfe04e0.json", scope)
    gc = gspread.authorize(credentials)

    # Open the Google Sheet by key
    workbook = gc.open_by_key(sheet_key)

    # Initialize an empty DataFrame to store the appended tables
    appended_table = pd.DataFrame()

    # Global variable to keep track of the total rows processed
    global_total_rows_processed = 0

    # Iterate through each sheet in the workbook
    for sheet in workbook.worksheets():
        # Get all values from the sheet
        data = sheet.get_all_values()

        # Convert the data to a DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        try:
            # Add a new column with the global unique index
#             df['UniqueIndex'] = range(global_total_rows_processed, global_total_rows_processed + len(df))
            
            # Add the specified code to clean up the DataFrame
            df.drop(index=df.index[:5],axis=0, inplace=True)
            df.columns = df.iloc[0]
            df = df[1:]
            df.dropna(axis=1, how='all', inplace=True)

            # Append the cleaned DataFrame to the main table
            appended_table = appended_table.append(df, ignore_index=True)
            
            # Update the global total rows processed
            global_total_rows_processed += len(df)
        except Exception as e:
            print(f"Error appending DataFrame from sheet '{sheet.title}': {e}")

    return appended_table


# Function to format integers to have 16 digits
def format_to_16_digits(x):
    formatted_x = str(int(x))  # Convert to int and then to string to remove decimal points
    if len(formatted_x) < 16:
        formatted_x = '0' * (16 - len(formatted_x)) + formatted_x
    return formatted_x


def transform_data(appended_table, company_name, country_name):
    cols = ['', 'Department', 'Week','Branch', '', 'Vendor', '', 'Item Name', '',
       'Item Description', '', 'Attribute', '', 'Size', '', 'Item #', '',
       'UPC', '', 'Alternate Lookup', '', 'Dept Code', '', 'Vendor Code', '',
       'CATEGORY', '', 'GENDER', '', 'SEASON', '', 'STYLE NAME/INT. CAT', '',
       'SKU', '', 'Ext Price(Inventory)', '', 'Qty(Inventory)', '', 'Ext Price(Sold)', '', 'Qty(Sold)']
#     cols = ['', 'Department', 'Week', '', 'Vendor', '', 'Item Name', '',
#        'Item Description', '', 'Attribute', '', 'Size', '', 'Item #', '',
#        'UPC', '', 'Alternate Lookup', '', 'Dept Code', '', 'Vendor Code', '',
#        'CATEGORY', '', 'GENDER', '', 'SEASON', '', 'STYLE NAME/INT. CAT', '',
#        'SKU', '', 'Quick Pick Group', '', 'Ext Price(Inventory)', '', 'Qty(Inventory)', '',
#        'Ext Cost(Inventory', '', 'Ext Price(Sold)', '', 'Qty(Sold)', '', 'Ext Cost(Sold)']
    appended_table.columns = cols
    appended_table['name'] = company_name
    appended_table['country'] = country_name
    appended_table = appended_table[['name','country','Week','Branch','Department','SKU','CATEGORY','SEASON',
                                     'STYLE NAME/INT. CAT','Attribute','Item Description','Ext Price(Inventory)',
                                     'Ext Price(Sold)','Qty(Sold)','Qty(Inventory)']]
    
    colss = ['PARTNER NAME','COUNTRY','WEEK','STORE','DEPARTMENT','ITEM CODE(16 DIGITS)','CLASSNAME','SEASON',
            'STYLE NAME','COLOUR NAME','DESCRIPTION','ORIGINAL RRP','SALES VALUE LAST WEEK LOCAL',
            'SALES UNITS LAST WEEK','STORE STOCK UNITS']
    appended_table.columns = colss
    
    appended_table['ITEM CODE(16 DIGITS)'] = appended_table['ITEM CODE(16 DIGITS)'].astype(str)

    problematic_values = []

    for idx, value in appended_table['ITEM CODE(16 DIGITS)'].items():
        try:
            # Skip the conversion attempt for empty strings
            if value != '':
                appended_table.at[idx, 'ITEM CODE(16 DIGITS)'] = format_to_16_digits(float(value))
        except ValueError:
            problematic_values.append(value)
            appended_table.at[idx, 'ITEM CODE(16 DIGITS)'] = None

    if problematic_values:
        print(f"Problematic values in 'ITEM CODE(16 DIGITS)' column: {problematic_values}")

    appended_table = appended_table.iloc[:-1]
    
    return appended_table






def authenticate_google_sheets():
    # Use Google Sheets API credentials
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name("dogwood-courier-406511-c8a6ccfe04e0.json", scope)
    gc = gspread.authorize(credentials)
    return gc

def load_to_google_sheets(data_frame, sheet_url, sheet_name):
    # Extract sheet key from the sheet URL
    sheet_key = sheet_url.split("/")[5]

    # Authenticate with Google Sheets
    gc = authenticate_google_sheets()

    # Open the Google Sheet by key
    workbook = gc.open_by_key(sheet_key)

    # Get the specified sheet by name; create it if not exists
    try:
        worksheet = workbook.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = workbook.add_worksheet(title=sheet_name, rows="100", cols="20")

    # Clear the existing data in the sheet
    worksheet.clear()

    # Manually convert problematic columns to strings to avoid numeric conversion issues
    for column in data_frame.columns:
        if data_frame[column].dtype == 'float64':
            data_frame[column] = data_frame[column].astype(str)

    # Write the DataFrame to the Google Sheet
    set_with_dataframe(worksheet, data_frame)

    print(f"Data successfully loaded into Google Sheet '{sheet_name}' in the workbook with key '{sheet_key}'")



company_name = 'SMARTMARTLTD'
country_name = 'NIGERIA'

sheet_key = "1e7ZZBRt37kEElHnyFF_VV_4bVRqjy3oXNw8cY1OhIcw"  # Replace with your Google Sheet key


# Apply transformations
appended_table = extract_and_append_all_tables(sheet_key)
transformed_data = transform_data(appended_table, company_name, country_name)


# Load transformed data to Google Sheets
google_sheet_url = "https://docs.google.com/spreadsheets/d/1uC6CVvxTUM3fmXRB7ec7xoF9hIcPVpB9SECXgxqSmX0/edit#gid=0"
sheet_name = "Inventory_Sales_Summary"
load_to_google_sheets(transformed_data, google_sheet_url, sheet_name)



# Function to load data into the DataFrame
def load_data(gsheet_url):
    df = pd.read_csv(gsheet_url)

    # Convert relevant columns to string
    string_cols = ['WEEK','SALES VALUE LAST WEEK LOCAL', 'ORIGINAL RRP', 'STORE STOCK UNITS']
    df[string_cols] = df[string_cols].astype(str)

    # Replace commas in string columns
    df['WEEK'] = df['WEEK'].str.replace(',', '')
    df['SALES VALUE LAST WEEK LOCAL'] = df['SALES VALUE LAST WEEK LOCAL'].str.replace(',', '')
    df['ORIGINAL RRP'] = df['ORIGINAL RRP'].str.replace(',', '')
    df['STORE STOCK UNITS'] = df['STORE STOCK UNITS'].str.replace(',', '')

    columns_to_select = ['PARTNER NAME', 'COUNTRY', 'STORE','WEEK', 'DEPARTMENT', 'ITEM CODE(16 DIGITS)',
                         'CLASSNAME', 'SEASON', 'STYLE NAME', 'COLOUR NAME', 'DESCRIPTION',
                         'ORIGINAL RRP', 'SALES VALUE LAST WEEK LOCAL', 'SALES UNITS LAST WEEK',
                         'STORE STOCK UNITS']
    df = df.loc[:, columns_to_select]

    numeric_cols = ['WEEK', 'ORIGINAL RRP', 'SALES VALUE LAST WEEK LOCAL', 'SALES UNITS LAST WEEK', 'STORE STOCK UNITS']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols)  # Drop rows with NaN values after conversion
    df['ID'] = np.arange(1, len(df) + 1)

    return df

# Load data into DataFrame from Google Sheets
gsheetid = '1uC6CVvxTUM3fmXRB7ec7xoF9hIcPVpB9SECXgxqSmX0'
sheet_name = 'Inventory_Sales_Summary'
gsheet_url = f'https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
df = load_data(gsheet_url)

df['ITEM CODE(16 DIGITS)'] = transformed_data['ITEM CODE(16 DIGITS)']


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
                'border-radius': '5px',
                'padding': '0 0 2px 0',
                'box-shadow': '0px 0px 17px 0px rgba(186, 218, 212, .5)',
                'text-align': 'center',
                'margin': 'auto',
                'border-left': '#91dfd2 solid 16px',  # Border to the left
                'border-right': '#91dfd2 solid 4px',  # Border to the right
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
                        height=400, width= 900,
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
                        href="https://docs.google.com/spreadsheets/d/1e7ZZBRt37kEElHnyFF_VV_4bVRqjy3oXNw8cY1OhIcw/edit#gid=256296495",
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
                                            step=1, value=1,
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
                value= 1, 
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
                children=[
                    html.A(
                        html.Button(children="REFRESH", id="refresh-button"),
                    ),]
                # children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
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
            interval= 60 * 1000,  # update every 1 minutes
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




# Callback to update df on "REFRESH" button click
@app.callback(
    Output("value-setter-store", "data"),  # You can use this to trigger updates, it doesn't need to return anything
    [Input("refresh-button", "n_clicks")],
    prevent_initial_call=True,
)
def refresh_data(n_clicks):
    global df  # Ensure df is treated as a global variable
    if n_clicks > 0:
        # Call the load_data() function to refresh the data and update df
        appended_table = extract_and_append_all_tables(sheet_key)
        transformed_data = transform_data(appended_table, company_name, country_name)
        load_to_google_sheets(transformed_data, google_sheet_url, sheet_name)
        df = load_data(gsheet_url)
        

    # Return any data (can be None) to trigger the update
    return None




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
    [Input("app-tabs", "value"), Input("interval-component", "n_intervals")],
    [State("n-interval-stage", "data")],
)
def render_tab_content(tab_switch, interval_n, stopped_interval):
    if interval_n > stopped_interval:
        # If the interval has passed the last stopped interval, update the data
        # Add your data update logic here
        load_data()

    if tab_switch == "tab1":
        return build_tab_1(), interval_n
    return (
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=[build_top_panel(interval_n), build_chart_panel()],
                ),
            ],
        ),
        interval_n,
    )

# ======= update progress gauge =========




# Running the server
if __name__ == "__main__":
    app.run_server(debug=False, port=8051)
