import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from main import run_test_cycle, detect_anomalies, tf  # Import from main

model = tf.keras.models.load_model('anomaly_model.h5')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AeroSensorTester: Real-Time Component Test Simulator"),
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(id='graph-update', interval=1000, n_intervals=0),  # Update every sec
    html.Div(id='metrics')  # For anomaly count
])

@app.callback(
    [Output('live-graph', 'figure'), Output('metrics', 'children')],
    [Input('graph-update', 'n_intervals')]
)
def update_graph(n):
    data = run_test_cycle(50)  # Fresh batch
    anomalies = detect_anomalies(model, data)
    x = np.arange(len(data))
    trace_raw = go.Scatter(x=x, y=data.flatten(), mode='lines', name='Sensor Data', line=dict(color='blue'))
    trace_anom = go.Scatter(x=x[anomalies], y=data.flatten()[anomalies], mode='markers', name='Anomalies', marker=dict(color='red', size=10))

    fig = go.Figure(data=[trace_raw, trace_anom])
    fig.update_layout(title='Live Sensor Simulation (e.g., Vibration in Extreme Conditions)',
                      xaxis_title='Time Samples', yaxis_title='Value',
                      template='plotly_dark')  # Sleek theme

    metrics = f"Detected Anomalies: {np.sum(anomalies)} / {len(data)} | Efficiency: {100 - (np.sum(anomalies)/len(data)*100):.1f}%"
    return fig, metrics

if __name__ == '__main__':
    app.run_server(debug=True)