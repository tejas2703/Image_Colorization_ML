from graphviz import Digraph

# Create a new directed graph
flowchart = Digraph(comment='Nature Navigator Flowchart')

# Define nodes
flowchart.node('A', 'Telemetry Data (CSV)')
flowchart.node('B', 'Data Preprocessing')
flowchart.node('C', 'Feature Engineering')
flowchart.node('D', 'Model Training (Random Forest)')
flowchart.node('E', 'Prediction Engine')
flowchart.node('F', 'Predicted Locations')
flowchart.node('G', 'Visualization (Map)')

# Define edges (connections between nodes)
flowchart.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# Render the flowchart
flowchart.render('nature_navigator_flowchart', view=True, format='png')
