import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# Define function to plot signal
def plot_signal(signal, first_derivative, second_derivative, first_scale, second_scale, annotations=None):
    """
    This function plots the signal, its first derivative, and second derivative with annotations.
    """
    fig = go.Figure()
    
    # Plot original signal
    fig.add_trace(go.Scatter(
        x=list(range(len(signal))),
        y=signal,
        mode='lines+markers',
        name='PPG signal',
        line=dict(color='blue')
    ))
    
    # Plot first derivative
    fig.add_trace(go.Scatter(
        x=list(range(len(first_derivative))),
        y=first_derivative * first_scale,
        mode='lines',
        name='1º Derivative',
        line=dict(color='red', width=1),
        opacity=0.7
    ))

    # Plot second derivative
    fig.add_trace(go.Scatter(
        x=list(range(len(second_derivative))),
        y=second_derivative * second_scale,
        mode='lines',
        name='2º Derivative',
        line=dict(color='green', width=1),
        opacity=0.5
    ))

    if annotations is not None:
        for point_name, x_coord in annotations.items():
            fig.add_trace(go.Scatter(
                x=[x_coord],
                y=[signal[x_coord]],
                mode='markers+text',
                name=point_name,
                text=[point_name],
                textposition="top center"
            ))

    fig.update_layout(autosize=False, width=800, height=400)
    return fig

# Create a main function
def main():
    """
    This function sets up the interface for the annotation app.
    Allows loading a CSV or Parquet file with signals and annotating the signals.
    Allows navigating through the signals and saving the annotations as a CSV or Parquet file.
    """
    st.title('PPG Signal Annotation App - BETA')
    
    # Load data
    uploaded_file = st.file_uploader("Load CSV or Parquet file", type=["csv", "parquet"])
    
    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith('.pickle'):
            df = pd.read_pickle(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload Parquet or Pickle file.")
            return

        # Get list of columns
        columns = df.columns.tolist()

        # Check if 'last_beat' is in the columns
        if 'last_beat' in columns:
            signal_column = 'last_beat'
        else:
            # If 'last_beat' is not present, allow the user to select the signal column
            signal_column = st.selectbox("Choose the column with the signal", columns)

        # Initialize session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'annotations' not in st.session_state:
            st.session_state.annotations = {}

        # Display current signal
        signal = df.iloc[st.session_state.current_index][signal_column]  # Select the signal column
        user = df.iloc[st.session_state.current_index]['id']
        identifier = df.iloc[st.session_state.current_index]['hash_id']
        
        # Modified line to include the current index
        st.write(f"Index: {st.session_state.current_index}, ID: {identifier}, Unique ID: {user}")
        
        signal_data = np.array(signal)  # Convert to numpy array
        
        # Calculate first and second derivatives
        first_derivative = np.gradient(signal_data)
        second_derivative = np.gradient(first_derivative)
        
        # Add sliders for scaling derivatives
        first_scale = st.slider("Scale for 1st Derivative", 1.0, 100.0, 10.0)
        second_scale = st.slider("Scale for 2nd Derivative", 1.0, 100.0, 10.0)
        
        fig = plot_signal(signal_data, first_derivative, second_derivative, first_scale, second_scale)

        # Selector de punto fiduciario
        fiducial_point = st.selectbox(
            "Select the fiducial point to annotate",
            ["Onset", "Systolic Peak", "Dicrotic Notch", "Diastolic Peak"]
        )

        # Display the plot and capture the selected points (clics)
        selected_points = plotly_events(
            fig,                    
            click_event=True,       # Enable capturing click events 
            hover_event=False,      # Disable capturing hover events
            select_event=False,     # Disable capturing select events
            override_height=400,    # Override the default height of the plot to 400 pixels
            override_width='100%'   # Override the default width of the plot to 100% of the container
        )

        # Initialise fiduciary points in the session state if they do not exist for this signal 
        if 'fiducial_points' not in st.session_state:
            st.session_state.fiducial_points = {
                'Onset': 0,
                'Systolic Peak': 0,
                'Dicrotic Notch': 0,
                'Diastolic Peak': 0
            }
        elif st.session_state.get('last_signal_id') != identifier:
            # if we have changed the signal, reset the fiducial points
            st.session_state.fiducial_points = {
                'Onset': 0,
                'Systolic Peak': 0,
                'Dicrotic Notch': 0,
                'Diastolic Peak': 0
            }

        # Save the last signal id
        st.session_state.last_signal_id = identifier

        # Process the selected points
        if selected_points:
            # Obtain the x-coordinate of the clicked point
            clicked_point = selected_points[0]['x']
            # update the fiducial point with the clicked point
            st.session_state.fiducial_points[fiducial_point] = clicked_point

        # Save the annotations
        st.write("Write the index of the fiducial points manually if needed:")
        onset = st.number_input(
            "Onset (índice)",
            min_value=0,
            max_value=len(signal_data)-1,
            value=st.session_state.fiducial_points['Onset'],
            key='onset_input'
        )
        systolic_peak = st.number_input(
            "Systolic Peak (índice)",
            min_value=0,
            max_value=len(signal_data)-1,
            value=st.session_state.fiducial_points['Systolic Peak'],
            key='systolic_peak_input'
        )
        dicrotic_notch = st.number_input(
            "Dicrotic Notch (índice)",
            min_value=0,
            max_value=len(signal_data)-1,
            value=st.session_state.fiducial_points['Dicrotic Notch'],
            key='dicrotic_notch_input'
        )
        diastolic_peak = st.number_input(
            "Diastolic Peak (índice)",
            min_value=0,
            max_value=len(signal_data)-1,
            value=st.session_state.fiducial_points['Diastolic Peak'],
            key='diastolic_peak_input'
        )

        # Actualizar los valores en st.session_state al cambiar manualmente
        st.session_state.fiducial_points['Onset'] = onset
        st.session_state.fiducial_points['Systolic Peak'] = systolic_peak
        st.session_state.fiducial_points['Dicrotic Notch'] = dicrotic_notch
        st.session_state.fiducial_points['Diastolic Peak'] = diastolic_peak
        
        # Save annotations
        if st.button("Save Annotations"):
            st.session_state.annotations[identifier] = {
                'Onset': onset,
                'Systolic Peak': systolic_peak,
                'Dicrotic Notch': dicrotic_notch,
                'Diastolic Peak': diastolic_peak
            }
            st.success("Anotaciones guardadas.")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        if col1.button("Previous Signal"):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                # Reset fiducial points for new signal
                st.session_state.fiducial_points = {
                    'Onset': 0,
                    'Systolic Peak': 0,
                    'Dicrotic Notch': 0,
                    'Diastolic Peak': 0
                }
        if col2.button("Next Signal"):
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
                # Reset fiducial points for new signal
                st.session_state.fiducial_points = {
                    'Onset': 0,
                    'Systolic Peak': 0,
                    'Dicrotic Notch': 0,
                    'Diastolic Peak': 0
                }
        
        # Export annotations
        if st.button("Exportar Anotaciones"):
            annotations_df = pd.DataFrame.from_dict(st.session_state.annotations, orient='index')
            csv = annotations_df.to_csv().encode('utf-8')
            st.download_button(
                label="Descargar anotaciones como CSV",
                data=csv,
                file_name='anotaciones.csv',
                mime='text/csv',
            )
            # Export to Parquet if needed
            # Note: st.download_button doesn't support binary files directly for Parquet
            # You might need to use an alternative method or export only as CSV

# Run the main function
if __name__ == '__main__':
    main()
