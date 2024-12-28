import io

from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server-side rendering
from matplotlib import pyplot as plt
import plotly.graph_objects as go

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for frontend development


# Route to upload and process the CSV
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = f'uploads/{file.filename}'
    file.save(file_path)

    try:
        # Read and process the CSV
        df = pd.read_csv(file_path)
        processed_data, discarded_rows = process_data(df)

        return jsonify({
            "data": processed_data,
            "discarded_rows": discarded_rows
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Route to get yearly data with averages and ±1σ (standard deviation)
@app.route('/get_yearly_avg_and_std', methods=['GET'])
def get_yearly_avg_and_std():
    try:
        # Process data to calculate averages and standard deviations
        processed_data, _ = process_data(
            pd.read_csv('uploads/your_data_file.csv'))  # Replace with dynamic file path if needed

        # Prepare the response with mean temperature and std deviation (±1σ)
        yearly_data = [{
            "year": data['year'],
            "mean_temp": data['mean_temp'],
            "std_dev_upper": data['mean_temp'] + data['std_temp'],
            "std_dev_lower": data['mean_temp'] - data['std_temp']
        } for data in processed_data]

        return jsonify(yearly_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Route to get data for a specific year (zoom functionality)
@app.route('/get_data_for_year/<int:year>', methods=['GET'])
def get_data_for_year(year):
    try:
        # Process data to calculate averages and standard deviations
        processed_data, _ = process_data(
            pd.read_csv('uploads/your_data_file.csv'))  # Replace with dynamic file path if needed

        # Find the data for the requested year
        year_data = next((data for data in processed_data if data['year'] == year), None)

        if not year_data:
            return jsonify({"error": "Year not found"}), 404

        return jsonify({
            "year": year_data['year'],
            "monthly_temps": year_data['monthly_temps'],
            "mean_temp": year_data['mean_temp'],
            "std_dev_upper": year_data['mean_temp'] + year_data['std_temp'],
            "std_dev_lower": year_data['mean_temp'] - year_data['std_temp']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/plot', methods=['POST'])
def plot_data():
    data = request.json
    view_type = data.get('viewType', 'monthly')  # 'monthly' or 'yearly'
    processed_data = data.get('processedData', [])

    # Generate the plot based on view type
    if view_type == 'yearly':
        return generate_yearly_plot(processed_data)
    else:
        return generate_monthly_plot(processed_data)


def process_data(df):
    monthly_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    result = []
    discarded_rows = 0

    for _, row in df.iterrows():
        # Handle missing data (null or NaN values)
        monthly_temps = row[monthly_columns].apply(pd.to_numeric, errors='coerce').values
        if np.any(np.isnan(monthly_temps)):
            discarded_rows += 1
            continue

        # Calculate the mean and standard deviation
        mean_temp = np.mean(monthly_temps)
        std_temp = np.std(monthly_temps)

        result.append({
            'year': int(row['Year']),
            'monthly_temps': monthly_temps.tolist(),
            'mean_temp': mean_temp,
            'std_temp': std_temp
        })

    return result, discarded_rows


def generate_monthly_plot(processed_data):
    # Generate a plot for monthly data
    plt.figure(figsize=(10, 6))

    # Plot each year's monthly data
    for year_data in processed_data:
        plt.plot(year_data['monthly_temps'], label=f"Year {year_data['year']}")

    plt.title('Monthly Temperature Data')
    plt.xlabel('Months')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    # Save plot to a BytesIO buffer (in-memory)
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

def generate_yearly_plot(processed_data):
    # Ensure years are integers
    years = [int(year_data['year']) for year_data in processed_data]
    means = [year_data['mean_temp'] for year_data in processed_data]
    std_devs = [year_data['std_temp'] for year_data in processed_data]

    # Create a Plotly figure
    fig = go.Figure()

    # Add the trace for the mean temperature line
    fig.add_trace(go.Scatter(x=years, y=means, mode='lines', name='Mean Temperature'))

    # Add the area for ±1 standard deviation
    fig.add_trace(go.Scatter(
        x=years,
        y=means + np.array(std_devs),
        fill=None,  # No fill for the upper boundary
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0)')  # Invisible line
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=means - np.array(std_devs),
        fill='tonexty',  # Fill the area between this and the previous line
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0)')  # Invisible line
    ))

    # Determine how many years are present
    num_years = len(years)

    # If there are more than 20 years, just show the range
    if num_years > 20:
        start_year = years[0]
        end_year = years[-1]

        # Update x-axis to display the range instead of individual years
        fig.update_layout(
            xaxis=dict(
                tickmode='array',  # Use custom tick values
                tickvals=[start_year, end_year],  # Display only the first and last year
                ticktext=[f'{start_year}-{end_year}'],  # Label the range
                showgrid=True
            )
        )
    else:
        # If there are 20 or fewer years, display all years
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',  # Linear spacing
                tickvals=years,  # Display all years
                tickformat='d',     # Display years as integers
                showgrid=True
            )
        )

    # Set other plot options
    fig.update_layout(
        title="Yearly Temperature Averages with ±1σ",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        showlegend=True
    )

    # Save the plot as a PNG image (static rendering for frontend use)
    img_io = io.BytesIO()
    fig.write_image(img_io, format='png')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

@app.route('/plot_yearly_without_deviation', methods=['POST'])
def plot_yearly_data_without_deviation():
    processed_data = data.get('processedData', [])
    if processed_data is None:
        return jsonify({"error": "No data available. Please upload a file first."}), 400

    # Generate the yearly plot without deviation (standard deviation)
    return generate_yearly_plot_without_deviation(processed_data)


def generate_yearly_plot_without_deviation(processed_data):
    # Ensure years are integers
    years = [int(year_data['year']) for year_data in processed_data]
    means = [year_data['mean_temp'] for year_data in processed_data]

    # Create a Plotly figure
    fig = go.Figure()

    # Add the trace for the mean temperature line (no deviation)
    fig.add_trace(go.Scatter(x=years, y=means, mode='lines', name='Mean Temperature'))

    # Determine how many years are present
    num_years = len(years)

    # If there are more than 20 years, just show the range
    if num_years > 20:
        start_year = years[0]
        end_year = years[-1]

        # Update x-axis to display the range instead of individual years
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[start_year, end_year],
                ticktext=[f'{start_year}-{end_year}'],
                showgrid=True
            )
        )
    else:
        # If there are 20 or fewer years, display all years
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tickvals=years,
                tickformat='d',
                showgrid=True
            )
        )

    # Set other plot options
    fig.update_layout(
        title="Yearly Temperature Averages",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        showlegend=True
    )

    # Save the plot as a PNG image (static rendering for frontend use)
    img_io = io.BytesIO()
    fig.write_image(img_io, format='png')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)