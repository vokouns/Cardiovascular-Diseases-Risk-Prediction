from flask import Flask, jsonify
import requests

app = Flask(__name__)

# Define the URL of the GeoJSON file
geojson_url = "https://colab.research.google.com/drive/1MHIfkMSeeiNonoM3OPAKGskSE_ihY5ku#scrollTo=vcm-54RLuvdb"

@app.route('/api/geojson', methods=['GET'])
def get_geojson():
    # Fetch the GeoJSON data from the web address
    response = requests.get(geojson_url)
    response.raise_for_status()  # Raise an error if the request failed

    # Return the GeoJSON content
    geojson_data = response.json() 
    return jsonify(geojson_data)

if __name__ == '__main__':
    app.run(debug=True)