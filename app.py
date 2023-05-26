from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load Data
    data_cluster = pd.read_csv("https://storage.googleapis.com/tim_panel1/DataCoba.csv")
    data_cluster = data_cluster[['Volume Produksi','Konsumsi']]
    
    # Feature Scaling
    sc_data_cluster = StandardScaler()
    data_cluster_std = sc_data_cluster.fit_transform(data_cluster.astype(float))
    
    # Clustering with KMeans
    kmeans = KMeans(n_clusters=4, random_state=42).fit(data_cluster_std)
    labels = kmeans.labels_
    
    # Add labels to a new column
    ikan = pd.read_csv("https://storage.googleapis.com/tim_panel1/DataCoba.csv")
    ikan['Cluster'] = labels

    # Data Frame Manipulation
    condition = [
        (ikan['Cluster'] == 0),
        (ikan['Cluster'] == 1),
        (ikan['Cluster'] == 2),
        (ikan['Cluster'] == 3)
    ]
    values = ['Kurang disarankan untuk ditangkap/dibudidaya', 
              'Bisa ditangkap/dibudidaya', 
              'Sangat disarankan untuk ditangkap/dibudidaya', 
              'Bisa ditangkap/dibudidaya' ]
    ikan['Status'] = np.select(condition, values)
    ikan = ikan.sort_values(by=['Status'])

    # Take one Region
    outputkab = ikan[['Kabupaten/Kota', 'Jenis Ikan', 'Status']].copy()

    # Limit the output to a maximum of 10 rows
    outputkab = outputkab.head(10)

    # Convert output to HTML table
    outputkab_html = outputkab.to_html(index=False)

    return render_template('index2.html', outputkab=outputkab_html)

if __name__ == '__main__':
    app.run(debug=True)
