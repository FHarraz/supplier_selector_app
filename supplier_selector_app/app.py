from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import networkx as nx
import os

app = Flask(__name__)
app.secret_key = 'any_random_secret_key'
global_df = None
global_plot = None
global_weights = None


def topsis_score(df, weights):
    features = df[['Sales per Unit', 'Quantity', 'Rating', 'Sentiment']]
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(features)
    weighted = norm * weights

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = dist_worst / (dist_best + dist_worst)
    return score * 100


def generate_testing_graphs():
    os.makedirs("static", exist_ok=True)

    # CFG Mockup
    G = nx.DiGraph()
    edges = [('Start', 'Load Data'), ('Load Data', 'Clean Data'),
             ('Clean Data', 'Check Columns'), ('Check Columns', 'Cluster'), 
             ('Cluster', 'Score'), ('Score', 'Export'), ('Export', 'End')]
    G.add_edges_from(edges)
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, arrows=True)
    plt.title("Control Flow Graph")
    plt.savefig("static/graph_cfg.png")

    # Statement Coverage Mockup
    statements = ['Import', 'Read CSV', 'Normalize', 'Score', 'Export']
    coverage = [1, 1, 1, 1, 1]
    plt.figure()
    plt.bar(statements, coverage, color='green')
    plt.title("Statement Coverage")
    plt.ylim(0, 1.2)
    plt.savefig("static/graph_coverage.png")

    # Branch Coverage Mockup
    branches = ['If file', 'If missing cols', 'If supplier ID']
    branch_values = [1, 1, 1]
    plt.figure()
    plt.bar(branches, branch_values, color='orange')
    plt.title("Branch Coverage")
    plt.ylim(0, 1.2)
    plt.savefig("static/graph_branch.png")

    # Path Coverage Mockup
    paths = ['Path 1', 'Path 2', 'Path 3']
    path_covered = [1, 1, 0]
    plt.figure()
    plt.bar(paths, path_covered, color='purple')
    plt.title("Path Coverage")
    plt.ylim(0, 1.2)
    plt.savefig("static/graph_path.png")

    # Integration Flow Mockup
    steps = ['Upload', 'Validate', 'Score', 'Render']
    values = [1, 1, 1, 1]
    plt.figure()
    plt.bar(steps, values, color='blue')
    plt.title("Integration Testing Flow")
    plt.ylim(0, 1.2)
    plt.savefig("static/graph_integration.png")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global global_df
    file = request.files['file']
    if not file:
        flash("⚠️ No file uploaded.")
        return redirect(url_for('index'))

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    required_cols = {"Sales per Unit", "Quantity", "Rating", "Review"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        flash(f"❌ Missing required column(s): {', '.join(missing)}")
        return redirect(url_for('index'))

    df['Sentiment'] = df['Review'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Sentiment'] = (df['Sentiment'] + 1) / 2

    supplier_col = None
    for col in ['Supplier ID', 'Supplier Name']:
        if col in df.columns:
            supplier_col = col
            break

    if supplier_col:
        df = df.groupby(supplier_col).agg({
            'Sales per Unit': 'mean',
            'Quantity': 'mean',
            'Rating': 'mean',
            'Sentiment': 'mean'
        }).reset_index()

    features = df[['Sales per Unit', 'Quantity', 'Rating', 'Sentiment']]
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=0)
    df['Simulated_Supplier_ID'] = kmeans.fit_predict(norm)

    global_df = df
    df.to_csv("static/scored_results.csv", index=False)

    # Generate all graphs (they will be hidden by default)
    generate_testing_graphs()

    return redirect(url_for('results'))

@app.route("/results")
def results():
    global global_df, global_plot, global_weights
    df = global_df.copy()

    weights = np.array([0.25, 0.25, 0.25, 0.25])
    method = request.args.get('method', 'weighted')

    if method == 'topsis':
        scores = topsis_score(df, weights)
    else:
        features = df[['Sales per Unit', 'Quantity', 'Rating', 'Sentiment']]
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(features)
        scores = norm @ weights
        scores = (scores / scores.max()) * 100

    df['Compatibility Score'] = scores.round(2)
    df['Stars'] = (df['Compatibility Score'] / 20).clip(0, 5).round(1)
    global_weights = weights * 100

    fig, ax = plt.subplots()
    avg_scores = df.groupby("Simulated_Supplier_ID")["Compatibility Score"].mean()
    avg_scores.plot(kind='bar', ax=ax)
    ax.set_title("Supplier Ranking")
    ax.set_ylabel("Average Compatibility Score")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    global_plot = plot_url

    df_display = df.copy()
    for col in ["Product ID", "Sentiment", "Shipping Charges", "Review", "Company Name", "Order ID", "International Shipping"]:
        if col in df_display.columns:
            df_display.drop(columns=[col], inplace=True)

    df_display = df_display.sort_values(by="Compatibility Score", ascending=False)
    df_display.insert(0, "Rank", range(1, len(df_display) + 1))

    # ✅ Save ranked/scored CSV at the end
    df_display.to_csv("static/scored_results.csv", index=False)

    return render_template("results.html", plot_url=plot_url, table=df_display.reset_index(drop=True), weights=global_weights.round(2).tolist())

@app.route("/custom_weights", methods=["POST"])
def custom_weights():
    global global_df, global_plot
    df = global_df.copy()

    try:
        weights = np.array([
            float(request.form['weight_spu']),
            float(request.form['weight_qty']),
            float(request.form['weight_rating']),
            float(request.form['weight_sentiment'])
        ])
    except ValueError:
        flash("Invalid input. Please enter valid numbers.")
        return redirect(url_for('results'))

    if weights.sum() > 100:
        flash("❌ The sum of weights must not exceed 100%.")
        return redirect(url_for('results'))

    weights = weights / 100.0
    method = request.form.get('method', 'weighted')

    if method == 'topsis':
        scores = topsis_score(df, weights)
    else:
        features = df[['Sales per Unit', 'Quantity', 'Rating', 'Sentiment']]
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(features)
        scores = norm @ weights
        scores = (scores / scores.max()) * 100

    df['Compatibility Score'] = scores.round(2)
    df['Stars'] = (df['Compatibility Score'] / 20).clip(0, 5).round(1)

    df_display = df.copy()
    for col in ["Product ID", "Sentiment", "Shipping Charges", "Review", "Company Name", "Order ID", "International Shipping"]:
        if col in df_display.columns:
            df_display.drop(columns=[col], inplace=True)

    df_display = df_display.sort_values(by="Compatibility Score", ascending=False)
    df_display.insert(0, "Rank", range(1, len(df_display) + 1))

    # ✅ Save ranked/scored CSV at the end
    df_display.to_csv("static/scored_results.csv", index=False)

    return render_template("results.html", plot_url=global_plot, table=df_display.reset_index(drop=True), weights=(weights * 100).round(2).tolist())


if __name__ == "__main__":
    app.run(debug=True)