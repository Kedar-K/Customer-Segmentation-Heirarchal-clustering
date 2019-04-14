from flask import (
    Flask,
    render_template,
    flash,
    redirect,
    url_for,
    session,
    request,
    logging,
)

import sys
import os
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
import pandas as pd
import pickle
import copy
import csv
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename


app = Flask(__name__)


# Configuration of MySQL
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "new-password"
app.config["MYSQL_DB"] = "clustering"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"
# MYSQL init
mysql = MySQL(app)

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values
model = pickle.load(open("model.pkl", "rb"))
new_X = copy.copy(dataset)


@app.route("/")
def hello():
    return render_template("home.html")


@app.route("/use")
def how():
    return render_template("how.html")


@app.route("/prev")
def prev():
    return render_template("prev.html")


# user login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        cur = mysql.connection.cursor()

        result = cur.execute(
            "SELECT * FROM CLUSTERING WHERE username = %s and password = %s",
            [username],
            [password],
        )

        if result > 0:
            flash("1")
        else:
            flash("0")
    return render_template("login.html")


@app.route("/upload")
def upload_file():
    return render_template("upload.html")


@app.route("/uploader", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        f = request.files["file"]
        f.save(secure_filename(f.filename))
        prediction = model.fit_predict(X)
        # Create new column and save the cluster number to it
        new_X["cluster"] = prediction

        # Get Hign Income Low Spending in seperate list
        """
        .values is necessary for porpouse of Converting it ti the csv 
        file so dont modify that
        """
        highIncomeLowSpending = new_X.loc[new_X["cluster"] == 0].values
        cluster2 = new_X.loc[new_X["cluster"] == 1].values
        cluster3 = new_X.loc[new_X["cluster"] == 2].values
        cluster4 = new_X.loc[new_X["cluster"] == 3].values
        cluster5 = new_X.loc[new_X["cluster"] == 4].values

        # Convert it into csv file
        with open("outputs/hILS.csv", "w", newline="") as fp:
            a = csv.writer(fp, delimiter=",")
            a.writerow(
                [
                    "CustomerID",
                    "Genre",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                    "cluster_no",
                ]
            )
            a.writerows(highIncomeLowSpending)
            # a.writerow(highIncomeLowSpending[:-1])

        with open("outputs/cluster2.csv", "w", newline="") as fp:
            a = csv.writer(fp, delimiter=",")
            a = csv.writer(fp, delimiter=",")
            a.writerow(
                [
                    "CustomerID",
                    "Genre",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                    "cluster_no",
                ]
            )
            a.writerows(cluster2)

        with open("outputs/cluster3.csv", "w", newline="") as fp:
            a = csv.writer(fp, delimiter=",")
            a = csv.writer(fp, delimiter=",")
            a.writerow(
                [
                    "CustomerID",
                    "Genre",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                    "cluster_no",
                ]
            )
            a.writerows(cluster3)

        with open("outputs/cluster4.csv", "w", newline="") as fp:
            a = csv.writer(fp, delimiter=",")
            a = csv.writer(fp, delimiter=",")
            a.writerow(
                [
                    "CustomerID",
                    "Genre",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                    "cluster_no",
                ]
            )
            a.writerows(cluster4)

        with open("outputs/cluster5.csv", "w", newline="") as fp:
            a = csv.writer(fp, delimiter=",")
            a = csv.writer(fp, delimiter=",")
            a.writerow(
                [
                    "CustomerID",
                    "Genre",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                    "cluster_no",
                ]
            )
            a.writerows(cluster5)

        # Visualising the clusters
        plt.scatter(
            X[prediction == 0, 0],
            X[prediction == 0, 1],
            s=100,
            c="red",
            label="Cluster 1",
        )

        plt.scatter(
            X[prediction == 1, 0],
            X[prediction == 1, 1],
            s=100,
            c="blue",
            label="Cluster 2",
        )

        plt.scatter(
            X[prediction == 2, 0],
            X[prediction == 2, 1],
            s=100,
            c="green",
            label="Cluster 3",
        )

        plt.scatter(
            X[prediction == 3, 0],
            X[prediction == 3, 1],
            s=100,
            c="cyan",
            label="Cluster 4",
        )

        plt.scatter(
            X[prediction == 4, 0],
            X[prediction == 4, 1],
            s=100,
            c="magenta",
            label="Cluster 5",
        )

        plt.title("Clusters of customers")
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.legend()
        plt.savefig("outputs/highincome-lowspending.pdf")
        plt.savefig("outputs/highincome-lowspending.png")
        plt.savefig("static/highincome-lowspending.png")
        # return render_template("/templates/latest_results.html")
        return render_template("/results.html")


# For debugging porpouses dont change it
'''
@app.route("/templates/latest_results")
def results():
    prediction = model.fit_predict(X)
    # Create new column and save the cluster number to it
    new_X["cluster"] = prediction

    # Get Hign Income Low Spending in seperate list
    """
    .values is necessary for porpouse of Converting it ti the csv 
    file so dont modify that
    """
    highIncomeLowSpending = new_X.loc[new_X["cluster"] == 0].values
    cluster2 = new_X.loc[new_X["cluster"] == 1].values
    cluster3 = new_X.loc[new_X["cluster"] == 2].values
    cluster4 = new_X.loc[new_X["cluster"] == 3].values
    cluster5 = new_X.loc[new_X["cluster"] == 4].values

    # Convert it into csv file
    with open("outputs/hILS.csv", "w", newline="") as fp:
        a = csv.writer(fp, delimiter=",")
        a.writerow(
            [
                "CustomerID",
                "Genre",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "cluster_no",
            ]
        )
        a.writerows(highIncomeLowSpending)
        # a.writerow(highIncomeLowSpending[:-1])

    with open("outputs/cluster2.csv", "w", newline="") as fp:
        a = csv.writer(fp, delimiter=",")
        a = csv.writer(fp, delimiter=",")
        a.writerow(
            [
                "CustomerID",
                "Genre",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "cluster_no",
            ]
        )
        a.writerows(cluster2)

    with open("outputs/cluster3.csv", "w", newline="") as fp:
        a = csv.writer(fp, delimiter=",")
        a = csv.writer(fp, delimiter=",")
        a.writerow(
            [
                "CustomerID",
                "Genre",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "cluster_no",
            ]
        )
        a.writerows(cluster3)

    with open("outputs/cluster4.csv", "w", newline="") as fp:
        a = csv.writer(fp, delimiter=",")
        a = csv.writer(fp, delimiter=",")
        a.writerow(
            [
                "CustomerID",
                "Genre",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "cluster_no",
            ]
        )
        a.writerows(cluster4)

    with open("outputs/cluster5.csv", "w", newline="") as fp:
        a = csv.writer(fp, delimiter=",")
        a = csv.writer(fp, delimiter=",")
        a.writerow(
            [
                "CustomerID",
                "Genre",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "cluster_no",
            ]
        )
        a.writerows(cluster5)

    # Visualising the clusters
    plt.scatter(
        X[prediction == 0, 0], X[prediction == 0, 1], s=100, c="red", label="Cluster 1"
    )

    plt.scatter(
        X[prediction == 1, 0], X[prediction == 1, 1], s=100, c="blue", label="Cluster 2"
    )

    plt.scatter(
        X[prediction == 2, 0],
        X[prediction == 2, 1],
        s=100,
        c="green",
        label="Cluster 3",
    )

    plt.scatter(
        X[prediction == 3, 0], X[prediction == 3, 1], s=100, c="cyan", label="Cluster 4"
    )

    plt.scatter(
        X[prediction == 4, 0],
        X[prediction == 4, 1],
        s=100,
        c="magenta",
        label="Cluster 5",
    )

    plt.title("Clusters of customers")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.savefig("outputs/highincome-lowspending.pdf")
    plt.savefig("outputs/highincome-lowspending.png")
    return render_template("/templates/latest_results.html")
'''

if __name__ == "__main__":
    app.run(debug=True)
