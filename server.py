from flask import Flask, render_template
import pandas as pd
import pickle
import copy
import csv
import matplotlib.pyplot as plt

app = Flask(__name__)

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values
model = pickle.load(open("model.pkl", "rb"))
new_X = copy.copy(dataset)


@app.route("/")
def hello():
    return render_template("index.html")


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

    # Convert it into csv file
    with open("hILS.csv", "w", newline="") as fp:
        a = csv.writer(fp, delimiter=",")
        a.writerows(highIncomeLowSpending)

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

    ############Some Tests For Errors Please dont change code below###########
    one = X[prediction == 0, 0], X[prediction == 0, 1]
    first, second = one
    Y = pd.DataFrame(X, prediction)
    ##########################################################################

    plt.title("Clusters of customers")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.savefig("highincome-lowspending.pdf")
    plt.savefig("highincome-lowspending.png")
    return render_template("/templates/latest_results.html")


if __name__ == "__main__":
    app.run(debug=True)
