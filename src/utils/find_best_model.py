import os

def best_score(x):
    with open("../../experiments/{}/results.txt".format(x), "r") as file:
        sentence = file.read().split(":")
        score = sentence[1].replace("[", "").replace("]", "").split(",")
        return float(score[1])

def find_best_model():
    file_list = os.listdir('../../experiments')
    file_list.remove("experiment_results.txt")
    print("Best model is: ", sorted(file_list, key=best_score)[0])

find_best_model()