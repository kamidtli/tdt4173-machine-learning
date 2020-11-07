import os

def best_score(x):
    with open("../../experiments/{}/results.txt".format(x), "r") as file:
        sentence = file.read().split(":")
        score = sentence[1].replace("[", "").replace("]", "").split(",")[1]
        return score

def find_best_model():
    file_list = os.listdir('../../experiments')
    print("Best model is: ", sorted(file_list, key=best_score)[0])

find_best_model()
