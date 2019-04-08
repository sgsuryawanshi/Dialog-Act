from __future__ import division
import csv
import sys
import pandas as pd
import scipy
import numpy as np
import math
import random
from random import shuffle
import matplotlib.pyplot as plt
import operator
def DialogsEstablish(Data_File):

    DictOfDialog = {}
    DictOfDialogFeatures = {}
    TotalNoInstances = 0
    TotalNoDialogs = 0

    with open(Data_File, encoding="utf8") as data:

        data.seek(0)

        for line in data:

            if line.find("Advisor") != -1:
                TotalNoInstances += 1

                dialog_act = line.split(" ")[1]
                if dialog_act.find('[') == 0:

                    if dialog_act not in DictOfDialog:
                        TotalNoDialogs += 1
                        DictOfDialog[dialog_act] = 1
                        DictOfDialogFeatures[dialog_act] = {}
                    else:
                        DictOfDialog[dialog_act] += 1

    return DictOfDialog, DictOfDialogFeatures, TotalNoInstances, TotalNoDialogs
def ParsesDataFile(Data_File, DictOfDialog, DictOfDialogFeatures):

    StripsTheList = ['.', '(', ')', ',', '-', '!', '?']

    with open(Data_File, encoding="utf8") as data:

        data.seek(0)

        previous_line = ''
        for line in data:

            if line.find("Advisor") != -1:

                dialog_act = line.split(" ")[1]
                if dialog_act.find('[') == 0:

                    features = previous_line.strip('\n')
                    if features.find("Student:") == 0:

                        features_split = features.split(' ')
                        for ID in range(1, len(features_split)):

                            for StripsTheItem in StripsTheList :
                                features_split[ID] = features_split[ID].strip(StripsTheItem)
                            features_split[ID] = features_split[ID].lower().strip()

                            if features_split[ID] not in DictOfDialogFeatures[dialog_act]:
                                DictOfDialogFeatures[dialog_act][features_split[ID]] = 1.0

            previous_line = line

    return DictOfDialog, DictOfDialogFeatures
def predict_instances(TestsFile, outputFile, DictOfDialog, DictOfDialogFeatures):

    StripsTheList = ['.', '(', ')', ',', '-', '!', '?']

    with open(TestsFile, encoding="utf8") as data:
        TestsCount= 0
        CorrectCount = 0

        data.seek(0)

        previous_line = ''
        for line in data:

            if line.find("Advisor") != -1:
                TestsCount+= 1

                dialog_act = line.split(" ")[1]
                true_dialog = dialog_act
                if dialog_act.find('[') == 0:

                    features = previous_line.strip('\n')
                    features_split = []
                    if features.find("Student:") == 0:

                        features_split = features.split(' ')
                        for ID in range(1, len(features_split)):

                            for StripsTheItem in StripsTheList :
                                features_split[ID] = features_split[ID].strip(StripsTheItem)
                            features_split[ID] = features_split[ID].lower().strip()

                            if features_split[ID] not in DictOfDialogFeatures[dialog_act]:
                                for dialog in DictOfDialog:
                                    DictOfDialogFeatures[dialog_act][features_split[ID]] = 0.01

                    Predicted_Probability = {}
                    for dialog in DictOfDialog:
                        Predicted_Probability[dialog] = 1.0

                        for word in features_split:

                            if (word != 'Student:'):

                                if word not in DictOfDialogFeatures[dialog]:
                                    DictOfDialogFeatures[dialog][word] = 0.01

                                Featured_Probability = DictOfDialogFeatures[dialog][word]/DictOfDialog[dialog]
                                Predicted_Probability[dialog] = Predicted_Probability[dialog] * Featured_Probability

                        dialog_prob = DictOfDialog[dialog]/sum(DictOfDialog.values())
                        Predicted_Probability[dialog] = Predicted_Probability[dialog]*dialog_prob

                    PredictedDialog = max(Predicted_Probability.items(), key=operator.itemgetter(1))[0]

                    if PredictedDialog == true_dialog:
                        CorrectCount += 1

                    outputFile.write(previous_line)
                    outputFile.write(PredictedDialog + ' ' + line)

            previous_line = line

    Accuracy = float(CorrectCount/TestsCount)
    print('Accuracy of the system is:' + str(Accuracy))

    return Accuracy

Data_File = "DialogAct.train"
TestsFile = "DialogAct.test"
outputName = "DialogAct.test.out"
outputFile = open(outputName, "w")
dialogs, features, TotalNoInstances, TotalNoDialogs = DialogsEstablish(Data_File)
print('Total number of instances:', TotalNoInstances)
print('Total number of dialogs:', TotalNoDialogs)

DictOfDialog, DictOfDialogFeatures = ParsesDataFile(Data_File, dialogs, features)

Accuracy = predict_instances(TestsFile, outputFile, DictOfDialog, DictOfDialogFeatures)
