import pandas as pd
from sklearn import preprocessing


def delim_map(delim):
    switch = {
        "comma": ",",
        "space": " "
    }
    return switch.get(delim)


def read_data(filename):
    with open(filename) as f:
        first_line = f.readline()
        config = first_line.strip().split(",")

    classPos = config[0]
    num_feat = int(config[1])

    feat_labels = ['f' + str(x) for x in range(num_feat)]
    if classPos == "classFirst":
        feat_labels.insert(0, "class")
    elif classPos == "classLast":
        feat_labels.append("class")
    else:
        raise ValueError(classPos)

    num_classes = int(config[2])
    delim = delim_map(config[3])

    rawData = pd.read_csv(filename, delimiter=delim, skiprows=1, header=None, names=feat_labels)
    labels = rawData['class']
    data = rawData.drop('class', axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    return {"data": data, "labels":labels}
