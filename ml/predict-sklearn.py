#!/usr/bin/env python3

import json
import sys

from joblib import load

if __name__ == "__main__":
    model = load(sys.argv[1])
    v = load(sys.argv[2])

    for line in sys.stdin:
        (sid, e1, e2, label, feature_json) = line.strip("\n").split("\t")
        vectors = v.transform(json.loads(feature_json))
        prediction = model.predict(vectors)

        if prediction != "null":
            print(sid, e1, e2, prediction[0], sep="|")
