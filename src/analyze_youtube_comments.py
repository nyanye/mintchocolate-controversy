import os
import re
import json
import glob
import numpy as np
from textblob import TextBlob


def preprocess_youtube_comments(comments_dir):
    json_filenames = glob.glob(os.path.join(comments_dir, "*.json"))
    every_comments = []
    for json_filename in json_filenames:
        with open(json_filename, "r", encoding="utf-8") as comment_json:
            every_comments.extend(json.load(comment_json))
    text_only_list = [comment["commentText"] for comment in every_comments if isinstance(comment, dict)]
    return list(map(lambda string: re.sub('[^A-Za-z0-9]+', ' ', string), text_only_list))


def analyze_sentiments(sentences):
    # The polarity score is a float within the range [-1.0, 1.0]
    # -1.0 will be very negative and 1.0 will be very positive
    polarity = []
    for sentence in sentences:
        # scale polarity from [-1.0, 1.0] to [0~1]
        normalized_polarity = (TextBlob(sentence).sentiment.polarity + 1) / 2
        # remove unanalyzed neutral results
        if normalized_polarity != 0.5:
            polarity.append(normalized_polarity)
    return np.average(polarity)


if __name__ == "__main__":
    percentile_result = analyze_sentiments(preprocess_youtube_comments("./youtube_comments/")) * 100
    print("민트초코 호: %s 불호: %s" % (percentile_result, 100 - percentile_result))
