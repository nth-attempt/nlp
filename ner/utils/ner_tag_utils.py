def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag.value == "O":
            continue
        split = tag.value.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1].value == "O":  # conversion IOB1 to IOB2
            tags[i].value = "B" + tag.value[1:]
        elif tags[i - 1].value[1:] == tag.value[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i].value = "B" + tag.value[1:]
    return True


# spacy bilou
# from spacy.gold import biluo_tags_from_offsets, offsets_from_bilou_tags


def bio2biluo(tags):
    """
    BIO -> BILUO
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("B-", "U-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("I-", "L-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


import spacy

print(spacy.__version__)
# spacy bilou
from spacy.gold import biluo_tags_from_offsets, offsets_from_biluo_tags

from spacy.lang.en import English

nlp = English()

text = "Can I buy aapl 23s @ $100?\nLooking for 10mm!"
spans = [(10, 14, "Ticker"), (15, 18, "Maturity"), (21, 25, "Price"), (39, 43, "Volume")]
bio_labels = [
    "O",
    "O",
    "O",
    "B-Ticker",
    "B-Maturity",
    "O",
    "B-Price",
    "I-Price",
    "O",
    "O",
    "O",
    "O",
    "B-Volume",
    "I-Volume",
    "O",
]
biluo_labels = [
    "O",
    "O",
    "O",
    "U-Ticker",
    "U-Maturity",
    "O",
    "B-Price",
    "L-Price",
    "O",
    "O",
    "O",
    "O",
    "B-Volume",
    "L-Volume",
    "O",
]
doc = nlp(text)

biluo_pred = bio2biluo(bio_labels)

assert biluo_labels == biluo_pred


tags = biluo_tags_from_offsets(doc, spans)
for i, token in enumerate(doc):
    print(repr(token.text), tags[i], bio_labels[i])

