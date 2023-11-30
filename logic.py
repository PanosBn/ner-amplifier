import string


def generate_synthetic_dataset(file: str):
    output = str(file)
    with open(f"data/from_json/{file}") as file:
        lines = file.readlines()
        text = ""
        rels = ""
        synthetic_data = list()
        for line in lines:
            # if line.startswith():
            if line.startswith("# text"):
                text = line
                synthetic_data.append("\n")
                synthetic_data.append(text)
                # synthetic_data.append("\n")
            if line.startswith("# relation"):
                rels = line
                synthetic_data.append(rels)
                # synthetic_data.append("\n")
                # print(line)
            if line.startswith(tuple(string.digits)):
                idx, word, tag = line.split("\t")
                # print(line)
                oov = False  # out of vocabulary word -Flag
                pos_tag = [token.pos_ for token in temp_pos_tagger(word)][
                    0
                ]  # check the POS tag of the word

                if not (
                    tag.startswith("S-PERSON")
                    or tag.startswith("I-PERSON")
                    or tag.startswith("O")
                    or tag.startswith("I-")
                ):
                    # print(line)
                    # print(tag)
                    try:
                        query = model.most_similar(word, topn=5)
                        replacement = random.choice(query)
                        oov = False
                        # print(f'New word {replacement}')
                    except Exception as ex:
                        # print(f'An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}')
                        oov = True
                    if oov:
                        try:
                            query = kpnw2v.wv.most_similar(word, topn=5)
                            # print(f' Case KPN word2vec {query}')
                            replacement = random.choice(query)
                            oov = False
                            # print(f'New word {replacement}')
                        except Exception as ex:
                            print(
                                f"KPN w2v An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}. \nWord: {word}. \nTag:{tag}"
                            )
                            oov = True
                        # print(replacement)
                    if oov:
                        line = ("\t").join([idx, word, tag])
                    else:
                        line = ("\t").join([idx, replacement[0], tag])
                elif pos_tag.startswith("NOUN") and tag.startswith("O"):
                    line = line
                    try:
                        query = model.most_similar(word, topn=5)
                        replacement = random.choice(query)
                        oov = False
                        # print(f'New word {replacement}')
                    except Exception as ex:
                        # print(f'An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}')
                        oov = True
                    if oov:
                        line = ("\t").join([idx, word, tag])
                    else:
                        line = ("\t").join([idx, replacement[0], tag])
                else:
                    line = line
                synthetic_data.append(line)
                # synthetic_data.append("\n")

        # print(synthetic_data)
        with open(
            f"data/from_json/synthetic_L_{output}", "w", encoding="utf-8"
        ) as outfile:
            outfile.write("".join(synthetic_data))
