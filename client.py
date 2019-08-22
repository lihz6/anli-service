from bert_serving.client import BertClient

bc = BertClient()
bc.encode(["First do it", "then do it right", "then do it better"])


def cossim(nda, ndb):
    from numpy import dot
    from numpy.linalg import norm

    return dot(nda, ndb) / (norm(nda) * norm(ndb))


if __name__ == "__main__":
    a = "你好。"
    b = "我听一下。"
    c = "你听我讲"
    with BertClient() as bc:
        nda = bc.encode([a])
        ndb = bc.encode([b])
        ndc = bc.encode([c])
        print(cossim(nda[0], ndb[0]))
        print(cossim(ndc[0], ndb[0]))

