import re
import random
from pomegranate import *

# Setting Damping constants and sampling limit
DAMPING = 0.85
SAMPLING = 10000


def read(file):
    pages = dict()
    # Iterating over files inside the corpus
    for file_name in os.listdir(file):
        # If the file is not of html format
        if not file_name.endswith('.html'):
            continue
        # Reading the contents inside the html file
        with open(os.path.join(file, file_name)) as f:
            links = re.findall(r"<a\s*href=\"([^>]*)\"", f.read())
            pages[file_name] = set(links) - {file_name}

    # Checking that only available links are selected
    for page in pages:
        pages[page] = set(
            link for link in pages[page]
            if link in pages
        )
    return pages


def transitions(corpus, page):
    table = []
    # PD for links within a page
    for web_page in corpus:
        for link in corpus[web_page]:
            table.append([web_page, link, DAMPING / len(corpus[web_page])])
    # PD for randomly choosing any of the page
    for web_page in corpus:
        for web_page_copy in corpus:
            table.append([web_page, web_page_copy, (1 - DAMPING) / len(corpus)])
    trans = ConditionalProbabilityTable(table, [page])
    return trans


def sample_page_rank(corpus):
    # Storing the initial probabilities to a dictionary
    dictionary = dict()
    for key in corpus:
        dictionary[key] = 1 / len(corpus)

    # Declaring the initial state using the dictionary
    state = DiscreteDistribution(dictionary)
    trans = transitions(corpus, state)
    model = MarkovChain([state, trans])
    outcomes = model.sample(SAMPLING)

    # Creating a dictionary of page rank and returning the same based on the
    # outcome
    page_rank = dict()
    for page in corpus:
        page_rank[page] = outcomes.count(page) / SAMPLING
    return page_rank


def normalize(page_rank):
    sum = 0
    for value in page_rank.values():
        sum += value
    alpha = 1 / sum

    # Changing each value to obtain the sum 1
    for key in page_rank:
        page_rank[key] *= alpha


def iteration_method(corpus):
    page_rank = dict()

    # Setting equal initial probabilities
    for page in corpus:
        page_rank[page] = 1 / len(corpus)

    # Iterating until convergence
    difference = [1, 1, 1, 1, 1, 1, 1]
    while True:
        state = random.choice(list(corpus))
        # if there are links present
        if corpus[state]:
            links = corpus[state]
        else:
            links = list(corpus)

        # Iterating over each link to modify the rank of the same link using
        # current page rank
        for link in links:
            copy = page_rank[link]
            page_rank[link] = ((1 - DAMPING) / len(corpus)) + \
                              (DAMPING * page_rank[state] / len(links))
            difference.pop()
            difference.insert(0, copy - page_rank[link])

        if all(x < 0.001 for x in difference):
            break
    normalize(page_rank)
    return page_rank


def main():
    # Storing the links found in each file in the corpus into a dictionary
    corpus = read('corpus1')

    # Using sampling method
    page_rank = sample_page_rank(corpus)
    print('PAGE RANK USING SAMPLING METHOD')
    for page in corpus:
        print('{}: {}'.format(page, page_rank[page]))
    print()

    # Using iteration method
    page_rank = iteration_method(corpus)
    print('PAGE RANK USING ITERATION METHOD')
    for page in corpus:
        print('{}: {}'.format(page, round(page_rank[page], 4)))


if __name__ == '__main__':
    main()
