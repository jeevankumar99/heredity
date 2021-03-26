import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # to easily access the father/mother and notfather/notmother values
    prob_on_type = {0: 0.01, 1: 0.50, 2: 0.99}
    # to keep in track each person's individual probability
    probabilities = {}
    # a dict that keeps track of a person's gene copies and traits
    gene_type = sort_gene_type(people, one_gene, two_genes, have_trait)

    for person in people:
        # No parents, if father is None then mother is also None
        if people[person]["father"] == None:
            probabilities[person] = PROBS["gene"][gene_type[person]["type"]] * PROBS["trait"][gene_type[person]["type"]][gene_type[person]["have_trait"]]
        # Person has parents
        else:
            # assings father and mother a values of 0.01 or 0.49 or 0.99 based on gene copies
            father = prob_on_type[gene_type[people[person]["father"]]["type"]]
            mother = prob_on_type[gene_type[people[person]["mother"]]["type"]]
            # not father and not mother gets the opposite
            not_father = 1 - father
            not_mother = 1 - mother
            # for 0 copies, prob = (NotFather And NotMother)
            if gene_type[person]["type"] == 0:
                probabilities[person] = not_father * not_mother
            # for 1 copy, prob = (Father And NotMother) Or (NotFather and Mother)
            elif gene_type[person]["type"] == 1:
                probabilities[person] = (father * not_mother) + (not_father * mother)
            # for 2 copies, prob = (Father And Mother)
            else:
                probabilities[person] = father * mother
            probabilities[person] *= PROBS["trait"][gene_type[person]["type"]][gene_type[person]["have_trait"]]

    # to multiply all individual probabilities
    joint_probs = 1
    for person in probabilities:
        joint_probs *= probabilities[person]

    return joint_probs


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # gene_type just maps the traits and gene copies to person
    gene_type = sort_gene_type(probabilities, one_gene, two_genes, have_trait)
    for person in probabilities:
        probabilities[person]["gene"][gene_type[person]["type"]] += p
        probabilities[person]["trait"][gene_type[person]["have_trait"]] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # genes, (gene divided by sum of all three genes)
        total = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        probabilities[person]["gene"][0] /= total
        probabilities[person]["gene"][1] /= total
        probabilities[person]["gene"][2] /= total
        # traits, (trait divided by sum of both traits)
        total = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        probabilities[person]["trait"][True] /= total
        probabilities[person]["trait"][False] /= total

def sort_gene_type(people, one_gene, two_genes, have_trait):
    """
    Maps gene copies type and trait to each person in people.
    This is just to reduce if statements in the functions.
    """
    gene_type = {}
    for person in people:
        # initialize a sub dict for every person
        gene_type[person] = {}
        # to check if person has trait
        if person in have_trait:
            gene_type[person]["have_trait"] = True
        else:
            gene_type[person]["have_trait"] = False
        # to check how many copies of gene a person has.
        if person in one_gene:
            gene_type[person]["type"] = 1
        elif person in two_genes:
            gene_type[person]["type"] = 2
        else:
            gene_type[person]["type"] = 0
    return gene_type

if __name__ == "__main__":
    main()
