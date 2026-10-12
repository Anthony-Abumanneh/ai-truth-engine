"""Built-in benchmark over the demo corpus: the traps SciFact doesn't test.

Each case has a gold label (SUPPORT / CONTRADICT / NEI, as an expert reading
the demo corpus would answer) and the red flags a careful reviewer should
raise. Veridamine, lumorex, calvetrin, brivo tea, zantorel, nervaplex and
Halcyon are fictional.

IN-SAMPLE WARNING: these claims (or close variants) were used while building
and tuning the rules in steps 4-6. Scores here show the features work as
designed; they are not a held-out estimate. SciFact is the held-out test.

Expected flags were completed after the first run with flags that are
factually true of the (fictional) documents (n=48 and manufacturer-funded,
one study only, ...). Gold labels were not changed to match outputs.
"""
from __future__ import annotations

CASES = [
    # echo chamber + failed replication
    ("Veridamine improves sleep quality by 18%.", "CONTRADICT", {"echo_chamber", "failed_replication", "industry_funded", "small_sample"}),
    ("Veridamine is clinically proven to improve sleep.", "CONTRADICT", {"overclaimed_certainty", "echo_chamber", "failed_replication", "industry_funded", "small_sample"}),
    ("A preregistered trial of 612 adults found no significant effect of veridamine on sleep.", "SUPPORT", set()),
    # retraction
    ("Lumorex reverses memory loss in older adults by 40%.", "NEI", {"retracted", "single_study"}),
    ("The 2019 lumorex memory trial was retracted.", "SUPPORT", set()),
    # animal -> human
    ("Calvetrin extends lifespan by 30%.", "NEI", {"animal_to_human", "single_study", "small_sample"}),
    ("Calvetrin extends lifespan in mice.", "SUPPORT", {"single_study", "small_sample"}),
    # observational -> causal
    ("Drinking brivo tea prevents dementia.", "NEI", {"correlation_as_causation", "single_study"}),
    ("People who drink brivo tea daily have a lower risk of dementia.", "SUPPORT", {"single_study"}),
    # relative vs absolute risk
    ("Zantorel cuts stroke risk in half.", "SUPPORT", {"relative_risk_only", "industry_funded", "single_study"}),
    ("Zantorel reduced stroke from 0.4% to 0.2% over three years.", "SUPPORT", {"industry_funded", "single_study"}),
    # in vitro -> human
    ("Nervaplex grows new brain cells.", "NEI", {"in_vitro_to_human", "single_study"}),
    # causal attribution (business demo)
    ("The Halcyon X1 was discontinued in 2024.", "SUPPORT", set()),
    ("The Halcyon X1 was discontinued because demand declined.", "CONTRADICT", set()),
    # encyclopedia: true facts vs. well-known errors in low-quality sources
    ("Einstein won the Nobel Prize in Physics for the theory of relativity.", "CONTRADICT", set()),
    ("Einstein won the Nobel Prize in Physics in 1921.", "SUPPORT", set()),
    ("The capital of Australia is Sydney.", "CONTRADICT", set()),
    ("The capital of Australia is Canberra.", "SUPPORT", set()),
    ("Penicillin was discovered by Alexander Fleming.", "SUPPORT", set()),
    ("Penicillin was discovered by Louis Pasteur.", "CONTRADICT", set()),
    ("Python was first released in 1991.", "SUPPORT", set()),
    ("Python was first released in 1995.", "CONTRADICT", set()),
    ("The Great Wall of China is visible from the Moon with the naked eye.", "CONTRADICT", set()),
    ("The speed of light in vacuum is 150,000 kilometres per second.", "CONTRADICT", set()),
    ("Michael Collins walked on the Moon.", "CONTRADICT", set()),
    ("The Eiffel Tower is 450 metres tall.", "CONTRADICT", set()),
    ("The Nile is definitely the longest river in the world.", "NEI", {"overclaimed_certainty"}),
    # nothing in the corpus
    ("Glorbium cures baldness in three days.", "NEI", set()),
]


def load() -> list[dict]:
    return [{"id": f"builtin-{i}", "claim": c, "gold": g, "expected_flags": sorted(f), "gold_docs": []}
            for i, (c, g, f) in enumerate(CASES)]
