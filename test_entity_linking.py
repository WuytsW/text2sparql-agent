"""
Quick comparison of the three entity linking backends.
Usage:  python test_entity_linking.py
        python test_entity_linking.py "Your custom question here"
"""
import sys
import time

from services.context_utils.entity_linking_dbpedia import (
    dbpedia_el,
    spotlight_external,
)
dbpedia_el_falcon = dbpedia_el  # alias: Falcon is used internally by dbpedia_el

DEFAULT_NLQ = "What is the time zone of Salt Lake City?"
DEFAULT_ENTITIES = ["Salt Lake City", "Time Zone"]


def run(label: str, fn, *args):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        print(f"  OK  ({elapsed:.2f}s)")
        if isinstance(result, list):
            for item in result:
                print(f"    {item}")
        else:
            print(f"    {result}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  FAIL ({elapsed:.2f}s): {e}")


if __name__ == "__main__":
    nlq = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NLQ
    entities = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_ENTITIES

    print(f"\nNLQ:      {nlq}")
    print(f"Entities: {entities}")

    run("1. SPARQL rdfs:label lookup  (dbpedia_el)", dbpedia_el, nlq, entities)
    run("2. Falcon 2.0               (dbpedia_el_falcon)", dbpedia_el_falcon, nlq, entities)
    run("3. DBpedia Spotlight         (spotlight_external)", spotlight_external, nlq)
