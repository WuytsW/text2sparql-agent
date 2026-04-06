import os
from shexer.shaper import Shaper
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompts.dbpedia import shape_selection_prompt, class_instances_prompt

load_dotenv(dotenv_path=".env")


def _llm_classify_and_get_instances(label, llm, nlq):
    """
    Uses an LLM to determine if `label` is a class/type or a named entity.
    For classes: returns a list of representative DBpedia resource name strings (e.g. ["Woolly_mammoth", "Tiger"]).
    For named entities: returns an empty list, signalling fallback to dbr:Label.
    """
    prompt = class_instances_prompt["en"].format(label=label, nlq=nlq)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    if raw.startswith("CLASS"):
        lines = raw.split("\n", 1)
        if len(lines) < 2:
            return []
        return [i.strip() for i in lines[1].strip().split(",") if i.strip()]

    # ENTITY or anything unexpected → treat as named entity
    return []


def select_relevant_shape_parts(nlq, shape, llm):
    """
    Given a natural language question and a shape, use an LLM to select the most relevant properties and classes from the shape that are likely to be useful for answering the question.
    Returns a comma-separated list of relevant properties and classes from the shape.
    If the shape is empty, returns an empty string.
    """
    prompt = shape_selection_prompt["en"].format(nlq=nlq, shape=shape)
    llm_response = llm.invoke([HumanMessage(content=prompt)])
    return llm_response.content.strip()

def generate_shape(nlq, entity_labels, llm=None, use_llm=False):

    load_dotenv(dotenv_path=".env")
    dbpedia_sparql_url = os.getenv("DBPEDIA_SPARQL_URL")

    shape_lines = []

    try:
        for label in entity_labels:
            label_clean = label.replace(' ', '_')
            label_clean = label_clean[0].upper() + label_clean[1:] #Capitalize first letter to match DBpedia resource format
            shape_label = f"http://shapes.dbpedia.org/{label_clean}"

            instance_names = _llm_classify_and_get_instances(label_clean, llm, nlq) if (llm) else []
            if instance_names:
                for name in instance_names:
                    instance_uri = f"http://dbpedia.org/resource/{name}"
                    shape_lines.append(f"<{instance_uri}>@<{shape_label}>")
            else:
                # Named entity or LLM unavailable: use the resource URI directly
                entity_id = f"http://dbpedia.org/resource/{label_clean}"
                shape_lines.append(f"<{entity_id}>@<{shape_label}>")

        shape_map_raw = "\n".join(shape_lines)

        namespaces_dict = {
            "http://example.org/": "ex",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
            "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
            "http://www.w3.org/2001/XMLSchema#": "xsd",
            "http://xmlns.com/foaf/0.1/": "foaf",
            "http://dbpedia.org/resource/": "dbr",
            "http://dbpedia.org/ontology/": "dbo",
            "http://dbpedia.org/property/": "dbp",
            "http://dbpedia.org/class/yago/": "yago",
            "http://purl.org/dc/terms/": "dcterms",
            "http://www.w3.org/2002/07/owl#": "owl",
            "http://www.w3.org/2007/05/powder-s#": "powders",
            "http://www.w3.org/ns/prov#": "prov",
            "http://umbel.org/umbel/rc/": "umbel",
            "http://schema.org/": "schema",
            "http://shapes.dbpedia.org/": "shapes"
        }

        shaper = Shaper(
            shape_map_raw=shape_map_raw,
            url_endpoint=dbpedia_sparql_url,
            namespaces_dict=namespaces_dict,
            disable_comments=True,
        )

        shape = shaper.shex_graph(string_output=True)
        shape = select_relevant_shape_parts(nlq, shape, llm) if (llm and use_llm) else shape

        return shape
    except Exception:
        return None
