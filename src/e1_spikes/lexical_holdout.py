from __future__ import annotations


# Independent, frozen evaluation set. It is never imported by training code.
CELL_PAIRS = {
    ("similar", "similar"): [
        ("felino", "felinos"), ("automóvil", "automóviles"),
        ("vehículo", "vehículos"), ("ligero", "ligeros"),
        ("vivienda", "viviendas"), ("lluvia", "lluvias"),
        ("tormenta", "tormentas"), ("árbol", "árboles"),
        ("hogar", "hogares"), ("can", "canes"),
        ("paró", "pararon"), ("frenó", "frenaron"),
    ],
    ("similar", "different"): [
        ("felino", "felipe"), ("automóvil", "autómata"),
        ("vehículo", "vesícula"), ("ligero", "liguero"),
        ("vivienda", "vivencia"), ("lluvia", "lidia"),
        ("tormenta", "trompeta"), ("árbol", "mármol"),
        ("hogar", "hongo"), ("can", "pan"),
        ("paró", "pagó"), ("frenó", "fregó"),
    ],
    ("different", "similar"): [
        ("perro", "felino"), ("can", "gato"),
        ("automóvil", "coche"), ("carro", "vehículo"),
        ("rápido", "ligero"), ("casa", "vivienda"),
        ("lluvia", "tormenta"), ("nube", "tormenta"),
        ("pino", "bosque"), ("árbol", "bosque"),
        ("paró", "frenó"), ("ladrido", "maullido"),
    ],
    ("different", "different"): [
        ("felino", "tormenta"), ("can", "vivienda"),
        ("automóvil", "ladrido"), ("vehículo", "bosque"),
        ("ligero", "maullido"), ("hogar", "rápido"),
        ("lluvia", "carro"), ("tormenta", "perro"),
        ("árbol", "frenó"), ("bosque", "coche"),
        ("paró", "nube"), ("maullido", "casa"),
    ],
}


def build_lexical_holdout():
    rows = []
    for (orthography, semantics), pairs in CELL_PAIRS.items():
        relation = (
            "inflection_holdout" if orthography == semantics == "similar"
            else "form_distractor_holdout" if orthography == "similar"
            else "concept_relation_holdout" if semantics == "similar"
            else "unrelated_holdout"
        )
        for left, right in pairs:
            rows.append({
                "left": left,
                "right": right,
                "orthography": orthography,
                "semantics": semantics,
                "relation": relation,
                "pos": "mixed",
                "exposure": "evaluation_only",
            })
    return rows
