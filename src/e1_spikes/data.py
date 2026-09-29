from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass

import torch


ALPHABET = (
    " abcdefghijklmnñopqrstuvwxyzáéíóúü"
    "0123456789.,;:¿?¡!-()"
)


def normalize(text: str) -> str:
    """Normalizacion deliberadamente minima para conservar grafemas espanoles."""
    return unicodedata.normalize("NFC", text.strip().lower())


@dataclass(frozen=True)
class EncodedEvents:
    events: torch.Tensor
    stimulus_ends: torch.Tensor
    event_times: tuple[tuple[int, ...], ...]
    settling_steps: int
    mode: str


class EventEncoder:
    """Convierte grafemas en eventos one-hot; no contiene embeddings aprendidos."""

    def __init__(self, char_steps: int = 2, post_steps: int = 12, max_chars: int = 24):
        self.char_steps = char_steps
        self.post_steps = post_steps
        self.max_chars = max_chars
        self.alphabet = ALPHABET
        self.index = {char: i for i, char in enumerate(self.alphabet)}

    @property
    def channels(self) -> int:
        return len(self.alphabet)

    @property
    def total_steps(self) -> int:
        return self.max_chars * self.char_steps + self.post_steps

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Legacy E1 encoding. This method is intentionally frozen."""
        events = torch.zeros(len(texts), self.total_steps, self.channels)
        for batch_index, raw_text in enumerate(texts):
            text = normalize(raw_text)[: self.max_chars]
            for char_index, char in enumerate(text):
                if char not in self.index:
                    continue
                time_index = char_index * self.char_steps
                events[batch_index, time_index, self.index[char]] = 1.0
        return events

    def schedule(self, raw_text: str) -> tuple[str, tuple[int, ...]]:
        text = normalize(raw_text)[: self.max_chars]
        times = tuple(index * self.char_steps for index in range(len(text)))
        return text, times

    def encode_explicit(
        self,
        texts: list[str],
        *,
        settling_steps: int | None = None,
        minimum_steps: int = 0,
    ) -> EncodedEvents:
        """Encode with an explicit horizon relative to each final real event.

        ``minimum_steps=self.total_steps`` is the exact legacy-parity mode.
        Otherwise the batch ends only after every example receives the same
        requested amount of stimulus-free recurrent evolution.
        """
        settling = self.post_steps if settling_steps is None else settling_steps
        if settling < 0:
            raise ValueError("settling_steps debe ser no negativo")
        schedules = [self.schedule(text) for text in texts]
        ends = [times[-1] if times else 0 for _, times in schedules]
        steps = max(max(ends, default=0) + 1 + settling, minimum_steps)
        events = torch.zeros(len(texts), steps, self.channels)
        for row, (text, times) in enumerate(schedules):
            for char, time in zip(text, times, strict=True):
                if char in self.index:
                    events[row, time, self.index[char]] = 1.0
        return EncodedEvents(
            events=events,
            stimulus_ends=torch.tensor(ends, dtype=torch.long),
            event_times=tuple(times for _, times in schedules),
            settling_steps=settling,
            mode="legacy" if steps == self.total_steps else "relative",
        )


@dataclass(frozen=True)
class Triplet:
    anchor: str
    positive: str
    negative: str
    kind: str


CONCEPTS = {
    "animal": ["perro", "can", "gato", "felino"],
    "vehiculo": ["automóvil", "carro", "coche", "vehículo"],
    "velocidad": ["rápido", "veloz", "ligero"],
    "vivienda": ["casa", "hogar", "vivienda"],
    "clima": ["lluvia", "nube", "tormenta"],
    "naturaleza": ["pino", "árbol", "bosque"],
    "accion_parar": ["paró", "se detuvo", "frenó"],
    "sonido_animal": ["ladrido", "maullido"],
}

HARD_NEGATIVES = [
    ("casa", "caso"),
    ("perro", "pero"),
    ("carro", "barro"),
    ("gato", "dato"),
    ("pino", "vino"),
    ("nube", "sube"),
    ("coche", "noche"),
    ("rápido", "rapado"),
]


def make_smoke_triplets(seed: int = 0, repeats: int = 8) -> list[Triplet]:
    """Dataset pequeno y explicito para validar que el pipeline puede aprender."""
    rng = random.Random(seed)
    groups = list(CONCEPTS.items())
    examples: list[Triplet] = []
    for _ in range(repeats):
        rng.shuffle(groups)
        for group_index, (_, words) in enumerate(groups):
            if len(words) < 2:
                continue
            anchor, positive = rng.sample(words, 2)
            other_groups = [g for i, (_, g) in enumerate(groups) if i != group_index]
            negative = rng.choice(rng.choice(other_groups))
            examples.append(Triplet(anchor, positive, negative, "semantic"))
        for anchor, negative in HARD_NEGATIVES:
            positive_candidates = next(
                (words for words in CONCEPTS.values() if anchor in words), None
            )
            if positive_candidates:
                positive = rng.choice([w for w in positive_candidates if w != anchor])
                examples.append(Triplet(anchor, positive, negative, "hard_orthographic"))
    rng.shuffle(examples)
    return examples


def split_iid(triplets: list[Triplet], validation_fraction: float = 0.2):
    cut = max(1, int(len(triplets) * (1.0 - validation_fraction)))
    return triplets[:cut], triplets[cut:]


def triplet_key(item: Triplet) -> tuple[str, str, str]:
    return item.anchor, item.positive, item.negative


def make_fixed_validation(random_size: int = 240) -> list[Triplet]:
    """Validacion IID grande, fija y sin depender de la seed de entrenamiento."""
    pool: list[Triplet] = []
    groups = list(CONCEPTS.values())
    for group_index, words in enumerate(groups):
        negatives = [word for i, group in enumerate(groups) if i != group_index for word in group]
        for anchor in words:
            for positive in words:
                if anchor == positive:
                    continue
                for negative in negatives:
                    pool.append(Triplet(anchor, positive, negative, "random_negative"))
    rng = random.Random(20260928)
    rng.shuffle(pool)
    selected = pool[: min(random_size, len(pool))]

    hard: list[Triplet] = []
    for anchor, negative in HARD_NEGATIVES:
        group = next((words for words in CONCEPTS.values() if anchor in words), None)
        if group:
            for positive in group:
                if positive != anchor:
                    hard.append(Triplet(anchor, positive, negative, "hard_orthographic"))
    return selected + hard


def make_e11_training_triplets(
    seed: int, validation: list[Triplet], repeats: int = 20
) -> list[Triplet]:
    """Muestreo IID sin tripletas exactamente iguales a las de validacion."""
    validation_keys = {triplet_key(item) for item in validation}
    candidates = make_smoke_triplets(seed=seed, repeats=repeats)
    return [item for item in candidates if triplet_key(item) not in validation_keys]


def batch_triplets(triplets: list[Triplet], batch_size: int, rng: random.Random):
    order = list(range(len(triplets)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [triplets[i] for i in order[start : start + batch_size]]
