#!/usr/bin/env python3

"""
Yang-Mills V1:
deterministicka adaptacija polja, samointerakcije i akcije.

Zavisnosti: numpy, scipy
CSV: sedam sortiranih brojeva, bez zaglavlja.
Redosled: najstarije kolo prvo, najnovije poslednje.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/data/loto7_4700_k81.csv",
    "/data/loto7_4700_k81_loto_2972.csv",
    "/data/loto7_4700_k81_loto_plus_1728.csv",
)

WINDOW = 96
MEMORY = 6
STARTS = 12

ALPHAS = (.01, .1, 1.)

DIM = 37
IDENTITY = np.eye(7)

# Osam razmaka sa fiksnim zbirom imaju sedam
# nezavisnih dimenzija. Baza cuva ceo taj prostor.
BASIS = np.sqrt(2. / 8.) * np.cos(
    np.pi
    * np.arange(1, 8)[:, None]
    * (np.arange(8)[None, :] + .5)
    / 8.
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != 7
                or values != sorted(values)
                or len(set(values)) != 7
                or not all(1 <= v <= 39 for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna kombinacija 7/39."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    # Prazna mesta pre, izmedju i posle brojeva.
    # Zbir svih osam razmaka je 32.
    return np.diff(edges, axis=1) - 1.


def states(g):
    return np.atleast_2d(g) @ BASIS.T / 5.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def support(history):
    tree = cKDTree(gaps(history) / 5.)

    distance = tree.query(
        gaps(history) / 5.,
        k=2,
        p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(distance, .75)),
        1e-12
    )


def allowed(rows, tree, radius):
    distance = tree.query(
        gaps(rows) / 5.,
        p=1
    )[0]

    return distance <= radius + 1e-10


def commutator(a, b):
    return a @ b - b @ a


def wedge(v, anchor):
    v = np.asarray(v)

    return (
        v[..., :, None] * anchor
        - anchor[:, None] * v[..., None, :]
    )


def cayley(a):
    return np.linalg.solve(
        IDENTITY - .5 * a,
        IDENTITY + .5 * a
    )


def context(g, nonlinear=True):
    s = states(g)

    changes = np.diff(
        s, axis=0
    )[-MEMORY:]

    anchor = s[-1] / (
        1. + np.linalg.norm(s[-1])
    )

    fields = wedge(changes, anchor)

    last = fields[-1]
    previous = fields[-2]

    strength = last - previous

    if nonlinear:
        strength = (
            strength
            + commutator(previous, last)
        )

        transport = IDENTITY.copy()

        # Hronoloski uredjen proizvod:
        # noviji operator deluje poslednji.
        for field in fields:
            transport = (
                cayley(field) @ transport
            )
    else:
        # Kontrola cuva zbir, uklanja redosled
        # proizvoda i komutatorske clanove.
        transport = cayley(
            fields.sum(axis=0)
        )

    return (
        s[-1],
        changes[-1],
        anchor,
        last,
        strength,
        transport - IDENTITY,
        nonlinear
    )


def norm2(a):
    return np.sum(
        a * a, axis=(-2, -1)
    )


def inner(a, b):
    return np.sum(
        a * b, axis=(-2, -1)
    )


def features(g, ctx):
    (
        reference,
        memory,
        anchor,
        last,
        old_strength,
        transport,
        nonlinear
    ) = ctx

    delta = states(g) - reference

    candidate = wedge(delta, anchor)
    derivative = candidate - last

    interaction = commutator(
        last, candidate
    )

    if not nonlinear:
        interaction = np.zeros_like(
            interaction
        )

    strength = derivative + interaction

    coupled_old = commutator(
        candidate, old_strength
    )

    coupled_transport = commutator(
        candidate, transport
    )

    if not nonlinear:
        coupled_old = np.zeros_like(
            coupled_old
        )

        coupled_transport = np.zeros_like(
            coupled_transport
        )

    # Diskretne akcije i njihove istorijske veze.
    # To su osobine za nauceni skor cele kombinacije.
    action = np.column_stack((
        norm2(derivative),
        norm2(interaction),
        norm2(strength),
        norm2(strength - old_strength),
        inner(interaction, old_strength),
        inner(interaction, transport),
        inner(candidate, transport),
        norm2(coupled_old),
        norm2(coupled_transport)
    ))

    return np.column_stack((
        delta,
        np.abs(delta),
        delta * delta,
        delta * memory,
        action
    ))


def prepare(history, nonlinear=True):
    needed = WINDOW + MEMORY + 1

    if len(history) < needed:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{needed} prethodnih redova."
        )

    h = history[-needed:]
    g = gaps(h)

    gram = np.zeros((DIM, DIM))
    rhs = np.zeros(DIM)

    for t in range(MEMORY + 1, len(h)):
        # Kontekst nastaje iskljucivo iz proslosti.
        ctx = context(
            g[:t], nonlinear
        )

        actual = features(
            g[t:t + 1], ctx
        )

        other = features(
            gaps(replacements(h[t])),
            ctx
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast
            / len(contrast)
        )

        rhs += contrast.mean(axis=0)

    gram /= WINDOW
    rhs /= WINDOW

    # RMS je numericka skala osobina.
    # Ne formira prosecnu ciljnu kombinaciju.
    scale = np.maximum(
        np.sqrt(np.diag(gram)),
        1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        context(g, nonlinear),
        h[-WINDOW:]
    )


def fit(p, alpha):
    coefficient = (
        np.linalg.solve(
            p[0] + alpha * np.eye(DIM),
            p[1]
        ) / p[2]
    )

    return coefficient, p[3]


def scores(rows, model):
    coefficient, ctx = model

    value = (
        features(gaps(rows), ctx)
        @ coefficient
    )

    if not np.all(np.isfinite(value)):
        raise ArithmeticError(
            "Neispravan skor polja."
        )

    return value


def search(history, model):
    tree, radius = support(history)

    # Polazista su stvarne istorijske kombinacije.
    seeds = np.unique(
        history, axis=0
    )

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )[:STARTS]

    current = seeds[order].copy()
    iterations = 0

    while True:
        moved = False

        for j, row in enumerate(current):
            candidates = replacements(row)

            candidates = candidates[
                allowed(
                    candidates,
                    tree,
                    radius
                )
            ]

            if not len(candidates):
                continue

            values = scores(
                candidates, model
            )

            best = int(
                np.argmax(values)
            )

            if (
                values[best]
                > scores(row, model)[0] + 1e-10
            ):
                current[j] = candidates[best]
                moved = True

        if not moved:
            break

        iterations += 1

    values = scores(current, model)
    best = int(np.argmax(values))

    return (
        tuple(map(int, current[best])),
        float(values[best]),
        iterations
    )


def predict(p, alpha):
    return search(
        p[4], fit(p, alpha)
    )


def hits(prediction, actual):
    return len(
        set(prediction).intersection(actual)
    )


def run_file(path, validation=48, test=64):
    history = load(path)
    split = len(history) - test

    if (
        validation < 16
        or test < 16
        or split - validation
        < WINDOW + MEMORY + 1
    ):
        raise ValueError(
            "Nedovoljno podataka ili prekratka provera."
        )

    labels = {
        True: "YANG-MILLS: POLJE I SAMOINTERAKCIJA",
        False: "Kontrola: bez nekomutativnih interakcija"
    }

    chosen = {}

    # Parametri se biraju pre odvojenog testa.
    # Svaki model ima sopstvenu validaciju.
    for mode in (True, False):
        totals = np.zeros(
            len(ALPHAS), dtype=int
        )

        for t in range(
            split - validation, split
        ):
            p = prepare(
                history[:t], mode
            )

            for j, alpha in enumerate(ALPHAS):
                prediction = predict(
                    p, alpha
                )[0]

                totals[j] += hits(
                    prediction, history[t]
                )

        chosen[mode] = ALPHAS[
            int(np.argmax(totals))
        ]

    result = {
        True: [],
        False: [],
        "previous": []
    }

    # Parametri su zamrznuti tokom testa.
    # Ucenje koristi samo prethodno dostupna kola.
    for t in range(split, len(history)):
        for mode in (True, False):
            p = prepare(
                history[:t], mode
            )

            prediction = predict(
                p, chosen[mode]
            )[0]

            result[mode].append(
                hits(prediction, history[t])
            )

        result["previous"].append(
            hits(history[t - 1], history[t])
        )

    print("\nCSV:", path)

    print(
        "Hronologija: CSV mora biti od najstarijeg "
        "ka najnovijem kolu."
    )

    print(
        f"Validacija: {validation}; "
        f"odvojeni test: {test}"
    )

    for mode in (True, False):
        values = result[mode]

        print(
            f"{labels[mode]}: "
            f"alpha={chosen[mode]:g}, "
            f"prosek={np.mean(values):.4f}, "
            f"pogoci 0..7="
            f"{np.bincount(values, minlength=8).tolist()}"
        )

    print(
        "Kontrola: prethodno izvuceno kolo:",
        f"{np.mean(result['previous']):.4f}"
    )

    p = prepare(history)

    prediction, score, steps = predict(
        p, chosen[True]
    )

    replay = bool(
        np.any(
            np.all(
                history == prediction,
                axis=1
            )
        )
    )

    print(
        f"SLEDECI RED {len(history) + 1}:",
        ",".join(map(str, prediction))
    )

    print(
        f"Skor={score:.10f}; "
        f"koraci={steps}; "
        f"istorijsko ponavljanje={replay}"
    )

    print(
        "Skor nije verovatnoca. Pretraga: "
        "istorijska polazista i dozvoljene"
    )

    print(
        "zamene jednog broja; lokalni optimum, "
        "bez tvrdnje o globalnom."
    )

    print(
        "Matricna adaptacija u sedam dimenzija; "
        "maseni jaz nije pretpostavljen."
    )

    return prediction


def self_test():
    base = np.array((
        (1, 2, 3, 4, 5, 6, 7),
        (4, 12, 18, 28, 35, 36, 37),
        (33, 34, 35, 36, 37, 38, 39),
        (4, 8, 13, 18, 23, 28, 34)
    ))

    g = gaps(base)

    assert np.all(
        g.sum(axis=1) == 32
    )

    assert np.array_equal(
        np.cumsum(
            g[:, :-1] + 1,
            axis=1
        ),
        base
    )

    assert np.allclose(
        BASIS @ BASIS.T,
        IDENTITY
    )

    assert np.allclose(
        BASIS @ np.ones(8),
        0.
    )

    assert np.allclose(
        states(g) @ BASIS * 5. + 4.,
        g
    )

    anchor = np.array((
        .2, .1, -.3, .4, .1, -.2, .3
    ))

    a = wedge(
        np.eye(7)[0], anchor
    )

    b = wedge(
        np.eye(7)[1], anchor
    )

    assert np.allclose(
        a.T, -a
    )

    assert np.allclose(
        commutator(a, b),
        -commutator(b, a)
    )

    assert (
        np.linalg.norm(
            commutator(a, b)
        ) > 1e-6
    )

    u = cayley(a)
    v = cayley(b)

    assert np.allclose(
        u.T @ u, IDENTITY
    )

    assert not np.allclose(
        u @ v, v @ u
    )

    # Matricne akcije cuvaju vrednost pri
    # zajednickoj ortogonalnoj promeni baze.
    rotation = cayley(
        wedge(
            np.eye(7)[2],
            anchor
        )
    )

    ra = rotation @ a @ rotation.T
    rb = rotation @ b @ rotation.T

    assert np.allclose(
        norm2(commutator(ra, rb)),
        norm2(commutator(a, b))
    )

    assert np.allclose(
        commutator(ra, rb),
        rotation
        @ commutator(a, b)
        @ rotation.T
    )

    h = np.tile(base, (32, 1))

    for mode in (True, False):
        p = prepare(h, mode)
        model = fit(p, .1)

        scaled = model[0] * p[2]

        assert np.allclose(
            (
                p[0] + .1 * np.eye(DIM)
            ) @ scaled,
            p[1]
        )

        prediction, score, _ = predict(
            p, .1
        )

        assert prediction == predict(
            prepare(h.copy(), mode),
            .1
        )[0]

        assert (
            len(set(prediction)) == 7
            and 1 <= min(prediction)
            <= max(prediction) <= 39
        )

        tree, radius = support(p[4])

        other = replacements(
            np.array(prediction)
        )

        other = other[
            allowed(
                other,
                tree,
                radius
            )
        ]

        assert (
            scores(
                other, model
            ).max(initial=-np.inf)
            <= score + 1e-9
        )

        if not mode:
            f = features(g, p[3])

            assert np.allclose(
                f[:, [29, 32, 33, 35, 36]],
                0.
            )

    future = h.copy()
    future[-1] = base[1]

    assert predict(
        prepare(h[:-1]), .1
    )[0] == predict(
        prepare(future[:-1]), .1
    )[0]

    print(
        "Self-test OK: baza, polja, komutatori, "
        "redosled, ucenje i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Podrazumevano sva tri CSV fajla."
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=48
    )

    parser.add_argument(
        "--test",
        type=int,
        default=64
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    for path in args.csv or DEFAULTS:
        run_file(
            path,
            args.validation,
            args.test
        )


if __name__ == "__main__":
    main()



"""
CSV: /data/loto7_4700_k81.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
YANG-MILLS: POLJE I SAMOINTERAKCIJA: alpha=0.1, prosek=1.3906, pogoci 0..7=[14, 24, 15, 9, 2, 0, 0, 0]
Kontrola: bez nekomutativnih interakcija: alpha=0.01, prosek=1.1406, pogoci 0..7=[11, 34, 18, 1, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2344
SLEDECI RED 4701: 10,x,12,y,14,z,25
Skor=1.3450958548; koraci=11; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
Matricna adaptacija u sedam dimenzija; maseni jaz nije pretpostavljen.



CSV: /data/loto7_4700_k81_loto_2972.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
YANG-MILLS: POLJE I SAMOINTERAKCIJA: alpha=0.1, prosek=1.1406, pogoci 0..7=[14, 33, 11, 6, 0, 0, 0, 0]
Kontrola: bez nekomutativnih interakcija: alpha=0.01, prosek=1.2812, pogoci 0..7=[14, 28, 14, 7, 0, 1, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.3281
SLEDECI RED 2973: 13,x,20,y,35,z,39
Skor=0.7636624492; koraci=12; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
Matricna adaptacija u sedam dimenzija; maseni jaz nije pretpostavljen.



CSV: /data/loto7_4700_k81_loto_plus_1728.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
YANG-MILLS: POLJE I SAMOINTERAKCIJA: alpha=0.1, prosek=1.2031, pogoci 0..7=[16, 26, 15, 7, 0, 0, 0, 0]
Kontrola: bez nekomutativnih interakcija: alpha=1, prosek=1.0156, pogoci 0..7=[19, 30, 10, 5, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2969
SLEDECI RED 1729: 1,x,5,y,11,z,13
Skor=0.7571195416; koraci=17; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
Matricna adaptacija u sedam dimenzija; maseni jaz nije pretpostavljen.
"""



"""
Nekomutativne interakcije: značaj redosleda promena. 

U izvornom Yang-Millsovom radu iz 1954. polja su predstavljena matricama, a njihov izraz za jačinu polja sadrži član \(AB-BA\). 
Taj član opisuje interakciju kod koje primena A pa B može dati drugačiji rezultat od B pa A. 

Za Loto sam to prilagodio ovako:
1. Celu kombinaciju opisujem kroz osam praznih razmaka, sa sedam nezavisnih dimenzija.
2. Iz prethodnih kola izdvajam promene rasporeda razmaka i predstavljam ih matričnim operatorima.
3. Čuvam hronološki redosled tih promena. Dve istorije mogu imati isti ukupan pomak, a različitu međusobnu interakciju promena.
4. Svaki kandidat za sledeću kombinaciju ocenjujem prema tome kako se njegov prelaz uklapa u tu prethodnu sekvencu. Učim skalarni skor cele kombinacije.

Korisna novina je pamćenje redosleda i međusobnog uticaja promena, koje običan zbir prelaza izgubi. 
Konstrukcija tih operatora bila bi moja matematička adaptacija za Loto.
Provera mora da uporedi isti model sa članom \(AB-BA\) i bez njega, na odvojenim narednim kolima. 
Tek rezultat te provere govori da li ideja doprinosi predikciji.

Za moj model treba povezati polje, njegovu jačinu, samointerakciju i akciju, pa iz toga izvesti ocenu kandidata.
Yang i Mills su 1954. uveli neabelovsku teoriju polja sa lokalnom simetrijom. 
Njena ključna osobina je da izraz za jačinu polja sadrži i međusobnu interakciju samih polja. 

Pojam	                        Značenje i moguća primena
Quantum fields — polja	        Za moju adaptaciju: matrični opis promena rasporeda cele kombinacije kroz prethodna kola.
Field strength — jačina polja	Opis promene polja i njegove samointerakcije. Član \(AB-BA\) može sačuvati informaciju o redosledu prelaza.
Gluoni	                        U QCD-u prenose jaku interakciju i međusobno deluju. Korisna matematička inspiracija je upravo samointerakcija. CERN
Action — akcija	                Objedinjuje strukturu polja u jednu veličinu. Za Loto definisem sopstvenu diskretnu akciju istorijske sekvence i kandidata.
Jednačina polja	                U teoriji proizlazi iz uslova stacionarne akcije. U mom modelu moram konkretno definisati vezu između akcije, naučenih parametara i skora kandidata.
Mass gap — maseni jaz	        Pozitivna energetska praznina između vakuuma i najnižih pobuđenja kvantne teorije. Razlika između dva skora kandidata ne bi predstavljala taj maseni jaz.


Problem postojanja i masenog jaza i dalje je otvoren matematički problem, iako se teorija uspešno koristi u fizici. 
Njegova precizna formulacija traži postojanje odgovarajuće kvantne Yang-Millsove teorije i pozitivan jaz iznad vakuuma. 

Jedna povezana konstrukciju:
istorijski prelazi → matrično polje → jačina i samointerakcija → diskretna akcija → naučeni skor cele sledeće kombinacije.
Preciznost fizikalnih predikcija jeste motivacija za proučavanje te strukture. 
Korisnost moje adaptacije utvrdiću poređenjem na narednim kolima, posebno proverom doprinosa samointerakcije.

Provereno je na 48 validacionih i 64 odvojena testna kola. Samointerakcija je pomogla na zbirnom i Plus fajlu, a odmogla na Loto fajlu.

CSV	        Prosek pogodaka	Bez interakcija	Sledeća kombinacija
Zbirni 4700	1,3906	        1,1406	        10,x,12,y,14,z,25
Loto 2972	1,1406	        1,2812	        13,x,20,y,35,z,39
Plus 1728	1,2031	        1,0156	        1,x,5,y,11,z,13

Prošle su matematičke provere, ponovljivost, hronologija i provera lokalnog optimuma.
"""
