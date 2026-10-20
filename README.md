# Pyxel Flight Simulator

En retroinspirerad flygsimulator byggd med [Pyxel](https://github.com/kitao/pyxel). Världen är procedurellt genererad och i praktiken oändlig – flyg åt vilket håll du vill och ny terräng skapas när du kommer dit.

> **Status:** fas 0–1 klara (renderingsspik, terränggenerering och kartvy). Nästa steg: fas 2, Voxel Space-renderaren.

---

## Innehåll

1. [Mål](#mål)
2. [Teknisk översikt](#teknisk-översikt)
3. [Terränggenerering](#terränggenerering)
4. [Rendering](#rendering)
5. [Flygmodell](#flygmodell)
6. [Kamera, HUD och kontroller](#kamera-hud-och-kontroller)
7. [Projektstruktur](#projektstruktur)
8. [Implementationsplan](#implementationsplan)
9. [Prestanda och risker](#prestanda-och-risker)
10. [Kom igång](#kom-igång)

---

## Mål

- **Utforskning först:** oändlig, deterministisk värld (samma seed ⇒ samma värld) med berg, dalar, sjöar, kust och snötoppar.
- **Retrokänsla:** låg upplösning (256×144), begränsad palett, pixlig estetik i stil med *Comanche* (1992).
- **Lättillgänglig flygning:** arkadbetonad men trovärdig flygmodell – gas, tipp, roll, överstegring och krasch.
- **Smidigt:** sikta på 30 FPS på en vanlig laptop.

Utanför scope (åtminstone initialt): realistisk aerodynamik, flera flygplanstyper, multiplayer.

## Teknisk översikt

| Del | Val | Motivering |
|---|---|---|
| Motor | Pyxel ≥ 2.9.9 | Fönster, input, ljud, palett och pixelbuffert ingår. |
| Matematik | NumPy | Pyxel saknar riktig 3D. Terräng och rendering måste vektoriseras för att Python ska hinna med. |
| 3D-teknik | **Voxel Space** (höjdkarta + raycasting per kolumn) | Ger bergig 3D-terräng billigt, passar perfekt ihop med en höjdkarta från brusfunktioner. |
| Bildöverföring | `pyxel.Image.data_ptr()` + NumPy | Hela bilden ritas i en NumPy-array och kopieras rakt in i Pyxels bildbuffert i ett svep, i stället för tusentals `pset`/`line`-anrop. |
| Pakethantering | uv | Redan uppsatt i projektet (`pyproject.toml`, `uv.lock`). |

Övergripande dataflöde per bildruta:

```
input ──► Aircraft.update() ──► Camera (position, kurs, tipp, roll)
                                      │
             TerrainWorld (chunkar) ◄─┤  hämtar höjd/färg runt kameran
                                      ▼
                         VoxelRenderer (NumPy) ──► framebuffer (uint8)
                                                         │
                                         data_ptr() ──► pyxel-bild ──► blt till skärm
                                                                          │
                                                                   HUD ritas ovanpå
```

## Terränggenerering

### Brus

Egen, vektoriserad brusimplementation i NumPy (inga extra beroenden):

- **Grundbrus:** gradientbrus (Perlin/OpenSimplex-liknande) som kan utvärderas på en hel grid av världskoordinater åt gången.
- **fBm (fractal Brownian motion):** 5–6 oktaver, `lacunarity ≈ 2.0`, `gain ≈ 0.5`, för detaljer på flera skalor.
- **Ridged noise:** `1 - |brus|` för skarpa bergskammar i bergsbiom.
- **Domain warping:** förskjut koordinaterna med ett annat brusfält för mer organiska former.
- **Determinism:** allt härleds från en global `seed`; varje oktav får en egen permutationstabell skapad med `numpy.random.default_rng(seed + i)`.

### Höjd och biom

Höjden byggs av flera lager:

```
continent = fbm(x * 0.0005)            # stora landmassor / hav
mountains = ridged(x * 0.003) * mask   # mask = smoothstep(continent)
hills     = fbm(x * 0.01)  * 0.2
height    = vattennivå + f(continent) + mountains + hills
```

Ett separat **fuktighetsfält** (lågfrekvent brus) kombineras med höjd för att välja biom:

| Höjd | Torrt | Fuktigt |
|---|---|---|
| Under vattennivå | Djupt/grunt vatten | Djupt/grunt vatten |
| Strand | Sand | Sand |
| Låglänt | Stäpp/öken | Gräs, skog |
| Medel | Klippor | Barrskog |
| Högt | Berg | Berg |
| Topp | Snö | Snö |

### Färg och skuggning

- Varje biom mappas till en **färgramp** (3–4 nyanser) i en utökad palett.
- **Ljussättning:** beräkna lutning från höjdkartans gradient (`np.gradient`) och skugga mot en fast solriktning (lambert). Det är detta som får terrängen att se tredimensionell ut.
- Valfritt: Pyxels `dither` eller ordnad dithering för mjukare övergångar.
- Vatten ritas med plan höjd = vattennivån, så att sjöar och hav blir platta ytor.

### Oändlig värld med chunkar

- Världen delas in i **chunkar** om 128×128 celler. En chunk innehåller `height: float32[128,128]` och `color: uint8[128,128]`.
- Chunkar genereras vid behov och cachas i en **LRU-cache** (t.ex. ~64 chunkar). Gamla chunkar slängs; de kan alltid återskapas från seeden.
- Eftersom brusfunktionen utvärderas på **världskoordinater** blir det inga skarvar mellan chunkar.
- Renderaren arbetar mot ett **lokalt fönster** (t.ex. 1024×1024 celler) som byggs av närliggande chunkar och flyttas när spelaren passerar en chunkgräns. Uppslag sker då med enkel array-indexering.
- Generering kan senare flyttas till en bakgrundstråd så att det aldrig hackar när nya chunkar laddas (NumPy släpper GIL:en under tunga operationer).

## Rendering

### Voxel Space-algoritmen

För varje bildruta:

1. Rita himmel (gradient, ev. moln) i framebuffern.
2. För avstånd `z` från nära till långt (med växande steglängd, t.ex. `dz *= 1.01`):
   - Beräkna en linje i världen vinkelrätt mot kameran på avstånd `z`, ett sampel per skärmkolumn (`W` sampel – vektoriserat med NumPy).
   - Slå upp höjd och färg för alla sampel.
   - Projicera: `screen_y = (cam_height - h) / z * scale + horizon`.
   - För varje kolumn: rita från `screen_y` ned till `ybuffer[x]` om den är synlig, uppdatera sedan `ybuffer[x] = min(ybuffer[x], screen_y)`.
3. Applicera **dimma**: blanda mot himmelsfärgen med ökande `z` (eller byt till mörkare/blekare palettindex), vilket också döljer att sikten är begränsad.

Fram-till-bak-ordningen med en `ybuffer` ger automatiskt korrekt skymning (occlusion) utan z-buffert.

### Vektorisering

Den naiva dubbelslingan (kolumner × avståndssteg) är för långsam i ren Python. Plan:

- **Steg 1 (enkel):** slinga över `z` i Python (~150–250 steg), allt per kolumn med NumPy. Fyllning av vertikala spann görs med en mask/`np.where` på hela framebuffern eller med en färgindex-array per kolumn.
- **Steg 2 (snabbare):** beräkna alla `z` på en gång (2D-array `Z×W`), ta `np.minimum.accumulate` längs `z` för att få ybuffern, och fyll pixlar per kolumn genom att hitta första `z` där ybuffern passerar varje `y` (`searchsorted`-liknande logik).
- Mät båda och välj den snabbaste. Fallback om det ändå är för långsamt: lägre intern upplösning (t.ex. 160×90) och uppskalning, eller valfri Numba-JIT.
- **Uppdatering efter fas 0:** steg 2 är implementerat i spiken och tar ~1,6 ms per bildruta. Det blir utgångspunkten.

### Tipp och roll

- **Tipp (pitch):** flytta horisontlinjen upp/ned (klassiskt Voxel Space-trick).
- **Roll:** skjuv per kolumn – horisonten för kolumn `x` förskjuts med `(x - W/2) * tan(roll)`. Ser övertygande ut upp till ca ±45°. Vid större rollvinklar kan hela bilden roteras i efterhand (NumPy-indexering med förberäknad rotationstabell).

### Palett

Pyxel använder palettindex. Vi ersätter/utökar standardpaletten via `pyxel.colors` med:

- himmelsgradient (4–6 färger),
- terrängramper per biom (ljus → skugga),
- dimnyanser,
- några HUD-färger.

Om paletten blir för stor används dithering mellan närliggande nyanser i stället.

### Alternativ som undersöks

Pyxel 2.x har `Image.blt3d` / `bltm3d`, som ritar en platt bild/tilemap i perspektiv (Mode 7-liknande). Det ger ingen höjd, men kan vara användbart för t.ex. en snabb vattenyta eller som tidig prototyp innan Voxel Space-renderaren är klar.

## Flygmodell

Arkadmodell med några fysikaliska drag. Tillstånd i `Aircraft`:

```python
pos: np.ndarray        # x, y (världen), z (höjd)
heading: float         # kurs (yaw), rad
pitch: float           # tipp, rad
roll: float            # roll, rad
speed: float           # lufthastighet
throttle: float        # 0..1
```

Uppdatering per tick (fast tidssteg, `dt = 1/30`):

1. **Input** ändrar `pitch_rate` och `roll_rate` (med dämpning så att planet planar ut när man släpper).
2. **Svängning:** `heading += g * tan(roll) / speed * dt` – planet svänger genom att banka, som ett riktigt flygplan.
3. **Fart:** `speed += (thrust(throttle) - drag(speed) - g * sin(pitch)) * dt` – man tappar fart när man stiger och vinner fart i dykning.
4. **Lyft och överstegring:** under `stall_speed` tappar planet lyft, nosen tippar nedåt och planet sjunker.
5. **Förflyttning:** hastighetsvektor från kurs + tipp, `pos += velocity * dt`.
6. **Kollision:** jämför `pos.z` med terränghöjden under planet. Mjuk kontakt på plan mark med låg sjunkhastighet = landning, annars krasch → explosion och respawn.
7. **Gränser:** maxhöjd (tunnare luft ⇒ mindre motorkraft).

Alla konstanter samlas i `config.py` så att flygkänslan lätt kan finjusteras.

## Kamera, HUD och kontroller

### Kamera

- **Cockpit-/nosvy:** kameran sitter i planet, följer kurs, tipp och roll.
- **Jaktvy (senare):** kameran något bakom och ovanför, med planet som sprite i mitten. Kamerahöjden hålls alltid över terrängen.

### HUD

- Konstgjord horisont / pitch-stege
- Fart och höjd (både över havet och över marken)
- Kompassband med kurs
- Gasreglage
- **Minikarta** – nedskalad färgkarta runt spelaren med planets position och riktning
- Varning vid överstegring och låg höjd (blinkande text + ljud)

### Kontroller

| Tangent | Gamepad | Funktion |
|---|---|---|
| `↑` / `↓` | Vänster spak Y | Tippa nos ned/upp |
| `←` / `→` | Vänster spak X | Rolla |
| `W` / `S` | `R1` / `L1` | Öka/minska gas |
| `A` / `D` | Höger spak X | Sidroder (liten girning) |
| `C` | `Y` | Byt kameravy |
| `M` | `X` | Stor karta |
| `R` | `Start` | Respawn |
| `F1` | – | Debug-överlägg (FPS, chunkar, koordinater) |
| `Q` / `Esc` | – | Avsluta |

### Ljud

Pyxels inbyggda ljudmotor: motorljud vars tonhöjd följer gasen, vindbrus som följer farten, varningspip vid överstegring och en explosion vid krasch.

## Projektstruktur

```
pyxel-flight-sim/
├── pyproject.toml
├── README.md
├── src/
│   └── flightsim/
│       ├── __main__.py        # startpunkt: python -m flightsim
│       ├── app.py             # Pyxel-appen: init, update, draw, spellägen
│       ├── config.py          # upplösning, seed, konstanter för flygmodell och rendering
│       ├── spike.py           # fas 0-spik (tillfällig), ersätts av render/voxel.py
│       ├── mapview.py         # debugkarta ovanifrån för att finjustera terrängen
│       ├── terrain/
│       │   ├── noise.py       # vektoriserat gradientbrus, fbm, ridged, domain warp
│       │   ├── generator.py   # höjd, fuktighet, biom → höjd- och färgkarta per chunk
│       │   ├── biomes.py      # biomtyper och regler för att välja biom
│       │   ├── world.py       # chunk-cache (LRU), lokalt fönster, höjdfråga för kollision
│       │   └── shading.py     # lutningsbaserad ljussättning
│       ├── render/
│       │   ├── palette.py     # palettdefinition och färgramper
│       │   ├── voxel.py       # Voxel Space-renderare (NumPy)
│       │   ├── sky.py         # himmel, dimma, moln
│       │   ├── framebuffer.py # NumPy-array ↔ pyxel.Image via data_ptr()
│       │   └── hud.py         # instrument, minikarta
│       ├── sim/
│       │   ├── aircraft.py    # flygmodell
│       │   └── camera.py      # kameralägen
│       ├── audio.py           # ljud och musik
│       └── controls.py        # tangentbord/gamepad → abstrakta kommandon
└── tests/
    ├── test_noise.py          # determinism, värdeintervall, inga skarvar mellan chunkar
    ├── test_world.py          # chunk-cache, höjdfrågor
    └── test_aircraft.py       # stall, svängning, kollision
```

## Implementationsplan

Varje fas ska resultera i något körbart.

### Fas 0 – Grund och teknisk spik ✅
- [x] Lägg till `numpy` (och `pytest` som dev-beroende) med `uv add`.
- [x] Skapa paketstrukturen och en tom Pyxel-app (`python -m flightsim`).
- [x] **Spik:** fyll en NumPy-array med brus och visa den via `data_ptr()`. Mät hur många ms det tar per bildruta. Detta avgör om renderingsstrategin håller.

**Resultat** (`uv run flightsim-spike --bench`, 256×144, WSL2):

| Mätning | Medel | p95 |
|---|---|---|
| fBm-brus för hela skärmen (5 oktaver), live varje bildruta | 2,5 ms | 3,0 ms |
| Voxel Space, 143 djupsteg, helt vektoriserad (variant 2 nedan) | 1,6 ms | 1,8 ms |
| Höjdkarta 512×512, 6 oktaver (engångskostnad) | 50 ms | – |

I ett riktigt fönster tar renderingen ~2,5 ms och spelet håller 30 FPS. **Slutsats:** strategin håller med stor marginal – ca 5 % av budgeten på 33 ms. Det ger utrymme för fler djupsteg, längre siktavstånd, dimma och högre upplösning. Den helt vektoriserade varianten (`searchsorted` över alla kolumner) fungerar, så Numba behövs inte.

Lärdomar:
- NumPy-vyn från `data_ptr()` håller inte `pyxel.Image` vid liv. Bilden måste refereras så länge vyn används, annars blir det segfault.
- Python 3.14.0a4 (alfa) fungerar inte med NumPy-hjulen. Använd en stabil 3.14.x (projektet kör 3.14.8 via uv).
- Spiken (`src/flightsim/spike.py`) är slit-och-släng-kod. Voxel-algoritmen därifrån flyttas till `render/voxel.py` i fas 2.

### Fas 1 – Brus och 2D-karta ✅
- [x] Implementera `noise.py` (gradientbrus + fbm + ridged) med tester.
- [x] `generator.py`: höjd, fuktighet, biom, färg och skuggning för en chunk.
- [x] Debugvy: scrollbar ovanifrån-karta av terrängen (piltangenter). Bra för att finjustera världens utseende innan 3D finns.

**Resultat:**
- `uv run flightsim-map` öppnar kartvyn. Den har panorering, zoom 1:1–1:64, lagren terräng/biom/fuktighet, byte av seed och info om höjd, biom och fuktighet där musen pekar.
- Höjden består av lager: kontinenter med domain warping, ett höjdsteg vid kusten, ridged-berg (bara inåt land och där en bergskedjemask tillåter) och kullar. Sea level = 0, toppar på ~300 enheter. Alla parametrar finns i `TerrainSettings`.
- 10 biom (djupt/grunt vatten, sand, gräs, skog, stäpp, öken, taiga, klippa, snö) väljs från höjd, fuktighet och lutning. Brant terräng blir klippa oavsett höjd.
- Paletten har 76 färger: Pyxels 16 standardfärger plus 10 biom × 6 skuggnivåer. Pyxel klarar upp till 256 (`pyxel.colors[:] = ...`), så det finns plats för dimma och himmel.
- Ljuset beräknas från lutningen, med sol från nordväst och ordnad dithering (Bayer 4×4) på världskoordinater. Generatorn räknar med en cells kant runt varje sampel, så skuggningen blir sömlös mellan chunkar (testat).
- **Prestanda:** en chunk på 128×128 tar ~21 ms och en hel skärm ~40 ms. Det är för mycket för att göras mitt i en bildruta, så fas 3 behöver generering i en bakgrundstråd och/eller optimering (t.ex. hoppa över bergsbrus där bergsmasken är 0).

### Fas 2 – Voxel Space-renderare
- [ ] `voxel.py` med fast kamera över en statisk höjdkarta.
- [ ] Himmel och dimma.
- [ ] Fri "spökkamera" (flytta/vrid/höj/sänk) för att utforska.
- [ ] Optimera tills målet ≥ 30 FPS nås i 256×144.

### Fas 3 – Oändlig värld
- [ ] `world.py`: chunkar, LRU-cache och lokalt fönster som följer kameran.
- [ ] Verifiera att det inte finns synliga skarvar och att chunkladdning inte hackar (annars: bakgrundstråd).

### Fas 4 – Flygmodell
- [ ] `aircraft.py` med gas, tipp, roll, bankade svängar, fart och överstegring.
- [ ] Koppla kameran till planet, inklusive roll via skjuvning.
- [ ] Kollision med terräng, krasch och respawn.
- [ ] Enhetstester för flygmodellen.

### Fas 5 – HUD och kontroller
- [ ] Instrument, kompass, minikarta och stor karta.
- [ ] Gamepad-stöd.
- [ ] Startskärm med seed-val och pausmeny.

### Fas 6 – Polish
- [ ] Ljud: motor, vind, varningar, krasch.
- [ ] Animerat vatten (palettcykling), moln, ev. dag/natt via palettbyte.
- [ ] Jaktkameraläge med flygplanssprite (ritas i Pyxels editor, `pyxel edit`).

### Fas 7 – Idéer för senare
- Landningsbanor och flygplatser placerade procedurellt på plana ytor.
- Ringar/checkpoints att flyga igenom, tidslopp.
- Procedurella objekt: träd, byar, fyrar (som sprites skalade efter avstånd).
- Väder: vind, turbulens, regn.
- Spara favoritplatser (seed + koordinater).
- Webbversion via `pyxel package` / `pyxel app2html`.

## Prestanda och risker

| Risk | Åtgärd |
|---|---|
| Python för långsamt för Voxel Space | Full NumPy-vektorisering, rita till array och kopiera via `data_ptr()`. Fallback: lägre upplösning, färre avståndssteg, Numba. |
| Hack när nya chunkar genereras | Generera i förväg i en ring runt spelaren, i bakgrundstråd. |
| Begränsad palett ger platt utseende | Utökad palett + dithering + lutningsskuggning. |
| Roll ser konstig ut vid stora vinklar | Skjuvning för små vinklar, bildrotation för stora. |
| Flygmodellen känns fel | Alla konstanter i `config.py`, debug-överlägg med live-värden. |

**Prestandabudget (30 FPS ⇒ 33 ms per bildruta):**

| Del | Mål |
|---|---|
| Flygmodell + input | < 1 ms |
| Voxel-rendering | < 20 ms |
| Kopiering till Pyxel + HUD | < 3 ms |
| Chunkgenerering (amorterad) | < 5 ms |

## Kom igång

Projektet använder [uv](https://docs.astral.sh/uv/) och Python 3.14.

```bash
uv sync                          # installera beroenden (installerar Python 3.14 vid behov)
uv run flightsim                 # starta spelet (just nu bara en startskärm)
uv run flightsim-map             # kartvy: pilar = panorera, Z/X = zoom, 1-3 = lager, N/B = seed
uv run flightsim-spike           # renderingsspiken: 1 = brus, 2 = voxel, pilar = flyg, W/S = höjd
uv run flightsim-spike --bench   # mät renderingstider utan fönster
uv run pytest                    # kör tester
uv run ruff check . && uv run ruff format src tests   # lint och formatering
```

`uv run python -m flightsim` fungerar också. Om pyenv klagar på `.python-version`: kör via `uv run`, som använder projektets egen Python-miljö.
