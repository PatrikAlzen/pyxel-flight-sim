# Pyxel Flight Simulator

En retroinspirerad flygsimulator byggd med [Pyxel](https://github.com/kitao/pyxel). Världen är procedurellt genererad och i praktiken oändlig – flyg åt vilket håll du vill och ny terräng skapas när du kommer dit.

> **Status:** fas 0–6 klara, plus full rotation: en spelbar flygsimulator i en oändlig procedurell 3D-värld där man kan rolla och loopa hela varv, med HUD, karta, ljud, moln, vågor, jaktkamera och dag/skymning/natt. Kvar: fas 7 (innehåll och utforskning), planerad i detalj nedan.

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
| 3D-teknik | **Voxel Space i vinkelrum** (höjdkarta + strålar per kompassriktning) | Ger bergig 3D-terräng billigt, passar perfekt ihop med en höjdkarta från brusfunktioner, och klarar alla orienteringar (se *Full rotation*). |
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
- Chunkar genereras vid behov. Gamla chunkar slängs; de kan alltid återskapas från seeden. (Det blev ingen separat LRU-cache: ringbufferten nedan *är* cachen.)
- Eftersom brusfunktionen utvärderas på **världskoordinater** blir det inga skarvar mellan chunkar.
- Renderaren arbetar mot ett **lokalt fönster** (t.ex. 1024×1024 celler) som byggs av närliggande chunkar och flyttas när spelaren passerar en chunkgräns. Uppslag sker då med enkel array-indexering.
- Generering sker i **bakgrundsprocesser**, inte trådar. Pyxels huvudloop håller GIL:en, se fas 3.

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

*Ursprunglig plan (fas 2–6):* tipp flyttade horisontlinjen och roll skjuvade varje kolumn. Det drog ut terrängen i skärmkanterna vid stor roll och gjorde loopar omöjliga.

### Full rotation (nuvarande renderare)

Varje skärmpixels riktiga 3D-stråle (från kamerans fullständiga rotation) beskrivs med två vinklar: **azimut** (kompassriktning) och **lutning** (tangens av höjdvinkeln). Terrängen samplas utåt från kameran längs ett antal azimutriktningar. Längs varje riktning är skyline den brantaste lutning `(höjd − kamerahöjd) / avstånd` som setts hittills (`np.maximum.accumulate`). En pixel visar det närmaste sampel vars skyline når pixelns lutning, annars himmel. Skyline är monoton per riktning, så en enda global `searchsorted` hittar träffen för alla pixlar på en gång, precis som kolumntricket i den gamla renderaren.

Kamerans rotation påverkar bara vilka vinklar pixlarna får. Därför fungerar alla orienteringar: full roll, ryggläge, rakt upp och rakt ned. Optimeringar som håller nere kostnaden när zenit eller nadir syns på skärmen (då är alla riktningar synliga):
- **Räckvidd per riktning:** en stråle nedåt når havsytan inom `höjd / −lutning`, och en stråle uppåt passerar den högsta toppen inom `(topp − höjd) / lutning`. Toppen hämtas lokalt från en grov terrängnivå. Riktningarna grupperas i längdklasser (steg om 20 %), så att arbetet blir rektangulära arrayer.
- **Startavstånd per riktning:** högt över alla toppar kan en stråle med lutning `s` inte träffa något närmare än `(höjd − topp) / −s`. Samplen nära kameran är tätast, så det sparar mest.
- **Azimutnivåer:** pixlar nära fluktpunkten rakt upp eller ned delar grövre riktningar (1600, 800, … 100 runt hela varvet).

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

Alla konstanter samlas i `AircraftSettings` (i `sim/aircraft.py`) så att flygkänslan lätt kan finjusteras.

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
| `↑` / `↓` | Vänster spak Y | Tippa nos ned/upp (menyer: flytta markör) |
| `←` / `→` | Vänster spak X | Rolla (startskärm: byt seed) |
| `W` / `S` | `R1` / `L1` eller triggers | Öka/minska gas |
| `A` / `D` | Höger spak X | Sidroder (på marken: styr) |
| `B` | `B` | Hjulbromsar (på marken) |
| `R` | `Back` | Respawn (startskärm: slumpa seed, gamepad `Y`) |
| `M` | `X` | Stor karta (`Z`/`X` eller `R1`/`L1` zoomar) |
| `Esc` / `P` | `Start` | Pausmeny: fortsätt, respawn, ny värld, avsluta |
| `Enter` / `Space` | `A` | Bekräfta |
| `G` | – | Spökkamera av/på (planet respawnar där kameran är) |
| `F1` | – | Debug-överlägg (FPS, tiles, koordinater, flygtillstånd) |
| `Esc` / `Q` på startskärmen | – | Avsluta |
| `C` | `Y` | Cockpit / jaktkamera |
| `T` | Höger spak (klick) | Tid på dygnet: dag → skymning → natt |

### Ljud

Pyxels inbyggda ljudmotor: motorljud vars tonhöjd följer gasen, vindbrus som följer farten, varningstoner för STALL och PULL UP, en krasch och blip-ljud i menyerna (se fas 6).

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
│       ├── mapview.py         # debugkarta ovanifrån för att finjustera terrängen
│       ├── terrain/
│       │   ├── noise.py       # vektoriserat gradientbrus, fbm, ridged, domain warp
│       │   ├── generator.py   # höjd, fuktighet, biom → höjd- och färgkarta per chunk
│       │   ├── biomes.py      # biomtyper och regler för att välja biom
│       │   ├── world.py       # LOD-nivåer som ringbuffertar som följer kameran, arbetsprocesser
│       │   └── shading.py     # lutningsbaserad ljussättning
│       ├── render/
│       │   ├── palette.py     # palettdefinition och färgramper
│       │   ├── voxel.py       # Voxel Space-renderare (NumPy) med himmel och dimma
│       │   ├── clouds.py      # molnlager på fast höjd, skyms av terräng
│       │   ├── sprites.py     # pixelgrafik som text (flygplanet)
│       │   ├── framebuffer.py # NumPy-array ↔ pyxel.Image via data_ptr()
│       │   ├── hud.py         # instrument: kompass, fart/höjd-band, tipp-stege, gas, minikarta
│       │   └── worldmap.py    # stor karta och startskärmens förhandsvisning
│       ├── sim/
│       │   ├── aircraft.py    # flygmodell
│       │   ├── camera.py      # kamerans position och orientering
│       │   └── chase.py       # jaktkamera och projektion av punkter till skärmen
│       ├── audio.py           # motor, vind, varningar, krasch, menyljud
│       └── controls.py        # tangentbord/gamepad → abstrakta kommandon
└── tests/
    ├── test_noise.py          # determinism, värdeintervall, kontinuitet
    ├── test_generator.py      # inga skarvar mellan chunkar, biom, vattenyta
    ├── test_palette.py        # palettlayout, dimma ↔ himmel
    ├── test_voxel.py          # horisont, tipp/roll, skymning, dimma, LOD, ringbuffert
    ├── test_world.py          # fönster som följer kameran, stegvis = nyfyllt, arbetsprocesser
    ├── test_framebuffer.py    # NumPy-vy över Pyxel-bilder
    ├── test_aircraft.py       # planflygning, svängar, stall, landning, krascher
    ├── test_controls.py       # tangentbord + gamepad, dödzon, knapprepetition
    └── test_effects.py        # ljudval, jaktkamera, sprite, vågor, moln
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
- Spiken var slit-och-släng-kod. Voxel-algoritmen flyttades till `render/voxel.py` i fas 2 och spiken togs bort.

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

### Fas 2 – Voxel Space-renderare ✅
- [x] `voxel.py` med fast kamera över en statisk höjdkarta.
- [x] Himmel och dimma.
- [x] Fri "spökkamera" (flytta/vrid/höj/sänk) för att utforska.
- [x] Optimera tills målet ≥ 30 FPS nås i 256×144.

**Resultat:**
- `uv run flightsim` startar 3D-vyn. En förloppsindikator visas medan terrängen genereras (~3 s), sedan styr man en fri kamera. Kontroller: `↑/↓` framåt/bakåt, `←/→` sväng, `A/D` sidled, `W/S` upp/ned, `R/F` titta upp/ned, `Z/X` roll, `V` nollställ vy, `Shift` snabbare, `F1` debuginfo.
- **Terrängnivåer (LOD):** (utökat till fyra nivåer och 2500 enheters sikt i fas 6) tre nivåer i `config.TERRAIN_LEVELS`: 1024² celler med steg 1 (nära, upp till ~270 enheter), 1024² med steg 2 (till ~540) och 512² med steg 8 (till 700). Grövre nivåer längre bort ger både längre sikt och mindre flimmer. Renderaren väljer automatiskt den finaste nivå som täcker varje djupsteg.
- **Ringbuffert:** varje nivå (`TerrainLevel`) lagrar världscell (x, y) på index `(x mod N, y mod N)`. Renderaren slår upp med en bitmask och behöver aldrig veta var fönstret ligger. I fas 3 räcker det därför att skriva över de tiles som scrollat ut.
- **Dimma och himmel:** paletten har nu tre dimkopior av terrängfärgerna (klar, 35 %, 68 %) och 8 himmelsnyanser, totalt 204 färger. Dimnivån per pixel ditheras mellan kopiorna, och fullt dimmig terräng har exakt samma färg som himlen vid horisonten. Därför smälter fjärran berg in i himlen utan synlig kant.
- **Tipp och roll** stöds redan i renderaren: tipp flyttar horisonten, roll skjuvar varje kolumn. (Ersatt av full rotation efter fas 6.)
- **Prestanda:** rendering 3,9 ms (p95 4,3 ms) med `uv run flightsim --bench`, ~5,6 ms i fönster. Spelet håller 30 FPS.
- **Begränsning:** världen är än så länge statisk runt origo. Flyger man längre än ~400 enheter upprepas terrängen, eftersom ringbufferten aldrig fylls på. (Åtgärdat i fas 3.)

### Fas 3 – Oändlig värld ✅
- [x] Låt varje `TerrainLevel` följa kameran: när kameran passerar en tilegräns genereras bara den nya raden/kolumnen av tiles, som skriver över den som scrollat ut.
- [x] Generering i bakgrunden, med prioritet för den finaste nivån och de närmaste tilesen. Det blev **processer** i stället för en tråd (se nedan).
- [x] Optimera generatorn (t.ex. hoppa över bergsbrus där bergsmasken är 0).
- [x] Verifiera att det inte finns synliga skarvar och att chunkladdning inte hackar.

**Resultat:**
- Världen är oändlig. Fönstret för varje nivå centreras på närmaste tilehörn, så kameran är aldrig mer än en halv tile från mitten. Varje plats i ringbufferten minns vilken världstile den innehåller, så `missing()` ser direkt vilka tiles som behövs.
- `TerrainWorld` delar ut saknade tiles till **2 arbetsprocesser** (`config.TERRAIN_WORKERS`). Den finaste nivån går först och inom en nivå närmast först. Varje process har högst två jobb åt gången, så att prioriteringen hinner följa med när kameran svänger. Färdiga tiles läggs in i `update()` mellan bildrutorna, så renderingen ser aldrig en halvskriven tile. Tiles som hunnit lämna fönstret slängs.
- **Generatorn är ~40 % snabbare:** 21 → 12,7 ms per tile. Gradienterna förpermuteras som komplexa tal (ett uppslag per hörn i stället för tre), bråkdelarna räknas i float32 och berg- och kullbrus räknas bara där deras mask är > 0. Resultatet är fortfarande bit-för-bit deterministiskt (testat).
- **Startladdning** 3,3 s → 1,7 s.
- **I riktigt fönster** (automatisk flygning i 12 s): stabila 30 FPS (medel 33,3 ms, max ~36–39 ms per bildruta) och **noll bildrutor med inaktuell terräng**, både i 10 och 25 enheter/bildruta (300–750 enheter/s). `flightsim --bench` hittar gränsen vid ungefär 30 enheter/bildruta.
- Debugraden (`F1`) visar `TILES PENDING` och `STALE`, alltså tiles som saknas inom en nivås `safe_radius` och därför ritas med gammal data. Det ska alltid vara 0.

**Lärdomar:**
- **Pyxels huvudloop släpper inte GIL:en** medan den väntar på nästa bildruta. En bakgrundstråd fick i praktiken ingen tid alls (en tile på ~15 s) trots att benchmarken såg bra ut. Benchmarken pausar med `time.sleep`, som släpper GIL:en. Lösningen blev `multiprocessing` med vanliga pipes som huvudtråden pollar själv. `multiprocessing.Queue` och `ProcessPoolExecutor` använder hjälptrådar i huvudprocessen, och de skulle svältas på samma sätt.
- Python 3.14 startar processer med **forkserver** på Linux, som importerar huvudmodulen på nytt. `__main__.py` och alla skript som startar spelet måste därför ha `if __name__ == "__main__":`.
- Pyxels `quit()` kan avsluta utan att Pythons exit-hanterare körs. Arbetsprocesserna avslutar sig därför själva när pipen stängs.

### Fas 4 – Flygmodell ✅
- [x] `aircraft.py` med gas, tipp, roll, bankade svängar, fart och överstegring.
- [x] Koppla kameran till planet, inklusive roll via skjuvning.
- [x] Kollision med terräng, krasch och respawn.
- [x] Enhetstester för flygmodellen.

**Resultat:**
- `uv run flightsim` startar nu i luften, 120 enheter över marken i ~100 enheter/s. Spökkameran finns kvar på `G`, och lämnar man den respawnar planet där kameran är, vilket fungerar som teleport.
- **Flygmodellen** ([sim/aircraft.py](src/flightsim/sim/aircraft.py)) känner inte till Pyxel. Terrängen kommer in som en funktion `ground(x, y)`. Alla konstanter finns i `AircraftSettings`.
  - Spaken styr tipp- och rollhastighet med 0,15 s tröghet. Effekten är svagare i låg fart, och rollen planar ut när man släpper. Tippen ligger kvar, så planet håller sin attityd.
  - Planet svänger genom att banka: girhastigheten är `g·tan(roll)/fart`. 60° bank ger ~27°/s i marschfart.
  - Fart = dragkraft − luftmotstånd − `g·sin(tipp)`. Toppfart ~140 vid havsytan, lägre på hög höjd (dragkraften ×`exp(-höjd/1500)`).
  - Lyftkraft = `min(1, (fart/stallfart)²)`. Lyft som saknas, i stall eller när vingen lutar i bank, blir sjunkhastighet. I stall faller nosen. Stallfart 50, lyftfart 58.
- **Mark:** en sättning med sjunkhastighet < 8, roll < 12°, nosen inte under −4° och lutning < 0,3 blir landning. Annars kraschar planet med en orsak (för hårt, vingspetsen i marken, nosen först, terrängen, vatten). Ungefär halva landytan är plan nog att landa på. På marken styr man med `←/→` eller `A/D`, bromsar med `B`, och planet lyfter när farten > 58 och nosen höjs. Rullar man in i terräng brantare än 0,6 kraschar planet.
- **Krasch:** röd blixt, orsaken visas, kameran tittar på kraschplatsen bakifrån och ovanifrån, och efter 3 s respawnar planet på samma plats (eller direkt med `R`).
- **Tillfällig HUD** längst ned: fart, höjd, höjd över marken, vertikal fart, kurs och gas, plus blinkande STALL och PULL UP. Den riktiga HUD:en kommer i fas 5.
- **17 enhetstester** för flygmodellen: stabil planflygning, gas → fart, bankade svängar, att rollen planar ut, energi vid stigning/dykning, stall, tunn luft, svag kontroll i låg fart, landning, inbromsning och ny start, olika kraschtyper och taxning.
- Provflygning i riktigt fönster med inspelade knapptryckningar (sväng, stigning, dykning in i en sluttning, respawn): stabila 30 FPS.

### Fas 5 – HUD och kontroller ✅
- [x] Instrument, kompass, minikarta och stor karta.
- [x] Gamepad-stöd.
- [x] Startskärm med seed-val och pausmeny.

**Resultat:**
- **Spelflöde:** startskärm → laddning → flygning ↔ paus/karta. ESC är inte längre en avsluta-knapp (`quit_key=KEY_NONE`) utan öppnar pausmenyn: *Resume*, *Respawn*, *New world* (tillbaka till startskärmen, världens processer stängs) och *Quit*. Simuleringen står still i paus och på kartan.
- **Startskärm:** titeltext skalad ×3 med `blt(scale=…)`, och en förhandskarta över den valda seedens värld (1:32 runt startpunkten) som bakgrund. `←/→` byter seed, `R` slumpar.
- **HUD** ([render/hud.py](src/flightsim/render/hud.py)):
  - Kompassband med kurs.
  - Fartband med rött stallområde.
  - Höjdband med stapel för vertikal fart, VS-värde och radarhöjd (`R`, gul under 40).
  - Tipp-stege var 10:e grad, streckad under horisonten.
  - Flygplanssymbol, gasreglage och STALL/PULL UP/LANDED.
  - Tipp-stegen skjuvas exakt som renderaren, så linjerna följer den riktiga horisonten även i bank.
  - Instrumenten har solid svart bakgrund. Halvgenomskinlig dither blev för brusig mot ljus himmel.
- **Minikarta:** 40×40 px, 16 enheter/px. Den följer kursen (nosen uppåt) och har en nordmarkör. Den läses direkt från den grövsta terrängnivån i ringbufferten, så den kostar nästan ingenting.
- **Stor karta** (`M`): genereras direkt från generatorn, så den visar mer än det som finns i minnet. Zoom 1:4–1:64, blinkande planmarkör, skalstock och nordpil. Att öppna eller zooma kostar en bildruta på ~50 ms, men spelet är pausat då.
- **Input** ([controls.py](src/flightsim/controls.py)): alla handlingar fungerar med både tangentbord och gamepad. Spakarna har en dödzon på 15 %, och tangentbord och spak läggs ihop och begränsas till [-1, 1]. Bara navigering och zoom repeterar när man håller knappen, så att ESC inte slår av och på pausen. 13 tester med fejkad input.
- Provkörd i riktigt fönster med inspelade tangenttryckningar genom hela flödet (seed-byte, start, bank, karta, zoom, paus, menynavigering, fortsätt): 30 FPS under flygning.
- **Inte provat:** en riktig fysisk gamepad. Mappningen är testad mot Pyxels gamepad-konstanter, men axelriktningar och triggers kan skilja mellan kontroller.

### Fas 6 – Polish ✅
- [x] Ljud: motor, vind, varningar, krasch.
- [x] Animerat vatten, moln, dag/natt via palettbyte.
- [x] Jaktkameraläge med flygplanssprite.

**Resultat:**
- **Ljud** ([audio.py](src/flightsim/audio.py)): fyra kanaler.
  - Motor: 8 förbyggda loopar vars tonhöjd följer gasen.
  - Vind: brus i 5 nivåer som följer farten.
  - Varningar: STALL och PULL UP som loopar så länge varningen gäller.
  - Engångsljud: krasch och menyljud.
  - Pyxel kan inte ändra tonhöjden på ett ljud som spelas, så kanalen byter mellan förbyggda loopar på 0,1 s. Bytet hörs inte.
  - Valet av vad som ska spelas (`SoundState.choose`) är rena funktioner och testas utan ljudkort. Varningsregeln (`Aircraft.alert`) delas mellan HUD och ljud.
  - Ljudet tystnar i paus, på kartan och i spökkameran.
- **Vatten:** i stället för palettcykling skjuts vattnets skuggnivå ±1 av ett vågmönster som räknas i *världskoordinater* och tid. Vågorna rör sig över ytan och följer inte skärmen. Land påverkas inte (testat).
- **Moln** ([render/clouds.py](src/flightsim/render/clouds.py)): ett molnlager på höjd 380 med en sömlöst upprepad brustextur som driver med vinden. Varje pixels stråle skärs mot lagret med samma projektion som terrängen och jämförs med terrängens djup i samma pixel. Det fungerar under och ovanför molnen, berg skymmer moln och de högsta topparna sticker upp. Kanterna ditheras och undersidan är gråare än ovansidan. Molnen syns 1,6 gånger längre än terrängen.
- **Längre sikt:** molnen visade att 700 enheters sikt var för kort. Från 600 enheters höjd syntes marken knappt. Nu är siktavståndet 2500 med fyra terrängnivåer (1024²@1, 1024²@2, 1024²@8, 512²@32), och renderingen tar fortfarande ~4 ms (`--bench`) eller ~7 ms i fönster med moln.
- **Jaktkamera** (`C`): kameran sitter 26 enheter bakom planet längs flygbanan och ligger lite efter i svängar. Den kopierar 40 % av planets roll och 60 % av tippen och hålls ovanför marken. Planet ritas som en sprite ([render/sprites.py](src/flightsim/render/sprites.py)), placerad med samma projektion som renderaren och roterad med skillnaden i roll. Spriten är definierad som text i koden i stället för i `pyxel edit`, så att den kan versionshanteras och granskas. Tipp-stegen döljs i jaktvyn.
- **Tid på dygnet** (`T`): dag, skymning och natt genom att bygga om paletten med tonad terräng och ny dimma och himmel. Det kostar ingenting per bildruta. HUD-färgerna (0–15) är oförändrade, och fullt dimmig terräng matchar alltid horisonten (testat för alla tre).
- **Prestanda i fönster:** stabila 30 FPS. Inget update- eller draw-anrop tar över ~18 ms. Den enda längre bildrutan i provkörningen kom från att testskriptet sparade skärmbilder.
- 14 nya tester (totalt 85).

### Efter fas 6 – Full rotation: rolla och loopa ✅

Önskemål: terrängen drogs ut lodrätt i skärmkanterna vid stor roll, och det skulle gå att rolla och loopa ett helt varv.

- [x] Renderare som arbetar i vinkelrum i stället för skärmkolumner (se *Full rotation* under Rendering). Inget töjs vid roll, och ryggläge, rakt upp och rakt ned fungerar.
- [x] Flygmodell utan gränser för tipp och roll.
- [x] HUD, moln, jaktkamera och spökkamera för alla orienteringar.

**Resultat:**
- **Kamera** ([sim/camera.py](src/flightsim/sim/camera.py)): Eulervinklar (kurs, tipp i [−90°, 90°], roll i (−180°, 180°]) kan beskriva alla orienteringar. `basis()` ger fram-, höger- och uppvektorer, `set_vectors()` går åt andra hållet och `project()` projicerar punkter och riktningar. Fram och tillbaka är testat för slumpade orienteringar.
- **Flygmodell:** orienteringen lagras som vektorerna `forward` (nos) och `up` (tak) och roteras runt planets egna axlar: tipp runt vingarna, roll runt nosen och girning runt taket. Vid ±90° tipp finns inget specialfall. `heading`, `pitch` och `roll` finns kvar som egenskaper med setters.
  - Banksväng: lyftkraftens horisontella del vrider flygbanan, så planet svänger åt andra hållet i ryggläge.
  - Lyftbehov: `cos(tipp) − lyft · up_z`. Det är 0 i planflykt och lodrät stigning, och i ryggläge drar lyftkraften nedåt, så man måste trycka spaken framåt.
  - Autoutplaning av rollen sker bara under 60° bank, så att rollvarv och ryggflygning inte störs.
  - Tipphastigheten är höjd till 1,2 rad/s, så att en loop går att flyga utan att först dyka.
  - Ny kraschorsak: *LANDED UPSIDE DOWN*.
- **Jaktkamera:** följer planets vektorer med eftersläpning och hänger därför med genom loopar. Spriten roteras med planets bank relativt kameran.
- **Tipp-stege:** varje steg är projektionen av en riktig riktning (kurs, höjdvinkel), så stegen följer horisonten även upp och ned.
- **Moln:** avståndet till molnlagret är `(molnhöjd − höjd) / lutning`. Opaciteten förberäknas i en 4× finare textur, så varje pixel gör ett uppslag i stället för bilinjär interpolation. Det sparade ~4 ms per bildruta.
- **Prestanda** (rendering inklusive moln, utan fönster): planflygning ~7 ms, roll 60° ~9 ms, ryggläge ~7 ms, rakt ned ~10 ms, rakt upp mot berg ~15 ms och brant nedåt på hög höjd ~15 ms. I `--bench` (utan moln, med arbetsprocesser igång) tar planflygning ~11 ms mot ~4 ms med den gamla kolumnrenderaren. Full rotation kostar alltså ungefär dubbelt i det vanligaste fallet men ryms gott i budgeten.
- **Provflugen i riktigt fönster** med inspelade tangenttryckningar: ett helt rollvarv, en loop i cockpitvyn och en i jaktkameran. Stabila 30 FPS hela vägen (medel 33,3 ms, max 38 ms).
- 11 nya tester (totalt 96):
  - Flygning: rollvarv, loop, ryggflygning, att orienteringen hålls ortonormal, landning upp och ned.
  - Rendering: Euler ↔ vektorer, ryggläge, 90° roll, rakt ned och att roll inte drar ut kanterna.
  - Jaktkamera: följer genom en loop.

### Efter fas 6 – Flimmer mellan snö och berg ✅

Problem: på höga berg blinkade terrängbitar mellan snö och klippa när man flög, nästan som z-fighting.

**Felsökning:** flimret mättes som pixlar vars biom går A → B → A över tre bildrutor medan kameran flyger mot bergen, uppdelat per avstånd. Rörelse ger sällan fram-och-tillbaka-växlingar, men flimmer gör det. Två orsaker hittades:

1. **Detaljnivåerna var oense om biomet.** Klippa väljs på branta sluttningar, och lutningen mättes mellan grannceller, alltså över 2 enheter i steg 1 men 16 enheter i steg 8. Grövre nivåer såg därför mjukare och snöigare berg: steg 8 hade ett annat biom än steg 1 på **23 %** av punkterna. Där två nivåer möttes på skärmen, en gräns som flyttar sig med kameran, bytte bergssidor färg.
2. **Färgen kom från närmaste sampel.** Längs strålarna ligger samplen 6–10 enheter isär på de avstånden, så på en brant bergssida täckte varje sampel 3–4 pixelrader. När samplen gled över terrängen hoppade snögränserna i block.

**Åtgärder:**
- Lutningen för biomval mäts över en fast sträcka i världen (`TerrainSettings.biome_slope_span = 16`) på alla nivåer. Steg 1, 2 och 8 är nu helt överens (testat). Ljussättningen använder fortfarande den fina lutningen. Genereringen tar ~1 ms mer per tile eftersom rutnätet behöver en bredare kant.
- Renderaren räknar ut exakt var varje pixels stråle skär terrängen mellan de två sampel den hamnar emellan, och hämtar färgen från *den punkten* med pixelns exakta riktning, ungefär som texturmappning. Färger hör då till fasta punkter i världen. Vågorna på vattnet räknas också per pixel. De lodräta ränderna i bergssidor blev samtidigt mycket mindre.
- Höjderna läses bilinjärt mellan cellerna (`TerrainLevel.smooth_heights`) i stället för som trappsteg. Annars vickade skärningspunkten ±1 pixel när samplen gled över trappstegen. Kollisionen använder samma jämna yta, så planet och bilden är överens om var marken är.

**Resultat:**

| Avstånd | 200–400 | 400–800 | 800–1600 | Totalt |
|---|---|---|---|---|
| Före | 2,2 % | 2,8 % | 0,3 % | 2,1 % |
| Efter | 0,3 % | 0,9 % | 0,4 % | 0,5 % |

Flimrande pixlar i en testbild: 335 → 89. De som är kvar är enstaka pixlar längs kanter, alltså vanlig pixelaliasing utan kantutjämning, och inga blinkande block.
- **Kostnad:** rendering ~9 ms vid planflygning (från ~7) och ~17–19 ms i de dyraste vyerna (brant nedåt högt upp, rakt upp mot berg). I fönster fortfarande 30 FPS (medel 33,4 ms). I loopens branta nedåtdel tar bildrutorna 22–28 ms.
- 2 nya regressionstester, som båda misslyckades med den gamla koden: nivåerna är överens om biom, och en snögräns på en brant vägg flyttar sig högst en pixel per litet kamerasteg.

### Efter fas 6 – Kraschvy med vrak och rök ✅

- **Partikelsystem** ([sim/particles.py](src/flightsim/sim/particles.py)): rök, eld och gnistor i världskoordinater, lagrade som NumPy-arrayer. Det är högst 700 partiklar, och de äldsta får ge plats.
  - Röken stiger, bromsas och driver med samma vind som molnen, och den växer och tunnas ut med åldern.
  - Elden stiger snabbt och dör ung.
  - Gnistor kastas ut och faller i bågar.
- **Krasch:** i nedslagsögonblicket kommer en explosion (eldklot, gnistor och rökpuff) och ett vrak lämnas på marken. Vraket brinner med eld och en rökpelare så länge planet ligger kvar. Efter respawn hänger röken kvar några sekunder, och vraket ligger kvar tills nästa krasch eller ny värld, så att man kan se sin förra kraschplats när man flyger förbi.
- **Rendering** ([render/effects.py](src/flightsim/render/effects.py)):
  - Partiklar och vrak projiceras med samma kamera som terrängen, ritas längst bort först och skalas efter avstånd.
  - De döljs bakom terräng via renderarens avstånd per pixel, så att rök bakom en ås klipps av.
  - Rökens genomskinlighet görs med Pyxels `dither`. Ny rök tonar in så att den inte döljer vraket.
  - Rökfärgerna kommer från klippans palett, så de följer dag, skymning och natt. Elden använder klara standardfärger som lyser även på natten.
  - Vraket ([PLANE_WRECK](src/flightsim/render/sprites.py)) har böjd stjärt, avbruten höger vinge, sot och glöd, och ritas efter partiklarna eftersom flammorna annars täcker det.
- **Gamla vrak ligger kvar** ([sim/wrecks.py](src/flightsim/sim/wrecks.py)): varje kraschplats i världen sparas, upp till `config.MAX_WRECKS = 100` (de äldsta försvinner först), och listan töms vid ny värld. Vrak inom 600 enheter ritas, skalade efter avstånd och dolda bakom terräng. Vrak som skulle bli under ~3 pixlar hoppas över. Bara det senaste vraket brinner. De äldre **pyr** med en tunn rökslinga, cirka 2,5 partiklar/s slumpat per vrak inom 1500 enheter, så att kraschplatser syns på långt håll. På den stora kartan markeras de med röda kryss. Vraken sparas inte mellan spelsessioner (se fas 7).
- **Kraschvyn** kommer närmare (40 enheter bort, 18 upp) och cirklar långsamt runt vraket (12°/s). Texten har flyttats till över- och underkanten så att vraket syns, och respawn sker efter 4 s i stället för 3.
- **Prestanda:** ritningen tar i snitt ~16 ms i kraschvyn med 100–130 partiklar (jämfört med ~14 ms utan).
- 13 nya tester för partiklar och vrak (totalt 111): explosionens innehåll, jämn brinntakt, att röken stiger med vinden, att gnistorna faller, livslängd, kapacitet, att röken växer och tunnas ut, att partiklar döljs bakom terräng, att gamla vrak pyr samt vraklistans ordning, tak och närhetssökning.

### Fas 7 – Innehåll och utforskning

Fas 0–6 gav en värld att flyga i. Fas 7 ska ge spelaren **saker att göra och hitta** i världen: platser att landa på, banor att flyga, landmärken att leta efter, väder som gör flygningen mer levande och en version som går att dela i webbläsaren. Varje delfas nedan är körbar för sig. Ordningen är vald så att senare delar kan bygga på grunder från tidigare.

#### Gemensamma grunder

Flera delfaser behöver samma byggstenar. De byggs en gång och återanvänds.

**1. Deterministisk placering i ett rutnät.** Allt som placeras i världen (flygfält, banor, byar, fyrar) måste bli likadant för samma seed, i varje arbetsprocess och för varje tile, precis som terrängen. Mönstret:
- Världen delas in i stora celler (t.ex. 2048 × 2048 enheter för flygfält och 256 × 256 för byar).
- Varje cell får ett eget slumptal från `hash(seed, celltyp, cx, cy)` (t.ex. `numpy.random.default_rng([seed, typ, cx, cy])`). Därmed beror innehållet bara på cellen, inte på i vilken ordning cellerna besöks.
- En cell innehåller noll eller ett objekt (eller ett litet antal). Kandidatplatser prövas mot terrängen (höjd, lutning, biom) via `TerrainGenerator`.
- Resultatet cachas per cell (`functools.lru_cache`) i både huvudprocessen och arbetsprocesserna.

**2. Objekt per tile.** Arbetsprocesserna genererar redan terrängen tile för tile. De ska även returnera de objekt (träd, hus, fyrar) som ligger i tilen, som små arrayer med position, typ och variant. Objekten lagras per plats i `TerrainLevel` (bara den finaste nivån) och följer med när ringbufferten skriver över tiles. Huvudprocessen hämtar objekten i tilesen närmast kameran.

**3. Billboard-sprites i 3D.** `draw_wrecks` i [render/effects.py](src/flightsim/render/effects.py) gör redan allt som behövs: den projicerar många världspunkter på en gång, skalar efter avstånd, döljer bakom terräng via `pixel_distance`, sorterar längst bort först och hoppar över för små objekt. Den generaliseras till `draw_billboards(sprites, positions, variants)` och återanvänds för träd, hus, fyrar, vindstrutar och mer.

**4. Sparad data per seed.** En liten JSON-fil per seed i `pyxel.user_data_dir("pyxel-flight-sim", "Pyxel Flight Sim")`. Den innehåller vrak, sparade platser, rekordtider, besökta flygfält och upptäckta landmärken. Filen skrivs när något ändras (atomiskt: skriv till en temporär fil och byt namn) och läses när världen startar. Formatet har ett versionsnummer så att det kan utökas.

**5. Budgetar** som varje delfas måste hålla:

| Resurs | Läge idag | Utrymme för fas 7 |
|---|---|---|
| Bildruta (33 ms) | rendering ~9 ms i planflykt, 22–28 ms i värsta fall | Nya överlägg ≤ 3 ms i planflykt. Värsta fallet får inte bli sämre. |
| Palett (256) | 204 används | 2 nya terrängramper à 18 platser (t.ex. asfalt och tak), 16 kvar till HUD och effekter |
| Ljudkanaler (4) | motor, vind, varningar, effekter | Inga nya kanaler. Regn delar kanal med vinden och checkpoint-ljud går på effektkanalen. |
| Generering per tile | ~13 ms | Objektplacering ≤ 2 ms per tile |

---

#### 7a – Sparad data och favoritplatser *(liten, gör först)*

Målet är att det man gjort i en värld finns kvar till nästa gång.

- `persist.py` med `WorldSave` (dataklass och JSON) och `load(seed)` / `save(seed, data)`. Grund 4 ovan.
- Vrak sparas och läses in. `Wrecks` finns redan, så bara serialiseringen behövs.
- **Favoritplatser:** `L` (eller ett val i pausmenyn; `B` är redan broms) sparar position, kurs och höjd med ett namn ("PLATS 3"). Pausmenyn får valet *Gå till plats…* med en lista, och valet respawnar planet där.
- På startskärmen visas, per seed, antal vrak och sparade platser och om det finns rekordtider.

**Tester:** att data sparas och läses tillbaka, att ett gammalt filformat kan läsas och att en trasig fil ger en tom sparning i stället för en krasch. **Klart när:** man kan avsluta, starta om med samma seed och se sina vrak och platser.

#### 7b – Flygfält och landningsbanor *(medel)*

Målet är riktiga platser att landa på, som också fungerar som start- och mittpunkter för annat innehåll.

- **Placering:** en kandidat per cell om 2048 × 2048 enheter (grund 1). För varje kandidat provas några riktningar för en bana på cirka 300 × 16 enheter. Banans höjdprofil mäts med generatorn, och den plattaste godkänns om lutningen är under ett tröskelvärde, banan ligger över havet och under trädgränsen och ingen del ligger i vatten. Ungefär var tredje cell bör få ett flygfält.
- **Plan mark:** terrängen under banan och en övergångszon runt den jämnas ut i `TerrainGenerator.elevation`: `h = lerp(h, banhöjd, mask)`, där masken är `smoothstep` på avståndet till bankanten. Det är punktvis och deterministiskt, så tiles förblir sömlösa. Varje tile behöver bara titta på flygfälten i de närmaste cellerna.
- **Utseende:** ett nytt biom `RUNWAY` (asfaltsramp i paletten) med mittlinje och tröskelmarkeringar ritade i världskoordinater. En vindstrut och en hangar blir billboards (grund 3).
- **Spel:**
  - Landning på en bana räknas: *LANDED AT FIELD 12 – SMOOTH (−3 VS)* med betyg efter sjunkhastighet och centrering.
  - Stillastående på en bana i 2 s fyller tanken, om bränsle införs (valfritt, se nedan).
  - Besökta fält sparas (7a).
  - HUD:en visar riktning och avstånd till närmaste fält. Kartan visar alla fält i närheten med banans riktning, och besökta fält i en annan färg.
- **Start:** man kan välja att börja på marken på närmaste flygfält i stället för i luften.

**Tester:** placeringen är densamma för samma seed och oberoende av besöksordning, banan är plan inom tolerans, och tiles som korsar en bana är sömlösa (befintligt sömtest med en bana i rutan). Ett test landar på en bana och ger betyg. **Risk:** den utjämnade marken kan ge synliga kanter, så övergångszonen måste vara bred nog.

#### 7c – Ringar och tidslopp *(medel, bygger på 7b)*

Målet är en utmaning med klar början och slut.

- **Bana:** utgår från ett flygfält och genereras deterministiskt från `hash(seed, fält)`. 8–15 ringar följer dalar och kustlinjer: varje nästa ring placeras 300–600 enheter bort i en riktning som väljs bland några kandidater, med den lägsta terrängen längs vägen, 40–120 enheter över marken. Svårighet (stramare svängar, lägre höjd och mindre ringar) kan bero på avståndet från världens mitt.
- **Ritning:** en ring är en cirkel i 3D (normalen pekar längs banan). Den samplas som 24 punkter, projiceras med kameran (`Camera.project`) och ritas som linjesegment. Segment bakom terräng döljs via `pixel_distance`, precis som partiklar. Nästa ring ritas i gult, följande i vitt och passerade ringar inte alls.
- **Passage:** planets rörelse under en bildruta är ett linjesegment. Om segmentet korsar ringens plan inom ringens radie är ringen passerad. Det är oberoende av bildfrekvens och fart.
- **HUD:** pil mot nästa ring när den är utanför skärmen, tid, delta mot rekord vid varje ring (*−1,2 s* grönt / *+0,4 s* rött) och antal kvar. Ett ljud på effektkanalen vid passage.
- **Rekord** sparas per seed och bana (7a). Ett "spöke" kan sparas som planets position var tionde bildruta och visas som en genomskinlig sprite vid nästa försök.

**Tester:** att det passerade segmentet räknas exakt (genom, bredvid, för högt, baklänges), att banan är deterministisk och att ingen ring ligger i terräng. **Klart när:** man kan välja *Tidslopp* vid ett flygfält, flyga banan och se sin tid jämförd med rekordet.

#### 7d – Träd, byar och fyrar *(stor)*

Målet är att världen ska kännas befolkad och ha landmärken att navigera efter.

- **Träd:** i skog och taiga placeras träd i ett jitterrutnät (ett träd per 8 × 8 enheter med sannolikhet efter biom och fuktighet) när tilen genereras (grund 2). De ritas som 2–3 varianter av billboards (lövträd, gran, snötäckt gran), bara inom ~250 enheter. Längre bort räcker terrängens skogsfärg. Budget: högst ~800 billboards per bildruta. Det mäts, och tätheten skalas ned om det blir för dyrt.
- **Byar:** i celler om 256 × 256 enheter på flackt låglandsgräs nära vatten placeras 4–12 hus i en klunga längs en "väg" (ett smalt biom eller bara husens ordning). Hus är billboards med 3–4 varianter i taknyansen (ny takramp). Byar får namn från en enkel stavelsegenerator (seedad) och visas på kartan.
- **Fyrar:** där land möter djupt vatten (en uddes spets: hög andel hav runt omkring) placeras högst en fyr per kustcell. På natten (`palette.NIGHT`) blinkar ett ljus i toppen, en ljus pixel som syns långt. Fyrar är landmärken som "upptäcks" när man flyger nära, och de sparas (7a).
- **Kollision (valfritt):** träd och hus är höga en bit över marken. Om planet flyger in i ett objekts radie nära marken blir det krasch, *HIT A TREE* eller *HIT A HOUSE*. Det kontrolleras bara mot objekt i tilen under planet.

**Tester:** placeringen är deterministisk och sömlös över tilegränser (objekt nära en kant hamnar i exakt en tile), det finns inga träd i vatten eller på klippa, och fyrar står bara vid kust. Ett prestandatest kör `draw_billboards` med 800 objekt och kräver under 3 ms. **Risk:** prestandan med många sprites i Python. Det mildras med hård avståndsgräns, täthetsskalning och med att objekt som täcks av terräng sorteras bort innan Pyxel-anropen.

#### 7e – Väder: vind, turbulens och regn *(medel)*

Målet är att flygningen ska variera och att förhållandena ska kännas.

- **Vädertillstånd:** `Weather` med vindriktning och vindstyrka, byighet, molntäckning, regnintensitet och sikt. Vädret byter långsamt mellan förinställningar (*klart*, *halvklart*, *mulet*, *regn*, *storm*) över några minuter, eller låses i pausmenyn.
- **Vind:** samma vindvektor används för planet, molnen och röken. Vinden läggs till planets förflyttning (vindavdrift). Landning i motvind ger lägre markfart, och vindstruten på flygfälten visar riktningen.
- **Turbulens:** små slumpmässiga tipp- och rollstörningar från brus över tid, starkare i storm, nära bergskammar (där terrängens lutning i lovart är hög) och i moln.
- **Moln:** molntäckningen styrs direkt av vädret. Tröskeln flyttas från förberäkningen till ritningen, så att täckningen kan ändras utan att texturen byggs om. I mulet väder sänks molnbasen.
- **Regn:** sneda streck i skärmrymden (linjer i 1–2 nyanser) vars lutning följer fart och vind, med mörkare palett och kortare sikt via en ny `Lighting`-variant (*mulet*: grå dimma och dämpad terräng). Ljudet: vindkanalen byter till ett regnbrus.
- **HUD:** vindpil och vindstyrka, plus *TURBULENCE* som varning i storm.

**Tester:** vindavdrift (ett plan med kurs norrut och vind från väster driver österut), att övergångarna mellan förinställningar är kontinuerliga och att molntäckningen vid en given tröskel stämmer. **Risk:** turbulens kan kännas frustrerande, så den behöver tydliga reglage och vara svag som standard.

#### 7f – Webbversion *(medel, gör sist)*

Målet är att spelet ska kunna spelas i webbläsaren och delas med en länk.

- **Paketering:** `pyxel package` och sedan `pyxel app2html` ger en `.pyxapp` och en HTML-fil (Pyodide i webbläsaren). NumPy behöver laddas som Pyodide-paket, och det måste verifieras med Pyxels webbstart.
- **Inga processer i webbläsaren:** `multiprocessing` finns inte i Pyodide. `TerrainWorld(workers=0)` fungerar redan men genererar inget i bakgrunden. Det behövs ett läge där huvudtråden genererar tiles med en tidsbudget per bildruta (t.ex. högst 8 ms, de mest brådskande först) i `TerrainWorld.update`.
- **Prestanda:** NumPy i WebAssembly är troligen 1,5–3 gånger långsammare. En förinställning `config.WEB` kan sänka siktavståndet, slå av molnen eller lägga dem i halv upplösning, minska antalet azimut och glesa ut träden. Mätningar i en riktig webbläsare avgör vad som behövs.
- **Sparning:** `pyxel.user_data_dir` i webbläsaren lagras i webbläsarens lokala lagring och måste verifieras.
- **Publicering:** HTML-filen kan läggas ut på t.ex. GitHub Pages, med en länk i README.

**Tester:** att terrängen blir likadan med `workers=0` och tidsbudget som med arbetsprocesser (befintligt test i [test_world.py](tests/test_world.py), utökat) och att tidsbudgeten hålls. **Klart när:** HTML-versionen startar, går att flyga i minst 20 FPS och sparar vrak och platser.

---

#### Ordning och beroenden

| Delfas | Storlek | Beroende av | Värde |
|---|---|---|---|
| 7a Sparad data | Liten | – | Grund för allt som ska finnas kvar |
| 7b Flygfält | Medel | 7a | Mål att flyga till, landningar som räknas |
| 7c Tidslopp | Medel | 7a, 7b | Utmaning med rekord |
| 7d Träd, byar, fyrar | Stor | grund 2–3 | Levande värld, landmärken |
| 7e Väder | Medel | – | Variation, svårighet |
| 7f Webb | Medel | allt ovan | Delbart |

**Fler idéer** som kan få egna delfaser senare:
- Bränsle och räckvidd, kopplat till flygfälten.
- Fler flygplanstyper med olika `AircraftSettings` och sprites.
- Utmärkelser, till exempel *landa på 10 fält* eller *loopa under 50 enheter*.
- Fotoläge (dölj HUD, fri kamera och `pyxel.screen.save`).
- Repris av den senaste minuten.
- Floder som skär genom terrängen.
- Trafik i form av andra plan som följer banor mellan flygfälten.

## Prestanda och risker

| Risk | Åtgärd |
|---|---|
| Python för långsamt för Voxel Space | Full NumPy-vektorisering, rita till array och kopiera via `data_ptr()`. Fallback: lägre upplösning, färre avståndssteg, Numba. |
| Hack när nya chunkar genereras | Löst i fas 3: arbetsprocesser (inte trådar – Pyxel håller GIL:en) och en halv tiles marginal i varje nivå. |
| Begränsad palett ger platt utseende | Utökad palett + dithering + lutningsskuggning. |
| Roll ser konstig ut vid stora vinklar | Löst efter fas 6: renderaren arbetar i vinkelrum och klarar alla orienteringar. |
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
uv run flightsim                 # flyg! (se Kontroller ovan; G = spökkamera)
uv run flightsim --bench         # mät terränggenerering och rendering utan fönster
uv run flightsim-map             # kartvy: pilar = panorera, Z/X = zoom, 1-3 = lager, N/B = seed
uv run pytest                    # kör tester
uv run ruff check . && uv run ruff format src tests   # lint och formatering
```

`uv run python -m flightsim` fungerar också. Om pyenv klagar på `.python-version`: kör via `uv run`, som använder projektets egen Python-miljö.
