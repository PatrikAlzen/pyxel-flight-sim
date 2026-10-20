# Pyxel Flight Simulator

En retroinspirerad flygsimulator byggd med [Pyxel](https://github.com/kitao/pyxel). Världen är procedurellt genererad och i praktiken oändlig – flyg åt vilket håll du vill och ny terräng skapas när du kommer dit.

> **Status:** fas 0–5 klara: startskärm med seed-val, flygbart plan med HUD och minikarta i en oändlig procedurell 3D-värld, karta, pausmeny och gamepad-stöd. Nästa steg: fas 6, polish (ljud, moln, jaktkamera).

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
| `C` | `Y` | Byt kameravy (fas 6) |

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
│       │   ├── sky.py         # moln (fas 6)
│       │   ├── framebuffer.py # NumPy-array ↔ pyxel.Image via data_ptr()
│       │   ├── hud.py         # instrument: kompass, fart/höjd-band, tipp-stege, gas, minikarta
│       │   └── worldmap.py    # stor karta och startskärmens förhandsvisning
│       ├── sim/
│       │   ├── aircraft.py    # flygmodell
│       │   └── camera.py      # kamerans position och orientering, kameralägen
│       ├── audio.py           # ljud och musik
│       └── controls.py        # tangentbord/gamepad → abstrakta kommandon
└── tests/
    ├── test_noise.py          # determinism, värdeintervall, kontinuitet
    ├── test_generator.py      # inga skarvar mellan chunkar, biom, vattenyta
    ├── test_palette.py        # palettlayout, dimma ↔ himmel
    ├── test_voxel.py          # horisont, tipp/roll, skymning, dimma, LOD, ringbuffert
    ├── test_world.py          # fönster som följer kameran, stegvis = nyfyllt, arbetsprocesser
    ├── test_framebuffer.py    # NumPy-vy över Pyxel-bilder
    ├── test_aircraft.py       # planflygning, svängar, stall, landning, krascher
    └── test_controls.py       # tangentbord + gamepad, dödzon, knapprepetition
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
- **Terrängnivåer (LOD):** tre nivåer i `config.TERRAIN_LEVELS`: 1024² celler med steg 1 (nära, upp till ~270 enheter), 1024² med steg 2 (till ~540) och 512² med steg 8 (till 700). Grövre nivåer längre bort ger både längre sikt och mindre flimmer. Renderaren väljer automatiskt den finaste nivå som täcker varje djupsteg.
- **Ringbuffert:** varje nivå (`TerrainLevel`) lagrar världscell (x, y) på index `(x mod N, y mod N)`. Renderaren slår upp med en bitmask och behöver aldrig veta var fönstret ligger. I fas 3 räcker det därför att skriva över de tiles som scrollat ut.
- **Dimma och himmel:** paletten har nu tre dimkopior av terrängfärgerna (klar, 35 %, 68 %) och 8 himmelsnyanser, totalt 204 färger. Dimnivån per pixel ditheras mellan kopiorna, och fullt dimmig terräng har exakt samma färg som himlen vid horisonten. Därför smälter fjärran berg in i himlen utan synlig kant.
- **Tipp och roll** stöds redan i renderaren: tipp flyttar horisonten, roll skjuvar varje kolumn.
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
| Hack när nya chunkar genereras | Löst i fas 3: arbetsprocesser (inte trådar – Pyxel håller GIL:en) och en halv tiles marginal i varje nivå. |
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
uv run flightsim                 # flyg! (se Kontroller ovan; G = spökkamera)
uv run flightsim --bench         # mät terränggenerering och rendering utan fönster
uv run flightsim-map             # kartvy: pilar = panorera, Z/X = zoom, 1-3 = lager, N/B = seed
uv run pytest                    # kör tester
uv run ruff check . && uv run ruff format src tests   # lint och formatering
```

`uv run python -m flightsim` fungerar också. Om pyenv klagar på `.python-version`: kör via `uv run`, som använder projektets egen Python-miljö.
