# Flygsimulatorn på nytt – designdokument

Det här dokumentet beskriver hur spelet skulle kunna byggas om från början. Kärnan behålls: en oändlig värld att utforska, en flygning som är lätt att lära, och flera flygplan och helikoptrar. Utseendet blir modernt, med hög upplösning och detaljerade modeller. Dokumentet jämför de viktigaste vägvalen och slutar i en rekommendation och en plan i faser.

Det nuvarande spelet (se [README.md](README.md)) kallas här **Pyxel-versionen**.

---

## Innehåll

1. [Kärnan som ska behållas](#1-kärnan-som-ska-behållas)
2. [Målbild](#2-målbild)
3. [Arkitektur](#3-arkitektur)
4. [Motor och språk](#4-motor-och-språk)
5. [Världen](#5-världen)
6. [Rendering](#6-rendering)
7. [Modeller och innehåll](#7-modeller-och-innehåll)
8. [Flygmodell](#8-flygmodell)
9. [Platser att upptäcka och interagera med](#9-platser-att-upptäcka-och-interagera-med)
10. [Kamera, gränssnitt, ljud](#10-kamera-gränssnitt-ljud)
11. [Test och verktyg](#11-test-och-verktyg)
12. [Prestanda](#12-prestanda)
13. [Risker](#13-risker)
14. [Implementationsplan](#14-implementationsplan)
15. [Rekommendation i korthet](#15-rekommendation-i-korthet)

---

## 1. Kärnan som ska behållas

- **Utforskning först:** en oändlig, deterministisk värld. Samma seed ger samma värld, med berg, dalar, sjöar, floder, kust och snötoppar. Platser finns att upptäcka och att interagera med.
- **Lättillgänglig flygning:** en arkadbetonad men trovärdig flygmodell, med gas, tipp, roll, överstegring och krasch.
- **Olika flygplan och helikoptrar:** med olika karaktär, inte bara olika siffror.

### Det som Pyxel-versionen lärt oss

Några lärdomar är värda mer än koden och bör styra den nya versionen från dag ett:

- **Allt i världen beräknas från (seed, plats).**
  - Flygfält, byar, fyrar, broar och helikopterplattor placeras per cell med slumptal från cellens koordinater. Det gör världen oberoende av i vilken ordning saker genereras, och av vilken tråd eller process som gör det.
  - Det är grunden för determinismen och ska vara en hård regel.
- **Modellerna och kollisionen delar mått.**
  - Hangarer och torn fick samma mått för modell och kollision först i efterhand ([terrain/airfields.py](src/flightsim/terrain/airfields.py)).
  - Den nya versionen bör ha *en* beskrivning av varje byggnad. Kollisionsformen härleds ur den eller kontrolleras mot den i test.
- **Detaljnivåer räknas i pixlar på skärmen, inte i meter.**
  - Bytet mellan sprite och modell (8k) blev fel när upplösningen ändrades, eftersom gränsen var satt i fel enhet. LOD-beslut ska ta upplösning och synfält med i beräkningen.
- **Spikar avgör upplevd fart, inte medelvärden.**
  - En bildruta över budget kostar en hel vsync. Former långt utanför skärmen, för många detaljerade modeller och minnesallokering per bildruta gav spikar som medelvärdet dolde.
  - Den nya versionen ska mäta den 99:e percentilen från början.
- **Arkadflygning behöver hjälpmedel som syns.**
  - Helikoptern blev flygbar först med svävbroms, driftindikator och landningsflare. Hjälpmedlen är en del av flygmodellen, inte ett lager ovanpå.
- **Repriser spelas in som tillstånd, inte som knapptryck.** Det gör dem robusta mot ändringar i flygmodellen och mot strömning av terräng.

---

## 2. Målbild

| | Pyxel-versionen | Ny version |
|---|---|---|
| Upplösning | 256×144 (terräng), 768×432 (överlägg) | 1080p–4K, dynamisk upplösning med uppskalning (FSR/DLSS/TAA) |
| Bildfrekvens | 30 | 60 (90+ som möjlighet) |
| Sikt | 2,5 km | 30–60 km, med atmosfär som tonar bort |
| Terräng | Voxel Space i vinkelrum | Polygonterräng med kontinuerlig LOD och detaljmaterial |
| Moln | Platt lager med täckning | Volymetriska moln som man kan flyga igenom |
| Modeller | Lådor och prismor, 8–100 ytor | Modellerade i Blender, 5–50k trianglar närmast, 3–4 LOD-steg |
| Ljus | Palett per tid på dygnet | Fysikbaserat ljus, sol och måne, skuggor, himmel som räknas fram |
| Målmaskin | Laptop | Speldator i mellanklass (GTX 1660/RX 6600 för 1080p/60). Steam Deck som nedre gräns |

### Stil: realistiskt eller stiliserat?

| | Fotorealism | Stiliserad modern (rekommenderas) |
|---|---|---|
| Exempel | Microsoft Flight Simulator | *Sky: Children of the Light*, *Tiny Glade*, *A Short Hike* i 3D, *Ultrawings* |
| Innehåll | Kräver satellitdata eller enorma mängder material | Procedurella material och enkla former räcker långt |
| Procedurell värld | Avslöjas lätt: brus ser inte ut som riktig geologi | Brus *är* stilen, och det fungerar |
| Prestanda | Dyr: täta texturer, tung vegetation | Billigare: färre texturer, mer i shaders |
| Åldras | Snabbt | Långsamt |

Rekommendation: **stiliserad realism**. Riktigt ljus, atmosfär och moln, men material med tydliga färgfält, mjuka former och en begränsad, tidsberoende färgpalett. Det är en naturlig fortsättning på Pyxel-versionens palett och är rimligt för ett litet team.

---

## 3. Arkitektur

Det viktigaste beslutet är att dela upp spelet i en **kärna utan grafik** och ett **skal** som ritar och tar emot input.

```
┌──────────────────────────── skal (motorn) ────────────────────────────┐
│ rendering · ljud · input · UI · kamera · effekter · plattform          │
└──────▲───────────────────────▲──────────────────────────▲─────────────┘
       │ höjder, material,     │ tillstånd per tick       │ händelser
       │ platser per ruta      │ (position, attityd …)    │ (landning, krasch …)
┌──────┴──────────┐  ┌─────────┴──────────┐  ┌────────────┴───────────┐
│ världsgenerator │  │ simulering         │  │ spelregler             │
│ brus, biom,     │◄─┤ flygmodell, kolli- │◄─┤ utmärkelser, tidslopp, │
│ floder, platser │  │ sion, trafik, väder│  │ bränsle, upptäckter    │
└─────────────────┘  └────────────────────┘  └────────────────────────┘
                     kärna: ren kod, deterministisk, testbar utan fönster
```

- **Världsgeneratorn** svarar på frågor som "höjd, biom och fukt för den här rutan" och "platser i den här cellen". Den har inget tillstånd utöver cacheminne. Samma funktion används för rendering, kollision, karta och tester.
- **Simuleringen** körs med fast tidssteg (120 Hz) och är helt fri från grafik. Kameran interpolerar mellan två tick, så bildfrekvensen inte påverkar flygningen.
- **Spelreglerna** lyssnar på händelser, som i dag (`EventLog`), och sparar tillstånd per seed.
- **Skalet** är utbytbart. Det gör motorvalet (avsnitt 4) mindre avgörande och låter kärnan testas och köras utan fönster, för benchmark, generering av kartbilder och CI.

### Gränssnittet mellan kärna och skal

| Alternativ | För | Emot |
|---|---|---|
| Kärnan i samma språk som motorn (t.ex. C# i Godot/Unity) | Enkelt, inget glapp | Låst till motorn. Determinism beror på motorns runtime |
| **Kärnan som bibliotek i Rust eller C++, bunden till motorn** (rekommenderas) | Snabb, trådsäker, motoroberoende, går att testa och köra utan fönster | Bindningar att underhålla, två byggsystem |
| Kärnan som egen process (IPC) | Total isolering | Latens och kopiering av terrängdata, krångligt |

---

## 4. Motor och språk

### Jämförelse

| | Godot 4 | Unreal Engine 5 | Unity 6 | Bevy (Rust) | Egen motor (wgpu/Vulkan) | Webb (Three.js/Babylon + WebGPU) | Python (Panda3D/moderngl) |
|---|---|---|---|---|---|---|---|
| Grafiktak | Bra: atmosfär, volymetrisk dimma och SDFGI. Moln och terräng gör man själv | Högst: Nanite, Lumen, Virtual Shadow Maps, Sky Atmosphere, volymetriska moln | Högt med HDRP. URP räcker för stiliserat | Medel, växer snabbt | Så högt man orkar | Medel. WebGPU höjer taket | Lågt–medel |
| Oändlig procedurell terräng | Egen lösning (clipmap-shader eller mesh-chunkar). Plugins som Terrain3D är byggda för redigerade världar | Landscape är byggt för redigerade världar. Körtidsgenerering kräver Procedural Mesh, Voxel Plugin eller egen kod | Terrain-systemet passar dåligt. Egen mesh-lösning | Egen lösning, flexibelt | Egen lösning, full kontroll | Egen lösning | Egen lösning |
| Iterationsfart | Snabb: liten motor, snabb start | Långsam: C++-byggen, stor motor | Medel | Medel (Rust-kompilering) | Långsam i början, snabb sen | Snabb | Snabb, men prestandan tar slut |
| Licens | MIT | 5 % royalty över 1 MUSD | Prenumeration över intäktsgräns, historiskt osäker | MIT/Apache | – | – | – |
| Teamstorlek | 1–5 | 3–50 | 2–20 | 1–5 | 1–3 med mycket tid | 1–3 | 1 |
| Plattformar | PC, Mac, Linux, Steam Deck, mobil, webb (begränsat) | PC, konsol, mobil | Bred | PC, Mac, Linux, webb | Det man bygger | Webbläsare | PC |
| Risk | Låg–medel | Medel (komplexitet) | Medel (licens) | Medel–hög (API ändras) | Hög | Medel (prestanda) | Hög (prestanda) |

### Bedömning

- **Unreal Engine 5** ger det snyggaste resultatet snabbast när det gäller ljus, himmel, moln och vatten. Men dess världsverktyg är byggda för redigerade världar, och en oändlig värld som genereras under spelets gång går emot motorns grundantaganden: World Partition, Landscape och förbakade data. Det passar ett större team och en värld som kan vara stor men ändlig, till exempel 64×64 km som bakas ut i förväg från seeden.
- **Godot 4** är litet, öppet och snabbt att arbeta i. Det saknar färdiga lösningar för stora världar, men de lösningarna skulle ändå behöva skrivas för en oändlig värld. Med kärnan i Rust via GDExtension blir prestanda och determinism inget problem. Det passar ett litet team som vill ha kontroll.
- **Unity** hamnar mellan de två, utan tydlig fördel här.
- **Bevy** är tilltalande om allt ska vara Rust, men renderaren och verktygen är fortfarande omogna för ett spel med hög grafisk ambition.
- **En egen motor** ger bäst terräng (GPU-driven, clipmaps, egna moln) men kräver allt annat också: UI, ljud, input, plattformsstöd. Det är bara rimligt om motorn i sig är målet.
- **Webben** ger räckvidd. Man delar en länk i stället för en installation. Men den ger lägre prestandatak och svårare trådning. Det passar en lättare demoversion av samma kärna kompilerad till WebAssembly.
- **Python** räcker inte för hög upplösning, tät vegetation och 60 fps. Pyxel-versionen visar var gränsen går.

**Rekommendation: Godot 4 som skal och en kärna i Rust** (världsgenerator, flygmodell, kollision, trafik och spelregler), bunden med GDExtension. Kärnan kan senare flyttas till Unreal eller Bevy om ambitionen växer, eller kompileras till WebAssembly för en webbdemo.

Vill man i stället ha högsta möjliga grafik tidigt, och kan acceptera en ändlig men mycket stor värld, är **Unreal Engine 5 med samma Rust-kärna** (via C-gränssnitt) det starkaste alternativet.

---

## 5. Världen

### 5.1 Determinism – regler

1. **Allt som påverkar spelet beräknas på CPU i kärnan:** höjd, biom, vatten, platser och kollision. GPU:n får bara lägga till *kosmetiska* detaljer, som småbuckligt mikrobrus i normaler och vågor, som aldrig påverkar kollision eller placering.
2. **Brus och slump är hashbaserade:** slumptalen bildas från (seed, cell, salt) med en heltalshash. Ingen global slumpgenerator och ingen ordningsberoende.
3. **Flyttal:** `f64` för världskoordinater, ingen `fast-math`, och egna implementationer av `sin`, `exp` och liknande där resultatet styr placering. Plattformarnas matematikbibliotek kan skilja i sista biten. För helt plattformsoberoende resultat kan positioner kvantiseras till heltal (till exempel cm) innan de används som nyckel.
4. **Generatorn har ett versionsnummer:** sparfiler och utmärkelser knyts till (seed, generatorversion), så att en förbättrad generator inte flyttar sparade platser och vrak.
5. **Regressionstest:** fasta seeds jämförs mot sparade kontrollsummor över höjd, biom och platser.

### 5.2 Representation

| | Höjdfält (2,5D) – rekommenderas | Voxlar / SDF (3D) | Hybrid |
|---|---|---|---|
| Former | Berg, dalar, kust. Inga överhäng | Överhäng, grottor, valv | Höjdfält plus enstaka 3D-objekt (valv, klippor) |
| Kostnad | Låg: 1 värde per punkt, enkel LOD | Hög: minne, meshning (marching cubes/dual contouring), sömmar mellan LOD | Medel |
| Kollision | Trivial och exakt | Kräver fältuppslag eller mesh | Höjdfält plus enkla former |
| Passar flygspel | Ja. Från luften syns överhäng sällan | Bara om grottflygning är en kärnfunktion | Ja, för landmärken |

Rekommendation: **höjdfält**, kompletterat med **landmärken som 3D-modeller** (valv, raukar, vulkaner) som placeras av generatorn.

### 5.3 Generering

Pyxel-versionens pipeline fungerar och behålls i stort: kontinenter, domänförvrängt fBm, ridged noise för bergskammar, biom från höjd, fukt och lutning, samt floder som nollkurvor i ett eget brusfält. Det som ändras är detaljrikedomen och hur trovärdig geologin ser ut.

| Metod för trovärdiga berg | Lokal? (krav för oändlig värld) | Kvalitet | Kostnad |
|---|---|---|---|
| fBm + ridged + domain warp (i dag) | Ja, punktvis | Bra på avstånd, "brusig" nära | Låg |
| **Analytisk erosion** (brus vars oktaver dämpas och böjs av lutningen, så kallad eroded fBm eller gradient trick) | Ja, punktvis | Raviner och åsar som ser eroderade ut | Låg–medel |
| Simulerad hydraulisk erosion per region | Nej: kräver hela regionen och sömmar mellan regioner | Bäst nära | Hög. Måste köras i stora block med överlapp och blandas vid kanterna |
| Stämplar (riktiga höjddata eller handgjorda former) placerade av brus | Ja | Mycket bra för landmärken | Medel: tillgångar att ta fram |

Rekommendation: **analytisk erosion** som grund, plus **stämplar** för särskilda landmärken. Simulerad erosion kan komma senare, som efterbearbetning av block om 4×4 km med 512 m överlapp.

**Floder:** Pyxel-versionens metod med nollkurvor i ett eget fält, utskurna med dalprofil, ger floder utan källor och utlopp. Två alternativ:

| | Nollkurvor (i dag) | Flödesackumulering per region |
|---|---|---|
| Realism | Floder utan riktning, kan gå uppför | Rinner nedför, förgrenar sig, mynnar i sjö eller hav |
| Lokal | Ja | Nej: regionvis med kända in- och utflöden vid gränserna |
| Rekommendation | Startpunkt | Mål: hierarkiskt flodnät på en grov skala (t.ex. 1 km) som bestämmer var dalar skärs ut på finare skala |

**Upplösning:** 1 m mellan höjdpunkter närmast kameran, upp till 512 m längst bort, i en kedja av detaljnivåer. Allt finare än 1 m är GPU-detalj.

### 5.4 Strömning och precision

- **Rutor (tiles)** på 256×256 punkter per detaljnivå genereras av en trådpool i kärnan, i ordning efter avstånd och färdriktning. Pyxel-versionens strömning med processer och ringbuffert per nivå fungerar som förlaga.
- **Flytande origo:** skalet ritar alltid i ett koordinatsystem nära kameran och flyttar origo varje 5–10 km. Kärnan räknar i `f64` världskoordinater. Utan det skakar modellerna redan några tiotals km från start med `f32`.
- **Cache på disk** för genererade rutor är *inte* nödvändig om genereringen är snabb nog (mål: en ruta på under 5 ms per tråd), men kan läggas till för svaga maskiner.

---

## 6. Rendering

### 6.1 Terräng

| Teknik | Hur | För | Emot |
|---|---|---|---|
| Chunkar med LOD (quadtree, CDLOD) | Fyrträd av mesh-rutor, finare nära kameran, mjuk övergång i vertex-shadern | Beprövad, fungerar i alla motorer, enkel att kollisionstesta | Många draw calls utan GPU-driven rendering. Sömmar måste hanteras |
| **Geometry clipmaps** (rekommenderas) | Fasta ringar av rutnät centrerade på kameran. Höjder läses ur texturer som uppdateras när kameran flyttar sig | Mycket få draw calls, konstant kostnad, passar en höjdfältstextur som strömmas in | Höjdtexturen måste fyllas från kärnan, och uppdateringen måste vara inkrementell |
| GPU-tessellation | Grovt rutnät som delas upp på GPU:n efter avstånd | Fina detaljer nära | Stöds olika väl (Metal, mobil). Svårare med exakt kollision |
| Nanite-liknande (meshlets) | Förgenererade kluster | Extremt detaljerat | Bara Unreal i praktiken, och förutsätter förbakade meshar |
| Voxel Space (i dag) | Strålar per kolumn | Billigt i låg upplösning | Skalar dåligt till 4K, ingen riktig ljussättning |

Rekommendation: **geometry clipmaps** med 8–10 nivåer och 255×255 hörn per nivå. Det räcker till 40+ km sikt. Höjd, normal och biomvikter laddas upp per ruta från kärnan.

### 6.2 Material

| | Splatkartor (färdiga texturer per biom) | **Procedurella material i shader** (rekommenderas) | Virtuella texturer (RVT/SVT) |
|---|---|---|---|
| Hur | Biomvikter blandar 4–16 texturlager, triplanart i branter | Färg, ton och detalj räknas i shadern ur biom, höjd, lutning och brus. Några få detaljtexturer | Terrängens färg bakas till en gigantisk virtuell textur vid behov |
| Utseende | Fotorealistiskt med bra texturer | Stiliserat, sammanhängande, inga upprepningsmönster | Bäst på avstånd, billigt i shadern |
| Kostnad | Minne och texturhämtningar | ALU, lite minne | Komplex infrastruktur |
| Passar stiliserad stil | Medel | Ja | Senare, för prestanda |

Biomernas vikter, lutning, fukt och "närhet till vatten" (för stränder, vass och våta kanter) räknas ut i kärnan och följer med varje ruta.

### 6.3 Himmel, ljus och tid på dygnet

- **Atmosfär:** fysikbaserad spridning (Rayleigh och Mie) med förberäknade tabeller (Bruneton eller Hillaire 2020). Den ger himmelsfärg, solnedgång och dis på avstånd, så att avlägsna berg tonar blå. Den ersätter Pyxel-versionens dimpaletter.
- **Sol, måne och stjärnor** följer tid på dygnet. Natten får månljus och stjärnhimmel, och fyrar och flygfältsljus blir ljuskällor med bloom.
- **Skuggor:** kaskaderade skuggkartor (3–4 kaskader) för modeller och vegetation. Terrängens egenskugga över långa avstånd kommer från en grov horisontkarta eller från skuggkartans yttersta kaskad.
- **Global belysning:** en enkel himmelsljusterm (SH från atmosfären) och omgivningsskugga (SSAO/GTAO) räcker för stiliserad stil. Tyngre GI (SDFGI, Lumen) är valfri.

### 6.4 Moln

| | 2D-lager (i dag) | Billboards/impostors | **Volymetriska, raymarchade** (rekommenderas) |
|---|---|---|---|
| Hur | Täckning per punkt på en höjd | Förrenderade molnbilder i 3D | Täthetsfält (brus + väderkarta) som strålas igenom i halv eller kvarts upplösning, med temporal återanvändning |
| Flyga igenom | Ja, men platt | Dåligt, man ser genom | Ja, med riktig dimma inuti |
| Kostnad | Låg | Låg | 1,5–3 ms på mellanklass-GPU |
| Väder | Täckning | Antal | Täckning, typ (stackmoln, skiktmoln, cumulonimbus), regnzoner |

Volymetriska moln (i stil med *Horizon Zero Dawn*/Nubis) är det enskilt största lyftet för känslan av att flyga. Vädret i kärnan (`Weather` i dag) styr en **väderkarta** som molnen läser: täckning, molntyp och nederbörd. Samma karta ger turbulens och regn i simuleringen.

### 6.5 Vatten

- **Hav:** ett oändligt rutnät som följer kameran. Vågor som summa av Gerstner-vågor (enkelt) eller FFT-spektrum (finare, dyrare), styrda av vindens fart och riktning från vädret.
- **Kust:** ett avståndsfält till stranden från generatorn ger skum, grundare färg och brytande vågor.
- **Sjöar och floder:** plana ytor på sin nivå. Floder får flödesriktning från flodnätet, med normalkartor som rör sig nedströms.
- **Reflektioner:** screen-space-reflektioner plus atmosfären som reservbild. Planar reflection är för dyrt i stor skala.
- Sjöflygplanet och vattenlandningar behåller sin logik i kärnan, där vattnet är plant med en sjöhävningsterm.

### 6.6 Vegetation, byggnader och platser

- **Placering:** som i dag, deterministiskt per cell (Poisson-skivor per biom med täthet från fukt och lutning). Inget träd hamnar på en bana, en väg, ett hus eller i vatten.
- **Ritning:** GPU-instansering med indirekta draw calls. Kärnan levererar positionsbuffertar per ruta. Kedjan är modell (LOD0–2) → **impostor** (bild från 8–16 vinklar, oktaedrisk) → **ingen alls**, och skogen bärs av terrängens material på avstånd. Pyxel-versionens sprite-billboards är förlagan till impostorerna.
- **Byggnader:** modulära delar (väggar, tak, fönster) som kombineras procedurellt per by, eller ett bibliotek av 20–40 färdiga hus med variationer i färg och tak. Fyrar, hangarer, torn, broar och helikopterplattor är egna modeller med kollisionsformer som tas fram ur samma källa (se 7.3).
- **Vägar mellan byar och flygfält** är en ny, naturlig funktion. De är splines på ett grovt nät som undviker branter, ritas som en dekal på terrängen och plattar ut marken längs vägen.

### 6.7 Väder och effekter

- Regn och snö som GPU-partiklar runt kameran, i världskoordinater (samma lärdom som i dag: annars "faller" regnet åt fel håll när man tippar). Vått material (mörkare, blankare) och vattendroppar på cockpitglaset.
- Dimma i dalar på morgonen (höjdbaserad dimma i atmosfären).
- Rök, eld och damm som partikelsystem. Vrak röker med samma livslängd som i dag.
- Kondensstrimmor från vingspetsar vid hög g-belastning och fuktig luft. Rotorvirvel som blåser upp damm och vatten under helikoptrar.

---

## 7. Modeller och innehåll

### 7.1 Flygplan och helikoptrar

- **Pipeline:** Blender → glTF 2.0 → motorn. Namngivna noder för rörliga delar: skevroder, höjdroder, sidoroder, klaffar, landningsställ, propeller, rotorer, dörrar. En sidofil (TOML/JSON) per typ kopplar noderna till flygmodellens kontrollytor och anger flygdata, ljud och kamerapunkter.
- **Detaljnivåer:** LOD0 (cockpit och närbild, 20–50k trianglar), LOD1 (5–10k), LOD2 (1–2k), samt impostor för trafik på långt håll.
- **Cockpit:** en enkel 3D-cockpit med fungerande instrument (render-to-texture eller instrument ritade i shader) ersätter HUD:en i cockpitvy. HUD:en finns kvar som val.
- **Propellrar och rotorer:** riktiga blad vid låga varvtal och en suddig skiva (alfablandad textur) vid höga, med en stroboskopeffekt som val.
- **Skador och vrak:** en vrakvariant per typ (avbruten vinge, böjd stjärt, sotiga ytor), som i dag men modellerad. Senare kan delar lossna vid krasch och bli fysikobjekt.
- **Livrén:** färger som parametrar i materialet (kropp, detaljer, nummer), så att spelaren kan välja och trafiken får variation.

### 7.2 Typer från början

| Kategori | Typer (som i dag, utökat) |
|---|---|
| Flygplan | Skolflygplan, akrobatplan, fraktplan, sjöflygplan. Senare: bushplan (STOL), jetplan |
| Helikoptrar | Spaningshelikopter, transporthelikopter, tandemrotor. Senare: räddningshelikopter med vinsch |

Varje typ beskrivs med data (massa, vingyta, lyftkurva, dragkraft, tröghet, styrhjälp), så att nya typer i första hand är innehåll, inte kod.

### 7.3 En källa per byggnad

Varje byggnad har en beskrivning i kärnan: mått, kollisionsform (lådor, cylindrar, välvt tak) och "intressepunkter" (porten, tornets hytt, fyrens lykta). Modellen i Blender görs efter samma mått, och ett test i CI laddar modellen och kontrollerar att dess omslutande form ryms i kollisionsformen inom en tolerans. Det förhindrar att modell och kollision glider isär, som hände med hangarerna i Pyxel-versionen.

---

## 8. Flygmodell

### 8.1 Alternativ

| | Punktmassa med regler (i dag) | **Stel kropp med förenklad aerodynamik** (rekommenderas) | Tabellbaserad (JSBSim/YASim-stil) |
|---|---|---|---|
| Hur | Fart, attityd och regler för lyft, stall och sjunk | Sex frihetsgrader. Krafter och moment från vinge, stabilisatorer och roder med lyftkurvor (inklusive stall), motståndspolar och dragkraft | Uppmätta koefficienttabeller per flygplan |
| Känsla | Arkad, förutsägbar | Trovärdig: överstegring, spinn, sidvind, markeffekt uppstår av sig själva | Realistisk, kräver data |
| Arbete per typ | Lite (några parametrar) | Medel: 15–30 parametrar per typ | Mycket |
| Hjälpmedel | Inbyggda | Ett lager ovanpå: autotrimning, stall-varning, koordinerade svängar, "enkel/normal/realistisk" | Svårt att lägga till utan att förstöra |

Rekommendation: **stel kropp med förenklade ytor och ett tydligt hjälplager.**

- **Läget Enkel** motsvarar dagens känsla. Kontrollerna styr önskad svängfart och attityd, och en regulator räknar fram roderutslagen. Överstegring finns men varnar tidigt och går lätt att häva.
- **Läget Normal** ger direkta roder med trimning och dämpning.
- **Läget Realistisk** stänger av hjälpen och lägger till spinn, P-faktor och vridmoment.

Simuleringen körs i kärnan med fast tidssteg (120 Hz) och semi-implicit integration. Motorns fysikmotor används *inte* för flygningen, eftersom det ger dålig kontroll och sämre determinism. Den kan däremot användas till lösa delar och vrak.

### 8.2 Helikoptrar

Samma uppdelning: **Enkel** som i dag (svävbroms mot marken, drift, landningsflare, kollektiv som stigfart), och **Realistisk** med kollektiv som rotorkraft, cyklisk som skivlutning, svansrotor mot vridmoment, markeffekt, translationellt lyft och virvelringstillstånd. Den gemensamma modellen är en rotorskiva som kraftkälla. Hjälpmedlen reglerar skivan.

### 8.3 Kollision och mark

- **Mot terrängen:** exakta uppslag i höjdfältet från kärnan, med flera kontaktpunkter per flygplan (hjul, vingspetsar, stjärt, rotorbladens spetsar), som `ROTOR STRIKE` i dag.
- **Mot byggnader och träd:** enkla former (lådor, cylindrar, kapslar) från byggnadsbeskrivningarna, med svept test mellan tick, så att snabba plan inte kan passera igenom.
- **Landningsställ:** fjädrar och dämpare per hjul, friktion och bromsar. Markytor från biomet (asfalt, gräs, sand, snö, vatten) ger olika friktion och rullmotstånd.

---

## 9. Platser att upptäcka och interagera med

### 9.1 Platstyper

| Plats | Upptäcka | Interagera | Finns i dag |
|---|---|---|---|
| Flygfält | Synligt på karta och kompass | Landa, tanka, byta typ, tidslopp, flyga förbi tornet | Ja |
| Helikopterplattor | Nära | Landa, tanka | Ja |
| Fyrar | Flyg nära | Upptäcks, lampa nattetid | Ja |
| Broar | Flyg nära | Flyg under | Ja |
| Byar | Synliga | Leveranser (se 9.2), vägar mellan dem | Delvis |
| Landmärken (valv, vattenfall, vulkan, vrak av fartyg, ruiner) | Flyg nära, foto | Unika utmaningar: flyg genom valvet, fotografera vattenfallet | Nej |
| Fjälltoppar | Synliga | Landa på toppen (helikopter) | Delvis |
| Utsiktsplatser | Foto | Fotouppdrag med motiv och vinkel | Nej |

Generatorn placerar platser per cell, som i dag, med en sannolikhet som beror på terrängen (vattenfall där floder korsar branter, valv på kustklippor). Varje plats har ett stabilt id från (seed, typ, cell) som används i sparfiler.

### 9.2 Uppdrag

Uppdrag är ett tunt lager ovanpå platserna och kräver ingen manuskriven berättelse:

- **Frakt:** flyg last eller passagerare mellan flygfält, plattor och byar. Kräver landningar på okända platser och bränsleplanering.
- **Spaning och räddning:** hitta en nödraket eller ett vrak i ett område (helikopter), med vinsch eller landning.
- **Foto:** fotografera ett landmärke inifrån en given sektor och höjd.
- **Tidslopp:** som i dag, men med banor som också går genom dalar och under broar.
- **Brandbekämpning** (sjöflygplan): skumma vatten ur en sjö och släpp det över en brand. Senare tillägg.

### 9.3 Utforskningens belöningar

Karta med "dimma" över det man inte sett, en loggbok med upptäckta platser och utmärkelser (som i dag, utökat), samt fler typer och livréer som låses upp av upptäckter och uppdrag.

---

## 10. Kamera, gränssnitt, ljud

- **Kameror:** cockpit med fri blick (huvudrörelser), jaktvy med fjädring, extern omloppsbana, torn, fri kamera och fotoläge med objektivval (brännvidd, bländare, skärpedjup). Fotoläget får mycket mer att arbeta med i ny grafik.
- **Repris:** tillståndsinspelning i 30 Hz (position, attityd, roder, motorvarv, rotorvinkel, händelser) med interpolation, som i dag. Inspelningen är liten (kB per minut) och oberoende av terrängen. Alternativet, att spela in knapptryck och simulera om, kräver bitexakt determinism i hela simuleringen inklusive strömningsordning. Det är ett onödigt krav.
- **Gränssnitt:** vektorbaserat och skalbart (Godots Control-noder eller motsvarande), med HUD-symbolik i en stil som matchar spelet, inte i pixelfont. Kartan ritas från generatorns grova nivå i ett separat fönster med zoom, platser och fältlinjer.
- **Ljud:** motorljud från lager av inspelade varvtal som blandas efter varv och last, propeller- och rotorljud efter bladfrekvens, vindbrus efter fart och anfallsvinkel, stall-varnare, regn mot glaset, samt dopplereffekt för trafik. Efterklang i dalar och hangarer.
- **Inmatning:** tangentbord och mus, handkontroll och joystick/HOTAS (SDL eller motorns stöd), med kurvor och dödzoner per axel.

---

## 11. Test och verktyg

| Område | Hur |
|---|---|
| Determinism | Kontrollsummor för höjd, biom och platser för fasta seeds på alla plattformar i CI |
| Generator | Egenskapstester: banor plana, inga träd i vatten, floder rinner nedför, platser på torr mark (många finns redan i Pyxel-versionen och kan översättas) |
| Flygmodell | Scenarier utan fönster: stallfart, stigfart, svängradie och toppfart per typ inom tolerans. Mjuk landning, spinn och återhämtning |
| Kollision | Byggnadsbeskrivning mot modellens omslutande form (7.3). Svepta tester för snabba plan |
| Rendering | Referensbilder från fasta kameror, jämförda med tolerans |
| Prestanda | Inspelade flygningar spelas upp utan input, med budget per del (12) och larm på den 99:e percentilen |
| Verktyg | Fristående kartvisare för generatorn (som `flightsim-map` i dag), parameterpanel med direkt uppdatering, felsökningsvy för LOD och för vilka rutor som laddats |

---

## 12. Prestanda

Budget per bildruta i 1080p/60 på en mellanklass-GPU (16,6 ms totalt, GPU och CPU parallellt):

| Del (GPU) | ms |
|---|---|
| Terräng (clipmaps, material) | 2,5 |
| Vegetation och byggnader | 2,5 |
| Skuggor | 2,0 |
| Moln (volymetriska, temporala) | 2,5 |
| Atmosfär, vatten | 1,5 |
| Efterbehandling (TAA/uppskalning, bloom, tonmappning) | 1,5 |
| Modeller, partiklar, UI | 1,5 |
| Marginal | 2,6 |

| Del (CPU, huvudtråd) | ms |
|---|---|
| Simulering (120 Hz, 2 tick per bild) | 0,5 |
| Kultning, LOD-val, draw calls | 3 |
| Spelregler, UI, ljud | 1 |
| Generering | 0 på huvudtråden (trådpool) |

**Mål:** den 99:e percentilen under 16,6 ms, och ingen bildruta över 33 ms vid normal flygning. Genereringen får aldrig blockera huvudtråden. Hellre en grövre ruta en stund än en spik.

---

## 13. Risker

| Risk | Sannolikhet | Konsekvens | Motåtgärd |
|---|---|---|---|
| Strömning hinner inte med i hög fart eller låg höjd | Medel | Hål eller grova rutor nära | Prioritera efter färdriktning. Visa alltid en grövre nivå. Begränsa topphastigheten nära marken |
| Determinism bryts mellan plattformar | Medel | Olika världar, sparfiler fel | Regler i 5.1, kontrollsummor i CI på Windows, Linux och macOS/ARM |
| Procedurell värld ser enformig ut | Hög | Utforskningen tappar värde | Landmärken, analytisk erosion, flodnät, regionala biomvarianter, stämplar |
| Innehållsmängden (modeller) blir för stor för teamet | Hög | Förseningar | Stiliserad stil, modulära byggnader, få men genomarbetade typer |
| Volymetriska moln för dyra på svaga datorer | Medel | Låg bildfrekvens | Kvalitetsnivåer: halv eller kvarts upplösning, 2D-reserv |
| Flygmodellen blir för svår | Medel | Kärnan "lättillgänglig" förloras | Hjälplagret är standard. Tester av känslan med nya spelare tidigt |
| Precision långt från origo | Låg (med flytande origo) | Skakningar | Flytande origo från dag ett |
| Motorbyte behövs senare | Låg–medel | Omskrivning | Kärnan i Rust är motoroberoende |

---

## 14. Implementationsplan

Varje fas ska resultera i något körbart, som i Pyxel-versionen. Faserna bygger på rekommendationerna ovan: en kärna i Rust, Godot 4 som skal, höjdfält med clipmaps, en stiliserad realistisk stil och en flygmodell med hjälplager.

### Översikt

| Fas | Innehåll | Klart när |
|---|---|---|
| 0 – Grund och spikar | Repo, CI, bindning Rust–Godot, clipmap, determinism, flytande origo | En fri kamera över oändlig brusterräng i 60 fps. Samma kontrollsumma på tre plattformar |
| 1 – Världsgeneratorn | Höjd med analytisk erosion, biom, strömning i trådpool, platser per cell, kartverktyg | Kartverktyget visar samma värld som spelet. Rutor hinner laddas i 300 m/s |
| 2 – Världens utseende | Procedurella material, atmosfär och tid på dygnet, hav och sjöar, skuggor, efterbehandling | Utsikt över berg, sjöar och kust med dis på 30 km inom GPU-budget |
| 3 – Flygning | Flygmodell med hjälplager, skolflygplanet, kameror, kollision, krasch | Starta, flyga, landa på ett genererat flygfält och krascha |
| 4 – Platser och byggnader | Byggnadsbeskrivningar, flygfält med hangar och torn, byar, fyrar, plattor, vegetation, upptäckter | Hitta och landa på ett flygfält. Upptäcka en fyr. Flyga lågt över skog |
| 5 – Moln och väder | Väder i kärnan, volymetriska moln, regn och vått material | Flyga genom ett regnväder in i molnen och ut igen |
| 6 – Flygplan och helikoptrar | Datadrivna typer, sjöflygplan, helikoptermodell, cockpits, ljud | Alla sju typer flygbara i lägena Enkel och Normal |
| 7 – Spelsystem | Bränsle, tidslopp, utmärkelser, sparfiler, menyer och HUD | Allt som finns i Pyxel-versionen finns igen |
| 8 – Repris och foto | Tillståndsinspelning, reprisens kameror, fotoläge med objektiv | Se om en landning från tornet och spara en bild i 4K |
| 9 – Mer värld | Flodnät, broar, vägar, landmärken | Följa en flod från fjället till havet och flyga under en bro |
| 10 – Trafik och uppdrag | Trafik med plan och helikoptrar, frakt, räddning, fotouppdrag | En kvälls spelande med egna mål |
| 11 – Polish och utgåva | Kvalitetsnivåer, Steam Deck, tillgänglighet, paketering, webbdemo | 60 fps på målmaskinen, 30 fps på Deck, ett installerbart spel |

### Projektstruktur

```
flightsim/
├── Cargo.toml                 # Rust-arbetsyta
├── core/
│   ├── worldgen/              # brus, höjd, erosion, biom, floder, platser, rutor
│   ├── sim/                   # flygmodell, helikopter, kollision, väder, trafik
│   ├── game/                  # händelser, utmärkelser, bränsle, tidslopp, uppdrag, sparfiler
│   └── shapes/                # byggnadsbeskrivningar: mått, kollision, intressepunkter
├── bindings/godot/            # GDExtension: kärnan som Godot-noder och -resurser
├── tools/
│   ├── worldmap/              # kartbilder (PNG) för seed och område, utan fönster
│   ├── bench/                 # generering och simulering utan fönster, med percentiler
│   └── impostors/             # bakar impostorbilder för träd och byggnader
├── game/                      # Godot-projektet
│   ├── terrain/               # clipmap-mesh och shaders
│   ├── sky/                   # atmosfär, moln, tid på dygnet
│   ├── water/
│   ├── aircraft/              # glTF-modeller, typfiler (TOML), ljud
│   ├── buildings/
│   ├── ui/
│   └── tests/                 # referensbilder, prestandaturer
└── assets-src/                # Blender-filer och ljudkällor
```

---

### Fas 0 – Grund och tekniska spikar

- [ ] Rust-arbetsytan och Godot-projektet enligt strukturen ovan. CI som bygger och testar på Linux, Windows och macOS (ARM).
- [ ] **Spik 1 – bindningen:** en Rust-funktion som fyller ett rutnät med brus och lämnar det till Godot. Mät kostnaden för anrop och kopiering (`PackedFloat32Array` mot direkt uppladdning till en `ImageTexture`) för en ruta på 256×256.
- [ ] **Spik 2 – clipmap:** en clipmap-mesh med 8 nivåer i Godot som läser höjd ur en R32F-textur per nivå. Mät GPU-tiden i 1080p och 1440p.
- [ ] **Spik 3 – determinism:** brus i `f64` med heltalshash. En kontrollsumma över 1024×1024 punkter för tre seeds jämförs i CI på alla tre plattformarna.
- [ ] **Spik 4 – flytande origo:** flyttar scenen när kameran är mer än 5 km från origo. Flyg 200 km rakt fram och kontrollera att modeller och terräng inte skakar.
- [ ] Fri kamera (som spökkameran i dag) och en debugrad med bildtid, GPU-tid och antal rutor som väntar.

**Mätningar som avgör vägen:**
- Tar clipmappen över 4 ms GPU i 1080p, prövas färre hörn per nivå eller en CDLOD-variant innan något annat byggs.
- Tar uppladdningen av en ruta över 1 ms på huvudtråden, laddas den upp i delar över flera bildrutor eller via en separat renderingstråd.
- Skiljer kontrollsummorna mellan plattformar, byts de matematikfunktioner som skiljer mot egna innan fas 1 börjar.

**Klart när:** en fri kamera flyger över oändlig brusterräng i 60 fps, och kontrollsummorna är lika på alla tre plattformarna.

### Fas 1 – Världsgeneratorn

#### 1a – Höjd

- [ ] För över Pyxel-versionens höjd till Rust: kontinenter med domain warping, kuststeg, ridged-berg med bergskedjemask och kullar. Parametrarna i en `TerrainSettings`, laddbar från TOML.
- [ ] Analytisk erosion: oktaver som dämpas och böjs efter lutningen, så att raviner och åsar bildas punktvis.
- [ ] Generatorversion och seed i en `WorldId` som alla sparade data knyts till.
- [ ] `tools/worldmap`: kartbilder per seed och område, med lagren höjd, skuggning, biom och fukt, samt jämförelse sida vid sida mellan två generatorversioner.

**Tester:** deterministiska kontrollsummor, att höjden är kontinuerlig över rutgränser, havsnivå och toppar inom givna gränser, och att erosionen inte ger hål (lokala minima under havet på land) i fler än en viss andel punkter.

#### 1b – Biom och materialdata

- [ ] Fukt, temperatur (från höjd och latitud i brus) och biom, med mjuka vikter i stället för en enda klass per punkt, så att materialen kan blandas.
- [ ] Lutning, krökning (för skuggor i raviner) och avståndsfält till stranden.
- [ ] Data per ruta: höjd (R32F), normal (RG16), biomvikter (RGBA8 × 2) och strandavstånd (R8).

#### 1c – Strömning

- [ ] Rutsystem med 8–10 detaljnivåer och ringbuffert per nivå (som `TerrainLevel` i dag), med en trådpool (`rayon`) och prioritet efter avstånd och färdriktning.
- [ ] Inkrementell uppladdning till clipmappens texturer: bara de rader och kolumner som rullat in.
- [ ] Mätvärdet "inaktuella rutor inom säker radie" (`STALE` i dag) i debugraden och i benchmarken. Det ska vara 0 vid normal flygning.
- [ ] Kollisionsuppslag i kärnan direkt från den finaste nivån, med reserv till direkt generering om rutan inte finns än.

**Klart när:** `tools/bench` flyger i 300 m/s på 50 m höjd utan inaktuella rutor, och en ruta tar under 5 ms per tråd.

#### 1d – Platser per cell

- [ ] För över placeringen av flygfält (med utplanad mark och banmarkering), byar, fyrar och helikopterplattor från Pyxel-versionen, med stabila id:n från (seed, typ, cell).
- [ ] Ett gemensamt API `places_in(område)` som karta, rendering, kollision och spelregler använder.
- [ ] Rensningszoner: inga träd eller hus på banor, plattor och strandkanter.

**Tester:** egenskapstester som i dag (banor plana, plattor torra, hus inte i vatten, samma platser oavsett i vilken ordning de efterfrågas).

### Fas 2 – Världens utseende

#### 2a – Terrängmaterial

- [ ] Clipmap-shadern läser biomvikter och blandar procedurella material: färgfält per biom med ton efter höjd och fukt, detaljbrus nära, triplanar i branter, snö efter höjd, lutning och väderstreck.
- [ ] Mjuk övergång mellan clipmappens nivåer (geomorfning) så att inga ringar syns.
- [ ] Detaljnormaler (GPU-brus) nära kameran. De påverkar bara utseendet.

#### 2b – Himmel och ljus

- [ ] Fysikbaserad atmosfär med förberäknade tabeller: himmel, dis på avstånd (luftperspektiv) och färg på sol- och månljus.
- [ ] Tid på dygnet med sol, måne och stjärnor. Exponering som följer ljuset.
- [ ] Kaskaderade skuggkartor (3–4 kaskader). Skugga från terrängen över långa avstånd via den yttersta kaskaden.

#### 2c – Vatten

- [ ] Hav som ett rutnät som följer kameran, med Gerstner-vågor styrda av en vindparameter.
- [ ] Skum och grunt vatten från strandavståndet. Sjöar på sin egen nivå.
- [ ] Screen-space-reflektioner med himlen som reserv. Dimma under ytan.

#### 2d – Efterbehandling och kvalitet

- [ ] TAA eller FSR 2, tonmappning, bloom, färgkorrigering per tid på dygnet.
- [ ] Kvalitetsnivåer (låg, medel, hög) för clipmappens storlek, skuggor och vattnets detaljer.
- [ ] Tio fasta kameravyer (dal, kust, fjäll, slätt, skymning, natt …) som referensbilder i CI.

**Klart när:** de tio vyerna ser ut som avsett och ryms inom GPU-budgeten för terräng, atmosfär och vatten (avsnitt 12).

### Fas 3 – Flygning

#### 3a – Flygmodellen i kärnan

- [ ] En stel kropp med sex frihetsgrader i `f64`, fast tidssteg 120 Hz och semi-implicit integration.
- [ ] Ytor: vinge (lyftkurva med stall), höjd- och sidostabilisator, roder, klaffar. Motståndspolar, propellerns dragkraft med fart och höjd, samt tröghet.
- [ ] Hjälplager: Enkel (kontrollerna anger önskad svängfart och attityd, med automatisk trimning och koordinerade svängar) och Normal (direkta roder med dämpning).
- [ ] Markkontakt: hjul med fjäder, dämpare, friktion och broms. Markytans typ från biomet.
- [ ] Skolflygplanets data i en typfil (TOML).

**Tester** (utan fönster, som Pyxel-versionens 17 flygtester): stallfart, stigfart, toppfart, svängradie vid 30° och 60° bank, att planet planar ut när spaken släpps i läget Enkel, mjuk landning, start och inbromsning, samt att samma indata ger samma bana.

#### 3b – I spelet

- [ ] Godot-noder som läser kärnans tillstånd och interpolerar mellan två tick.
- [ ] Inmatning för tangentbord, handkontroll och joystick, med kurvor och dödzoner per axel.
- [ ] Jaktkamera med fjädring och cockpitkamera med fri blick.
- [ ] Ett första skolflygplan från Blender (glTF) med rörliga roder, propeller som blir en skiva i högt varv och tre LOD-steg. Tillfälliga ljud.

#### 3c – Kollision och krasch

- [ ] Kontaktpunkter mot terrängen (hjul, vingspetsar, nos, stjärt) och kraschorsaker som i dag.
- [ ] Svept test mellan tick mot enkla former, förberett för byggnader i fas 4.
- [ ] Vrak med rök och eld, kraschvy och respawn.
- [ ] Minimal HUD: fart, höjd, kurs, gas och stallvarning.

**Klart när:** man kan starta från ett genererat flygfält (1d), flyga, landa på ett annat och krascha i ett berg, med en känsla som nya spelare kan ta till sig på några minuter.

### Fas 4 – Platser och byggnader

#### 4a – Byggnadsbeskrivningar

- [ ] `core/shapes`: mått, kollisionsformer (lådor, cylindrar, välvda tak) och intressepunkter (port, hytt, lykta) per byggnad.
- [ ] Ett CI-test som laddar varje glTF-modell och kontrollerar att dess omslutande form stämmer med beskrivningen inom en tolerans.

#### 4b – Flygfält

- [ ] Banmaterial med markeringar, banljus och PAPI som tänds i skymningen.
- [ ] Hangar, flygledartorn och vindstrut som modeller med LOD och impostor. Vindstruten följer vinden.
- [ ] Kollision mot byggnaderna och en nära passage förbi tornet (som *BUZZING THE TOWER* i dag).

#### 4c – Byar, fyrar och plattor

- [ ] Ett bibliotek av modulära husdelar, eller 20–40 färdiga hus med färgvarianter, satta ihop per by av generatorn.
- [ ] Fyrar med roterande ljuskägla nattetid och helikopterplattor med märkning.

#### 4d – Vegetation

- [ ] Placering per cell (Poisson-skivor) med täthet efter biom, fukt och lutning, och rensningszoner från 1d.
- [ ] GPU-instansering (`MultiMesh`) per ruta. LOD-kedja modell → impostor → ingen, med gränser räknade i skärmpixlar.
- [ ] `tools/impostors` som bakar oktaedriska impostorer för träd och byggnader.

#### 4e – Upptäckter och karta

- [ ] Upptäckter av platser (avstånd eller sikt), sparade per `WorldId`.
- [ ] Kartfönster från generatorns grova nivå med platser, flygfältens nummer och dimma över det som inte setts.

**Mätning:** 50 000 synliga träd och 200 byggnader inom budgeten för vegetation och byggnader (2,5 ms).

**Klart när:** man hittar ett flygfält via karta och kompass, landar, och upptäcker en fyr på vägen över skog och byar.

### Fas 5 – Moln och väder

#### 5a – Väder i kärnan

- [ ] Förinställt väder (klart, halvklart, mulet, regn, storm), övergångar över tid och en vindprofil med höjden, som `Weather` i dag.
- [ ] Turbulens från väder, berg (lä och rotor) och moln.
- [ ] En väderkarta som textur (täckning, molntyp, nederbörd) som delas mellan simulering och rendering.

#### 5b – Volymetriska moln

- [ ] Raymarchade moln i kvarts upplösning med temporal återanvändning, täthet från väderkartan och brus, samt ljus från sol och himmel.
- [ ] Molnskuggor på terrängen. Dimma inuti molnen när man flyger igenom.
- [ ] En billig reserv för låg kvalitet (2D-lager med parallax).

#### 5c – Nederbörd

- [ ] Regn och snö som GPU-partiklar i världskoordinater runt kameran.
- [ ] Vått material, droppar på cockpitglaset, blixtar i storm.

**Mätning:** molnen under 2,5 ms i 1080p på mellanklass-GPU, inklusive vid flygning inuti dem.

**Klart när:** man kan flyga in i ett regnväder, genom molnen och ut ovanför dem i solsken, utan hack.

### Fas 6 – Flygplan och helikoptrar

#### 6a – Fler flygplan

- [ ] Akrobatplan, fraktplan och sjöflygplan som typfiler. Sjöflygplanets flottörer får kontakt med vatten och en egen landningsbedömning.
- [ ] Läget Realistisk (utan hjälp, med spinn, P-faktor och vridmoment) som tillval.

#### 6b – Helikoptrar

- [ ] Rotorskivan som kraftkälla: kollektiv, cyklisk och svansrotor, markeffekt och translationellt lyft.
- [ ] Läget Enkel som i Pyxel-versionen: svävbroms mot marken, driftindikator och landningsflare. Läget Normal med direkt cyklisk.
- [ ] Spaningshelikopter, transporthelikopter och tandemrotor.

#### 6c – Modeller och cockpit

- [ ] Modeller för alla typer med rörliga delar, LOD och vrakvariant. Livréer som parametrar i materialet.
- [ ] Cockpit per typ med fungerande instrument.
- [ ] Startskärm med den valda typen roterande, som i dag.

#### 6d – Ljud

- [ ] Motorljud i lager efter varv och last, rotorljud efter bladfrekvens, vind efter fart och anfallsvinkel, stallvarnare och doppler.

**Tester:** toppfart, stallfart och stigfart per typ inom tolerans, svävning i vind, mjuk landning med helikopter, samt sjöflygplanets start och landning på vatten.

**Klart när:** alla sju typer är flygbara i lägena Enkel och Normal och känns olika.

### Fas 7 – Spelsystem

#### 7a – Bränsle

- [ ] Förbrukning efter gas och typ, räckvidd på kartan, tankning på flygfält och plattor.

#### 7b – Tidslopp

- [ ] Ringbanor per flygfält (genererade som i dag, men också genom dalar), rekord per bana och ett spöke av rekordet.

#### 7c – Händelser, utmärkelser och sparfiler

- [ ] Händelselogg i kärnan (landning, krasch, ring, bro, nära passage, upptäckt).
- [ ] Utmärkelserna från Pyxel-versionen överförda, med samma detektorer och syntetiska flygbanor som tester.
- [ ] Profil och sparfiler per `WorldId`: platser, vrak, upptäckter och rekord.

#### 7d – Gränssnitt

- [ ] HUD med instrument, kompass, minikarta och varningar, skalbar efter upplösning.
- [ ] Startskärm, pausmeny, platser, utmärkelser och inställningar (grafik, kontroller, hjälplägen).

**Klart när:** allt som finns i Pyxel-versionen finns igen, utom det som kommer i fas 8–10.

### Fas 8 – Repris och foto

#### 8a – Repris

- [ ] Tillståndsinspelning i 30 Hz av position, attityd, roder, motorvarv, rotorvinkel och händelser, de senaste minuterna i ett ringminne.
- [ ] Uppspelning med interpolation, spolning, fart och kamerorna jakt, cockpit, torn och fri.

#### 8b – Fotoläge

- [ ] Fri kamera och omloppskamera med objektiv (brännvidd, bländare, skärpedjup), tid på dygnet och väder som går att ändra.
- [ ] Sparar i upp till 4K oberoende av skärmens upplösning, utan gränssnitt.

**Klart när:** man kan se om en landning från tornet och spara bilden i 4K.

### Fas 9 – Mer värld

#### 9a – Flodnät och broar

- [ ] Flodnät på grov skala (ungefär 1 km) med flödesackumulering per region och kända in- och utflöden vid gränserna. Dalar skärs ut på finare skala.
- [ ] Rinnande vatten med flödesriktning, vattenfall där floder korsar branter.
- [ ] Broar där vägar eller byar möter floder, med kollision och passager under.

#### 9b – Vägar

- [ ] Vägnät mellan byar och flygfält som undviker branter, ritat som dekal med utplanad mark.

#### 9c – Landmärken

- [ ] Valv, raukar, vulkaner och fartygsvrak som modeller eller stämplar, placerade efter terrängen.

**Tester:** floder rinner aldrig uppför, når hav eller sjö, och är sömlösa över regiongränser. Inga vägar i vatten.

**Klart när:** man kan följa en flod från fjället till havet och flyga under en bro på vägen.

### Fas 10 – Trafik och uppdrag

#### 10a – Trafik

- [ ] Deterministiska tidtabeller mellan flygfält och plattor (som i dag), nu även med helikoptrar och en svävande sista bit.
- [ ] Lanternor, landningsljus och nära passager.

#### 10b – Uppdrag

- [ ] En uppdragstavla på flygfält och plattor: frakt mellan platser, spaning och räddning (helikopter med vinsch), fotouppdrag med motiv, sektor och höjd.
- [ ] Belöningar: nya typer, livréer och utmärkelser.

**Klart när:** en kväll av spelande har egna mål utan att man behöver hitta på dem själv.

### Fas 11 – Polish och utgåva

- [ ] Kvalitetsnivåer och en profil för Steam Deck. Prestandaturer i CI som larmar på den 99:e percentilen.
- [ ] Tillgänglighet: omtilldelning av alla kontroller, tydlig HUD för färgblinda och textstorlek.
- [ ] Paketering för Steam och itch.io, kraschrapporter och sparfilernas versioner.
- [ ] Valfritt: webbdemo med kärnan kompilerad till WebAssembly och en enklare renderare.

**Klart när:** spelet går att installera, håller 60 fps på målmaskinen och 30 fps på Steam Deck.

### Ordning och beroenden

- **Fas 0 avgör allt annat.** Spikarna visar om Godot, clipmappen och determinismen håller innan något byggs ovanpå.
- **Fas 1 och fas 2** kan delvis göras parallellt. Utseendet behöver bara höjd och biomvikter, inte platserna.
- **Fas 3 behöver 1d**, eftersom man ska kunna starta och landa på genererade flygfält.
- **Fas 4 och fas 5** är oberoende av varandra och kan byta plats.
- **Fas 6** bygger på flygmodellen (3a) och byggnaderna (4a, för cockpit- och kollisionsmönstret).
- **Fas 7** kan börja så fort fas 3 finns, men är mest värd när platserna (4) finns.
- **Fas 9** kan börja tidigt i kärnan (flodnätet) men visas först när vatten och material (2c, 2a) är klara.
- **Fas 10** kommer sist: trafik och uppdrag kräver typer, platser och spelsystem.

---

## 15. Rekommendation i korthet

- **Arkitektur:** en deterministisk kärna i **Rust** (världsgenerator, flygmodell, kollision, spelregler) och **Godot 4** som skal för rendering, ljud, input och UI. Unreal Engine 5 är alternativet om grafiken måste vara i toppklass och en mycket stor, ändlig värld räcker.
- **Världen:** höjdfält med analytisk erosion, flodnät och landmärken. Allt beräknas från (seed, cell) med versionsnummer på generatorn och flytande origo.
- **Grafik:** geometry clipmaps, procedurella material i en stiliserad realistisk stil, fysikbaserad atmosfär, volymetriska moln, Gerstner-hav samt instanserad vegetation med impostors.
- **Flygning:** en stel kropp med förenklad aerodynamik och tre lägen för hjälp. Enkel är standard och känns som dagens spel.
- **Innehåll:** modeller från Blender i glTF med rörliga delar och LOD. En gemensam källa för byggnaders mått och kollision.
- **Arbetssätt:** som i Pyxel-versionen, spelbara faser med tester och mätningar, men med budget på den 99:e percentilen och kontrollsummor för determinism från första dagen.
