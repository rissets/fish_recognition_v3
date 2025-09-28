import json
import csv
import os
import re
from collections import defaultdict

def normalize_scientific_name(name):
    """Normalize scientific name for better matching"""
    # Remove extra spaces and standardize
    return re.sub(r'\s+', ' ', name.strip())

def split_name_candidates(value: str):
    """Split raw name fields into clean candidate list."""
    if not value:
        return []
    parts = re.split(r'[;,/]|\bor\b|\band\b', value)
    candidates = []
    for part in parts:
        if not part:
            continue
        cleaned = re.sub(r'\s+', ' ', part).strip().strip('"')
        if cleaned and cleaned not in {'0', '-'}:
            candidates.append(cleaned)
    return candidates

def format_candidate(name: str) -> str:
    """Format candidate name into preferred casing."""
    cleaned = re.sub(r'\s+', ' ', name.strip())
    if not cleaned:
        return ''
    if cleaned.isupper():
        cleaned = cleaned.title()
    else:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned

def preferred_name_from_row(row: dict) -> str:
    """Pick the best Indonesian name from a CSV row."""
    for field in ('nama_daerah', 'nama_umum'):
        for candidate in split_name_candidates(row.get(field, '')):
            formatted = format_candidate(candidate)
            if formatted:
                return formatted
    return ''

def default_fallback_name(scientific_name: str) -> str:
    parts = scientific_name.split()
    epithet = parts[1] if len(parts) > 1 else parts[0]
    epithet = epithet.replace('-', ' ')
    formatted = format_candidate(epithet)
    return f"Ikan {formatted}" if formatted else f"Ikan {scientific_name}"

def apply_name_replacements(name: str) -> str:
    if not name:
        return name

    replacements = [
        ('Ikan Grouper', 'Kerapu'),
        ('Ikan Snapper', 'Kakap'),
        ('Ikan Jack', 'Kuwe'),
        ('Ikan Puffer', 'Buntal'),
        ('Ikan Parrotfish', 'Ikan Kakaktua'),
        ('Ikan Surgeon', 'Botana'),
        ('Ikan Catfish', 'Lele'),
        ('Ikan Tilapia', 'Ikan Nila'),
        ('Ikan Carp', 'Ikan Mas'),
        ('Ikan Bream', 'Katombal'),
        ('Ikan Sunfish', 'Ikan Matahari'),
        ('Ikan Snook', 'Kakap Putih'),
        ('Ikan Filefish', 'Ikan Kulit'),
        ('Ikan Pike', 'Ikan Tombak'),
        ('Ikan Eel', 'Belut'),
        ('Ikan Moray', 'Belut Moray'),
        ('Ikan Triggerfish', 'Ikan Trigger'),
        ('Ikan Barracuda', 'Barakuda'),
        ('Ikan Bonefish', 'Bandeng Laut'),
    ]

    formatted = name
    for old, new in replacements:
        formatted = re.sub(rf'\b{re.escape(old)}\b', new, formatted)

    return formatted

def ensure_unique_name(name: str, scientific_name: str, used_names: set) -> str:
    base = name
    if base not in used_names:
        used_names.add(base)
        return base

    genus_parts = scientific_name.split()
    genus = genus_parts[0] if genus_parts else ''
    species = genus_parts[1] if len(genus_parts) > 1 else ''

    candidates = [
        f"{base} ({genus})" if genus else '',
        f"{base} ({genus} {species})" if genus and species else '',
    ]

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in used_names:
            used_names.add(candidate)
            return candidate

    suffix = 2
    while True:
        candidate = f"{base} ({suffix})"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1

def load_ikan_database(csv_file):
    """Load ikan database and create lookup dictionaries."""
    scientific_to_common = {}
    genus_to_common = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scientific = normalize_scientific_name(row.get('nama_latin', ''))
            if not scientific:
                continue

            preferred = preferred_name_from_row(row)
            key = scientific.lower()
            if preferred and key not in scientific_to_common:
                scientific_to_common[key] = preferred

            genus = scientific.split()[0].lower()
            if preferred and genus and genus not in genus_to_common:
                genus_to_common[genus] = preferred

            for alias in split_name_candidates(row.get('nama_sinonim_latin', '')):
                alias_norm = normalize_scientific_name(alias)
                if not alias_norm:
                    continue
                alias_key = alias_norm.lower()
                if preferred and alias_key not in scientific_to_common:
                    scientific_to_common[alias_key] = preferred

    return scientific_to_common, genus_to_common

def get_indonesian_name(scientific_name, idx, species_dict, genus_dict, existing_idx_to_name):
    """Get Indonesian name from database or provide fallback."""
    normalized = normalize_scientific_name(scientific_name)
    key = normalized.lower()

    if key in species_dict:
        return species_dict[key], 'database'

    for stored_key, value in species_dict.items():
        if stored_key.lower() == key:
            return value, 'database'

    if existing_idx_to_name and idx in existing_idx_to_name:
        return existing_idx_to_name[idx], 'existing'

    # Manual translations for common species not in database
    manual_translations = {
        "Abramis brama": "Bream Eropa",
        "Abudefduf saxatilis": "Ikan Damselfish",
        "Abudefduf sordidus": "Ikan Damselfish Hitam",
        "Acanthocybium solandri": "Tenggiri",
        "Acanthopagrus australis": "Katombal Australia",
        "Acanthopagrus butcheri": "Katombal Hitam",
        "Acanthopagrus latus": "Katombal Lebar",
        "Acanthostracion quadricornis": "Ikan Buntal",
        "Acanthurus chirurgus": "Botana Bergaris",
        "Acanthurus coeruleus": "Botana Biru",
        "Acanthurus dussumieri": "Botana Dussumier",
        "Achoerodus viridis": "Ikan Balong Hijau",
        "Acipenser fulvescens": "Sturgeon Kuning",
        "Acipenser transmontanus": "Sturgeon Putih",
        "Acroteriobatus annulatus": "Ikan Pari Cincin",
        "Aetobatus narinari": "Ikan Pari Elang",
        "Albula vulpes": "Ikan Bonefish",
        "Aldrichetta forsteri": "Ikan Yellowtail",
        "Alectis ciliaris": "Ikan Threadfin",
        "Alopias vulpinus": "Hiu Rubah",
        "Alosa chrysochloris": "Ikan Shad Emas",
        "Alosa pseudoharengus": "Ikan Shad Atlantik",
        "Alosa sapidissima": "Ikan Shad Amerika",
        "Aluterus monoceros": "Ikan Filefish",
        "Aluterus schoepfii": "Ikan Filefish Schopfi",
        "Aluterus scriptus": "Ikan Filefish Bertulisan",
        "Ambloplites rupestris": "Ikan Sunfish Batu",
        "Ameiurus catus": "Ikan Lele Hitam",
        "Ameiurus melas": "Ikan Lele Hitam",
        "Ameiurus natalis": "Ikan Lele Natal",
        "Ameiurus nebulosus": "Ikan Lele Kabur",
        "Amia calva": "Ikan Bowfin",
        "Amniataba percoides": "Ikan Gudgeon",
        "Amphilophus citrinellus": "Ikan Cichlid Midas",
        "Amphiprion percula": "Ikan Nemo",
        "Amphistichus argenteus": "Ikan Senorita",
        "Anchoa mitchilli": "Ikan Anchovy",
        "Anguilla anguilla": "Belut Eropa",
        "Anguilla reinhardtii": "Belut Australia",
        "Anguilla rostrata": "Belut Amerika",
        "Anisotremus davidsonii": "Ikan Porkfish Davidson",
        "Anisotremus surinamensis": "Ikan Porkfish Suriname",
        "Anisotremus virginicus": "Ikan Porkfish Atlantik",
        "Aplodinotus grunniens": "Ikan Drum",
        "Aprion virescens": "Ikan Snapper Jobfish",
        "Archosargus probatocephalus": "Ikan Sheepshead",
        "Archosargus rhomboidalis": "Ikan Sheepshead",
        "Argyrosomus japonicus": "Ikan Mulloway",
        "Ariopsis felis": "Ikan Lele Laut",
        "Arothron hispidus": "Ikan Buntal Berbulu",
        "Arothron stellatus": "Ikan Buntal Bintang",
        "Arripis trutta": "Ikan Kahawai",
        "Arripis truttacea": "Ikan Kahawai",
        "Astronotus ocellatus": "Ikan Oscar",
        "Atherinops affinis": "Ikan Topsmelt",
        "Atractosteus spatula": "Ikan Alligator",
        "Atule mate": "Ikan Yellowtail",
        "Aulostomus maculatus": "Ikan Trumpetfish",
        "Bagre marinus": "Ikan Lele Laut",
        "Bairdiella chrysoura": "Ikan Drum Emas",
        "Balistapus undulatus": "Ikan Triggerfish Orange",
        "Balistes capriscus": "Ikan Triggerfish Abu-abu",
        "Balistes vetula": "Ikan Triggerfish",
        "Balistoides viridescens": "Ikan Triggerfish Hijau",
        "Barbonymus gonionotus": "Tembang",
        "Barbus barbus": "Ikan Barbel",
        "Belone belone": "Ikan Garfish",
        "Bidyanus bidyanus": "Ikan Silver Perch",
        "Blicca bjoerkna": "Ikan Bream Putih",
        "Bodianus rufus": "Ikan Hogfish",
        "Boops boops": "Ikan Bogue",
        "Brachaelurus waddi": "Ikan Blindshark",
        "Brevoortia tyrannus": "Ikan Menhaden",
        "Calamus bajonado": "Ikan Porgy",
        "Campostoma anomalum": "Ikan Stoneroller",
        "Cantherhines pullus": "Ikan Filefish",
        "Canthigaster rostrata": "Ikan Puffer",
        "Caranx bartholomaei": "Ikan Jack",
        "Caranx caninus": "Ikan Jack",
        "Caranx crysos": "Ikan Jack Emas",
        "Caranx hippos": "Ikan Jack",
        "Caranx ignobilis": "Ikan Jack",
        "Caranx latus": "Ikan Jack Lebar",
        "Caranx lugubris": "Ikan Jack",
        "Caranx melampygus": "Ikan Jack",
        "Caranx papuensis": "Ikan Jack Papua",
        "Caranx ruber": "Ikan Jack Merah",
        "Caranx sexfasciatus": "Ikan Jack",
        "Carassius auratus": "Ikan Mas",
        "Carassius carassius": "Ikan Crucian",
        "Carassius gibelio": "Ikan Crucian Gibel",
        "Carcharhinus acronotus": "Hiu Hitam",
        "Carcharhinus brevipinna": "Hiu Sirip Hitam",
        "Carcharhinus falciformis": "Hiu Sutra",
        "Carcharhinus isodon": "Hiu Gigi Halus",
        "Carcharhinus leucas": "Hiu Bulu",
        "Carcharhinus limbatus": "Hiu Sirip Hitam",
        "Carcharhinus melanopterus": "Hiu Sirip Hitam",
        "Carcharhinus obscurus": "Hiu Gelap",
        "Carcharhinus plumbeus": "Hiu Timah",
        "Carcharias taurus": "Hiu Pasir",
        "Carcharodon carcharias": "Hiu Putih",
        "Carpiodes carpio": "Ikan Quillback",
        "Catostomus catostomus": "Ikan Sucker",
        "Catostomus commersonii": "Ikan Sucker Putih",
        "Centrarchus macropterus": "Ikan Sunfish",
        "Centropomus parallelus": "Ikan Snook",
        "Centropomus undecimalis": "Ikan Snook",
        "Centropristis striata": "Ikan Black Sea Bass",
        "Cephalopholis argus":  "Ikan Kerapu",
        "Cephalopholis cruentata": "Ikan Kerapu",
        "Cephalopholis fulva": "Ikan Kerapu",
        "Cephalopholis miniata": "Ikan Kerapu",
        "Chaetodipterus faber": "Ikan Atlantic Spadefish",
        "Chaetodon capistratus": "Ikan Kupu-kupu",
        "Chaetodon ephippium": "Ikan Kupu-kupu",
        "Channa argus": "Gabus",
        "Channa aurolineata": "Gabus",
        "Channa marulius": "Gabus",
        "Channa micropeltes": "Gabus",
        "Channa pseudomarulius": "Gabus",
        "Channa striata": "Gabus",
        "Cheilinus trilobatus": "Ikan Napoleon",
        "Cheilinus undulatus": "Ikan Napoleon",
        "Cheilio inermis": "Ikan Cigarfish",
        "Chitala ornata": "Ikan Clown Knifefish",
        "Chloroscombrus chrysurus": "Ikan Atlantic Bumper",
        "Chlorurus sordidus": "Ikan Parrotfish",
        "Chromis chromis": "Ikan Damselfish",
        "Chrysoblephus laticeps": "Ikan Roman",
        "Cichla ocellaris": "Ikan Peacock Bass",
        "Cirrhitus pinnulatus": "Ikan Hawkfish",
        "Clarias batrachus": "Lele Dumbo",
        "Clarias gariepinus": "Lele Afrika",
        "Clupea harengus": "Ikan Herring",
        "Clupea pallasii": "Ikan Herring Pasifik",
        "Cnidoglanis macrocephalus": "Ikan Tandan",
        "Conger conger": "Belut Laut",
        "Coptodon rendalli": "Ikan Tilapia",
        "Coptodon zillii": "Ikan Tilapia",
        "Coregonus clupeaformis": "Ikan Cisco",
        "Coris julis": "Ikan Rainbowfish",
        "Coryphaena hippurus": "Ikan Mahi-mahi",
        "Cottus bairdii": "Ikan Sculpin",
        "Ctenolabrus rupestris": "Ikan Wrasse",
        "Ctenopharyngodon idella": "Ikan Grass",
        "Cymatogaster aggregata": "Ikan Shiner",
        "Cynoscion nebulosus": "Ikan Spotted Seatrout",
        "Cynoscion regalis": "Ikan Weakfish",
        "Cyprinella lutrensis": "Ikan Red Shiner",
        "Cyprinella spiloptera": "Ikan Spotfin Shiner",
        "Cyprinella venusta": "Ikan Blacktail Shiner",
        "Cyprinus carpio": "Ikan Karper",
        "Cyprinus carpio carpio": "Ikan Karper",
        "Cyprinus rubrofuscus": "Ikan Karper",
        "Dactylopterus volitans": "Ikan Flying Gurnard",
        "Dasyatis pastinaca": "Ikan Pari",
        "Decapterus macarellus": "Ikan Mackerel",
        "Dentex dentex": "Ikan Dentex",
        "Diagramma pictum": "Ikan Painted Sweetlips",
        "Dicentrarchus labrax": "Ikan Sea Bass",
        "Dicentrarchus punctatus": "Ikan Spotted Sea Bass",
        "Dichistius capensis": "Ikan Galjoen",
        "Diodon holocanthus": "Ikan Puffer",
        "Diodon hystrix": "Ikan Puffer",
        "Diplectrum formosum": "Ikan Sand Perch",
        "Diplodus annularis": "Ikan Annular Seabream",
        "Diplodus capensis": "Ikan Cape Seabream",
        "Diplodus cervinus": "Ikan Zebra Seabream",
        "Diplodus holbrookii": "Ikan Spottail Pinfish",
        "Diplodus puntazzo": "Ikan Sharpsnout Seabream",
        "Diplodus sargus": "Ikan White Seabream",
        "Diplodus vulgaris": "Ikan Two Banded Seabream",
        "Dorosoma cepedianum": "Ikan Gizzard Shad",
        "Echeneis naucrates": "Ikan Remora",
        "Elagatis bipinnulata": "Ikan Rainbow Runner",
        "Ellochelon vaigiensis": "Ikan Mackerel",
        "Elops saurus": "Ikan Ladyfish",
        "Embiotoca jacksoni": "Ikan Black Surfperch",
        "Enneacanthus gloriosus": "Ikan Bluespotted Sunfish",
        "Enneacanthus obesus": "Ikan Banded Sunfish",
        "Ephippion guttifer": "Ikan Priacanthid",
        "Epinephelus adscensionis": "Ikan Grouper",
        "Epinephelus coioides": "Kerapu",
        "Epinephelus fasciatus": "Ikan Grouper",
        "Epinephelus fuscoguttatus": "Ikan Grouper",
        "Epinephelus guttatus": "Ikan Grouper",
        "Epinephelus itajara": "Ikan Grouper",
        "Epinephelus labriformis": "Ikan Grouper",
        "Epinephelus lanceolatus": "Ikan Grouper",
        "Epinephelus malabaricus": "Kerapu",
        "Epinephelus marginatus": "Ikan Grouper",
        "Epinephelus merra": "Ikan Grouper",
        "Epinephelus morio": "Ikan Grouper",
        "Epinephelus striatus": "Ikan Grouper",
        "Epinephelus tauvina": "Kerapu",
        "Epinephelus tukula": "Ikan Grouper",
        "Esox americanus": "Ikan Pike",
        "Esox lucius": "Ikan Pike Utara",
        "Esox masquinongy": "Ikan Muskellunge",
        "Esox masquinongy X Esox lucius": "Ikan Tiger Muskellunge",
        "Esox niger": "Ikan Chain Pickerel",
        "Etelis oculatus": "Ikan Snapper",
        "Etheostoma caeruleum": "Ikan Rainbow Darter",
        "Eucinostomus gula": "Ikan Spotfin Mojarra",
        "Euthynnus affinis": "Tuna Mata Besar",
        "Euthynnus alletteratus": "Tuna Sirip Kuning",
        "Fundulus chrysotus": "Ikan Golden Topminnow",
        "Fundulus diaphanus": "Ikan Banded Killifish",
        "Fundulus olivaceus": "Ikan Blackspotted Topminnow",
        "Gadus morhua": "Ikan Cod Atlantik",
        "Galeocerdo cuvier": "Hiu Harimau",
        "Galeorhinus galeus": "Hiu Soupfin",
        "Gambusia affinis": "Ikan Mosquito",
        "Gasterosteus aculeatus": "Ikan Stickleback",
        "Gerres cinereus": "Ikan Mojarra",
        "Ginglymostoma cirratum": "Hiu Karpet",
        "Girella elevata": "Ikan Parore",
        "Girella tricuspidata": "Ikan Opaleye",
        "Gnathanodon speciosus": "Ikan Golden Trevally",
        "Gobiesox rhessodon": "Ikan Clingfish",
        "Gobio gobio": "Ikan Gudgeon",
        "Gobius paganellus": "Ikan Rock Goby",
        "Gymnosarda unicolor": "Ikan Dogtooth Tuna",
        "Gymnothorax funebris": "Belut Laut",
        "Gymnothorax miliaris": "Belut Laut",
        "Gymnothorax moringa": "Belut Laut",
        "Haemulon album": "Ikan Margate",
        "Haemulon aurolineatum": "Ikan Tomtate",
        "Haemulon chrysargyreum": "Ikan Smallmouth Grunt",
        "Haemulon flavolineatum": "Ikan French Grunt",
        "Haemulon parra": "Ikan Sailor Choice",
        "Haemulon plumierii": "Ikan White Grunt",
        "Haemulon sciurus": "Ikan Bluestriped Grunt",
        "Halichoeres bivittatus": "Ikan Slippery Dick",
        "Halichoeres garnoti": "Ikan Yellowhead Wrasse",
        "Halichoeres maculipinna": "Ikan Clown Wrasse",
        "Halichoeres radiatus": "Ikan Puddingwife",
        "Halichoeres semicinctus": "Ikan Rock Wrasse",
        "Hampala macrolepidota": "Sebarau",
        "Hemigymnus melapterus": "Ikan Blackeye Thicklip",
        "Hephaestus fuliginosus": "Ikan Sooty Grunter",
        "Herichthys cyanoguttatus": "Ikan Texas Cichlid",
        "Heterodontus francisci": "Hiu Tanduk",
        "Heterodontus portusjacksoni": "Hiu Tanduk",
        "Hexagrammos decagrammus": "Ikan Kelp Greenling",
        "Hexagrammos stelleri": "Ikan Whitespotted Greenling",
        "Hiodon alosoides": "Ikan Goldeye",
        "Hiodon tergisus": "Ikan Mooneye",
        "Hippoglossus stenolepis": "Ikan Pacific Halibut",
        "Holacanthus bermudensis": "Ikan Blue Angelfish",
        "Holacanthus ciliaris": "Ikan Queen Angelfish",
        "Holocentrus adscensionis": "Ikan Squirrelfish",
        "Hoplias malabaricus": "Ikan Trahira",
        "Hoplopagrus guentherii": "Ikan Red Porgy",
        "Hybognathus hankinsoni": "Ikan Brassy Minnow",
        "Hypanus americanus": "Ikan Southern Stingray",
        "Hypanus sabinus": "Ikan Atlantic Stingray",
        "Hypentelium nigricans": "Ikan Northern Hog Sucker",
        "Hypophthalmichthys molitrix": "Ikan Silver Carp",
        "Hypophthalmichthys nobilis": "Ikan Bighead Carp",
        "Hyporthodus niveatus": "Ikan Snowy Grouper",
        "Hypsypops rubicundus": "Ikan Garibaldi",
        "Ictalurus furcatus": "Ikan Blue Catfish",
        "Ictalurus punctatus": "Ikan Channel Catfish",
        "Ictiobus bubalus": "Ikan Smallmouth Buffalo",
        "Ictiobus cyprinellus": "Ikan Bigmouth Buffalo",
        "Istiophorus albicans": "Ikan Sailfish Atlantik",
        "Istiophorus platypterus": "Ikan Sailfish Indo-Pasifik",
        "Kajikia audax": "Ikan Striped Marlin",
        "Katsuwonus pelamis": "Cakalang",
        "Kuhlia rupestris": "Ikan Flagtail",
        "Kyphosus sectatrix": "Ikan Bermuda Chub",
        "Kyphosus vaigiensis": "Ikan Brassy Chub",
        "Labeobarbus marequensis": "Ikan Redeye Labeo",
        "Labrus bergylta": "Ikan Ballan Wrasse",
        "Labrus mixtus": "Ikan Cuckoo Wrasse",
        "Lachnolaimus maximus": "Ikan Hogfish",
        "Lagocephalus laevigatus": "Ikan Puffer",
        "Lagocephalus lagocephalus": "Ikan Puffer",
        "Lagodon rhomboides": "Ikan Pinfish",
        "Lates calcarifer": "Ikan Kakap",
        "Leiostomus xanthurus": "Ikan Spot",
        "Lepisosteus oculatus": "Ikan Spotted Gar",
        "Lepisosteus osseus": "Ikan Longnose Gar",
        "Lepisosteus platostomus": "Ikan Shortnose Gar",
        "Lepisosteus platyrhincus": "Ikan Florida Gar",
        "Lepomis auritus": "Ikan Redbreast Sunfish",
        "Lepomis cyanellus": "Ikan Green Sunfish",
        "Lepomis gibbosus": "Ikan Pumpkinseed",
        "Lepomis gulosus": "Ikan Warmouth",
        "Lepomis humilis": "Ikan Orangespotted Sunfish",
        "Lepomis macrochirus": "Ikan Bluegill",
        "Lepomis marginatus": "Ikan Dollar Sunfish",
        "Lepomis megalotis": "Ikan Longear Sunfish",
        "Lepomis microlophus": "Ikan Redear Sunfish",
        "Lepomis miniatus": "Ikan Redspotted Sunfish",
        "Lepomis peltastes": "Ikan Northern Sunfish",
        "Lepomis punctatus": "Ikan Spotted Sunfish",
        "Leptocottus armatus": "Ikan Pacific Staghorn Sculpin",
        "Lethrinus nebulosus": "Ikan Spangled Emperor",
        "Lethrinus obsoletus": "Ikan Orange Spotted Emperor",
        "Leuciscus aspius": "Ikan Asp",
        "Leuciscus idus": "Ikan Ide",
        "Leuciscus leuciscus": "Ikan Dace",
        "Lichia amia": "Ikan Leerfish",
        "Lithognathus lithognathus": "Ikan White Steenbras",
        "Lithognathus mormyrus": "Ikan Sand Steenbras",
        "Lobotes surinamensis": "Ikan Tripletail",
        "Loligo Vulgaris": "Cumi-cumi",
        "Lota lota": "Ikan Burbot",
        "Lutjanus analis": "Ikan Snapper",
        "Lutjanus apodus": "Ikan Snapper",
        "Lutjanus argentimaculatus": "Ikan Snapper",
        "Lutjanus argentiventris": "Ikan Snapper",
        "Lutjanus bohar": "Ikan Snapper",
        "Lutjanus campechanus": "Ikan Snapper",
        "Lutjanus carponotatus": "Ikan Snapper",
        "Lutjanus cyanopterus": "Ikan Snapper",
        "Lutjanus decussatus": "Ikan Snapper",
        "Lutjanus ehrenbergii": "Ikan Snapper",
        "Lutjanus fulviflamma": "Ikan Snapper",
        "Lutjanus fulvus": "Ikan Snapper",
        "Lutjanus gibbus": "Ikan Snapper",
        "Lutjanus griseus": "Ikan Snapper",
        "Lutjanus jocu": "Ikan Snapper",
        "Lutjanus johnii": "Ikan Snapper",
        "Lutjanus kasmira": "Ikan Snapper",
        "Lutjanus mahogoni": "Ikan Snapper",
        "Lutjanus monostigma": "Ikan Snapper",
        "Lutjanus novemfasciatus": "Ikan Snapper",
        "Lutjanus rivulatus": "Ikan Snapper",
        "Lutjanus russellii": "Ikan Snapper",
        "Lutjanus sebae": "Ikan Snapper",
        "Lutjanus synagris": "Ikan Snapper",
        "Lutjanus vivanus": "Ikan Snapper",
        "Luxilus chrysocephalus": "Ikan Striped Shiner",
        "Luxilus cornutus": "Ikan Common Shiner",
        "Maccullochella peelii": "Ikan Murray Cod",
        "Macquaria ambigua": "Ikan Golden Perch",
        "Makaira nigricans": "Ikan Blue Marlin",
        "Mayaheros urophthalmus": "Ikan Mayan Cichlid",
        "Megalops atlanticus": "Ikan Tarpon",
        "Megalops cyprinoides": "Ikan Oxeye Herring",
        "Melanogrammus aeglefinus": "Ikan Haddock",
        "Melichthys niger": "Ikan Black Durgon",
        "Menidia menidia": "Ikan Atlantic Silverside",
        "Menticirrhus americanus": "Ikan Southern Kingfish",
        "Menticirrhus littoralis": "Ikan Gulf Kingfish",
        "Menticirrhus saxatilis": "Ikan Northern Kingfish",
        "Merlangius merlangus": "Ikan Whiting",
        "Meuschenia freycineti": "Ikan Sixspine Leatherjacket",
        "Micropogonias undulatus": "Ikan Atlantic Croaker",
        "Micropterus coosae": "Ikan Redeye Bass",
        "Micropterus dolomieu": "Ikan Smallmouth Bass",
        "Micropterus floridanus": "Ikan Florida Bass",
        "Micropterus henshalli": "Ikan Alabama Bass",
        "Micropterus nigricans": "Ikan Largemouth Bass",
        "Micropterus notius": "Ikan Suwannee Bass",
        "Micropterus punctulatus": "Ikan Spotted Bass",
        "Micropterus treculii": "Ikan Guadalupe Bass",
        "Minytrema melanops": "Ikan Spotted Sucker",
        "Monacanthus chinensis": "Ikan Fanbelly Leatherjacket",
        "Monodactylus argenteus": "Ikan Mono",
        "Morone americana": "Ikan White Perch",
        "Morone chrysops": "Ikan White Bass",
        "Morone chrysops X Morone saxatilis": "Ikan Hybrid Striped Bass",
        "Morone mississippiensis": "Ikan Yellow Bass",
        "Morone saxatilis": "Ikan Striped Bass",
        "Moxostoma anisurum": "Ikan Silver Redhorse",
        "Moxostoma carinatum": "Ikan River Redhorse",
        "Moxostoma duquesnei": "Ikan Black Redhorse",
        "Moxostoma erythrurum": "Ikan Golden Redhorse",
        "Moxostoma macrolepidotum": "Ikan Shorthead Redhorse",
        "Moxostoma valenciennesi": "Ikan Greater Redhorse",
        "Mugil cephalus": "Ikan Mullet",
        "Mugil curema": "Ikan White Mullet",
        "Mulloidichthys flavolineatus": "Ikan Yellowstripe Goatfish",
        "Mulloidichthys martinicus": "Ikan Yellow Goatfish",
        "Mustelus canis": "Hiu Smoothhound",
        "Mustelus lenticulatus": "Hiu Spur Dog",
        "Mustelus mustelus": "Hiu Common Smoothhound",
        "Mycteroperca bonaci": "Ikan Black Grouper",
        "Mycteroperca microlepis": "Ikan Gag",
        "Mycteroperca phenax": "Ikan Scamp",
        "Mycteroperca rubra": "Ikan Comb Grouper",
        "Mycteroperca tigris": "Ikan Tiger Grouper",
        "Mycteroperca venenosa": "Ikan Yellowfin Grouper",
        "Myliobatis aquila": "Ikan Eagle Ray",
        "Myliobatis californica": "Ikan Bat Ray",
        "Myrichthys breviceps": "Belut Laut",
        "Myripristis berndti": "Ikan Bigscale Soldierfish",
        "Naso unicornis": "Ikan Bluespine Unicornfish",
        "Nebrius ferrugineus": "Ikan Tawny Nurse Shark",
        "Negaprion brevirostris": "Hiu Lemon",
        "Nematistius pectoralis": "Ikan Roosterfish",
        "Neogobius melanostomus": "Ikan Round Goby",
        "Nocomis biguttatus": "Ikan Hornyhead Chub",
        "Nocomis leptocephalus": "Ikan Bluehead Chub",
        "Nocomis micropogon": "Ikan River Chub",
        "Notemigonus crysoleucas": "Ikan Golden Shiner",
        "Notolabrus celidotus": "Ikan Spotty",
        "Notolabrus fucicola": "Ikan Banded Parrotfish",
        "Notolabrus gymnogenis": "Ikan Crimson Parrotfish",
        "Notolabrus tetricus": "Ikan Blue Parrotfish",
        "Notorynchus cepedianus": "Hiu Broadnose Sevengill",
        "Notropis atherinoides": "Ikan Emerald Shiner",
        "Notropis hudsonius": "Ikan Spottail Shiner",
        "Notropis stramineus": "Ikan Sand Shiner",
        "Noturus flavus": "Ikan Stonecat",
        "Ocyurus chrysurus": "Ikan Yellowtail Snapper",
        "Oligoplites saurus": "Ikan Leatherjacket",
        "Oncorhynchus aguabonita": "Ikan Golden Trout",
        "Oncorhynchus clarkii": "Ikan Cutthroat Trout",
        "Oncorhynchus clarkii clarkii": "Ikan Coastal Cutthroat Trout",
        "Oncorhynchus gorbuscha": "Ikan Pink Salmon",
        "Oncorhynchus keta": "Ikan Chum Salmon",
        "Oncorhynchus kisutch": "Ikan Coho Salmon",
        "Oncorhynchus mykiss": "Ikan Rainbow Trout",
        "Oncorhynchus mykiss X Color variant2": "Ikan Steelhead Trout",
        "Oncorhynchus mykiss X Oncorhynchus clarkii": "Ikan Cutbow Trout",
        "Oncorhynchus nerka": "Ikan Sockeye Salmon",
        "Oncorhynchus tshawytscha": "Ikan Chinook Salmon",
        "Ophiodon elongatus": "Ikan Lingcod",
        "Opisthonema oglinum": "Ikan Atlantic Thread Herring",
        "Opsanus tau": "Ikan Oyster Toadfish",
        "Oreochromis aureus": "Ikan Blue Tilapia",
        "Oreochromis niloticus": "Ikan Nila",
        "Orthopristis chrysoptera": "Ikan Pigfish",
        "Pachymetopon blochii": "Ikan Hottentot",
        "Pagellus acarne": "Ikan Axillary Seabream",
        "Pagrus auratus": "Ikan Snapper",
        "Pagrus pagrus": "Ikan Red Porgy",
        "Parachromis dovii": "Ikan Wolf Cichlid",
        "Parachromis managuensis": "Ikan Jaguar Cichlid",
        "Paralabrax clathratus": "Ikan Kelp Bass",
        "Paralabrax maculatofasciatus": "Ikan Spotted Sand Bass",
        "Paralabrax nebulifer": "Ikan Barred Sand Bass",
        "Paralichthys albigutta": "Ikan Gulf Flounder",
        "Paralichthys californicus": "Ikan California Halibut",
        "Paralichthys dentatus": "Ikan Summer Flounder",
        "Paralichthys lethostigma": "Ikan Southern Flounder",
        "Parapercis colias": "Ikan Blue Cod",
        "Parupeneus cyclostomus": "Ikan Goldsaddle Goatfish",
        "Parupeneus indicus": "Ikan Indian Goatfish",
        "Parupeneus multifasciatus": "Ikan Manybar Goatfish",
        "Parupeneus spilurus": "Ikan Blackspot Goatfish",
        "Pelmatolapia mariae": "Ikan Cichlid",
        "Peprilus triacanthus": "Ikan Butterfish",
        "Perca flavescens": "Ikan Yellow Perch",
        "Perca fluviatilis": "Ikan European Perch",
        "Percalates novemaculeatus": "Ikan Australian Bass",
        "Percina caprodes": "Ikan Logperch",
        "Percopsis omiscomaycus": "Ikan Trout Perch",
        "Phoxinus phoxinus": "Ikan Minnow",
        "Pimelodus maculatus": "Ikan Pimelodus",
        "Pimephales promelas": "Ikan Fathead Minnow",
        "Pimephales vigilax": "Ikan Bullhead Minnow",
        "Platax teira": "Ikan Batfish",
        "Platichthys flesus": "Ikan European Flounder",
        "Platichthys stellatus": "Ikan Starry Flounder",
        "Platycephalus fuscus": "Ikan Dusky Flathead",
        "Platycephalus indicus": "Ikan Bartail Flathead",
        "Platyrhinoidis triseriata": "Ikan Thornback Ray",
        "Plectropomus laevis": "Ikan Coral Trout",
        "Plectropomus leopardus": "Ikan Leopard Coral Trout",
        "Plectropomus maculatus": "Ikan Barred Coral Trout",
        "Pleuronectes platessa": "Ikan European Plaice",
        "Plotosus lineatus": "Ikan Striped Catfish",
        "Poecilia reticulata": "Ikan Guppy",
        "Pogonias cromis": "Ikan Black Drum",
        "Pollachius pollachius": "Ikan Pollack",
        "Pollachius virens": "Ikan Pollock",
        "Polyodon spathula": "Ikan Paddlefish",
        "Pomacanthus arcuatus": "Ikan Gray Angelfish",
        "Pomacanthus paru": "Ikan French Angelfish",
        "Pomadasys commersonnii": "Ikan Javelin Grunter",
        "Pomadasys kaakan": "Ikan Grunter",
        "Pomatomus saltatrix": "Ikan Bluefish",
        "Pomoxis annularis": "Ikan White Crappie",
        "Pomoxis nigromaculatus": "Ikan Black Crappie",
        "Poroderma africanum": "Hiu Pyjama",
        "Prionace glauca": "Hiu Biru",
        "Prionotus evolans": "Ikan Striped Searobin",
        "Prosopium williamsoni": "Ikan Mountain Whitefish",
        "Pseudobatos productus": "Ikan Shovelnose Guitarfish",
        "Pseudolabrus guentheri": "Ikan Günther's Wrasse",
        "Pseudopleuronectes americanus": "Ikan Winter Flounder",
        "Pseudupeneus maculatus": "Ikan Blackspot Goatfish",
        "Pterois volitans": "Ikan Singa",
        "Ptychocheilus grandis": "Ikan Sacramento Pikeminnow",
        "Ptychocheilus oregonensis": "Ikan Northern Pikeminnow",
        "Pylodictis olivaris": "Ikan Flathead Catfish",
        "Rachycentron canadum": "Ikan Cobia",
        "Raja clavata": "Ikan Thornback Ray",
        "Rhabdosargus sarba": "Ikan Yellowfin Bream",
        "Rhincodon typus": "Hiu Paus",
        "Rhinecanthus rectangulus": "Ikan Wedge-tail Triggerfish",
        "Rhinichthys atratulus": "Ikan Blacknose Dace",
        "Rhinichthys cataractae": "Ikan Longnose Dace",
        "Rhinichthys obtusus": "Ikan Western Blacknose Dace",
        "Rhizoprionodon terraenovae": "Hiu Atlantic Sharpnose",
        "Rhomboplites aurorubens": "Ikan Vermilion Snapper",
        "Rutilus rutilus": "Ikan Roach",
        "Rypticus saponaceus": "Ikan Greater Soapfish",
        "Salminus brasiliensis": "Ikan Dorado",
        "Salmo salar": "Ikan Salmon Atlantik",
        "Salmo trutta": "Ikan Brown Trout",
        "Salmo trutta X Salvelinus fontinalis": "Ikan Tiger Trout",
        "Salvelinus alpinus": "Ikan Arctic Char",
        "Salvelinus confluentus": "Ikan Bull Trout",
        "Salvelinus fontinalis": "Ikan Brook Trout",
        "Salvelinus malma": "Ikan Dolly Varden",
        "Salvelinus namaycush": "Ikan Lake Trout",
        "Sander canadensis": "Ikan Sauger",
        "Sander lucioperca": "Ikan Pikeperch",
        "Sander vitreus": "Ikan Walleye",
        "Sander vitreus X Sander canadensis": "Ikan Saugeye",
        "Sarda sarda": "Ikan Atlantic Bonito",
        "Sarotherodon melanotheron": "Ikan Tilapia",
        "Sarpa salpa": "Ikan Salema",
        "Scaphirhynchus platorynchus": "Ikan Shovelnose Sturgeon",
        "Scardinius erythrophthalmus": "Ikan Rudd",
        "Scarus coeruleus": "Ikan Parrotfish",
        "Scarus ghobban": "Ikan Parrotfish",
        "Scarus guacamaia": "Ikan Rainbow Parrotfish",
        "Scarus iseri": "Ikan Striped Parrotfish",
        "Scarus niger": "Ikan Dusky Parrotfish",
        "Scarus psittacus": "Ikan Palenose Parrotfish",
        "Scarus taeniopterus": "Ikan Princess Parrotfish",
        "Scarus vetula": "Ikan Queen Parrotfish",
        "Scatophagus argus": "Ikan Scat",
        "Sciaenops ocellatus": "Ikan Red Drum",
        "Scomber australasicus": "Ikan Mackerel",
        "Scomber japonicus": "Ikan Mackerel",
        "Scomber scombrus": "Ikan Atlantic Mackerel",
        "Scomberoides commersonnianus": "Ikan Talang Queenfish",
        "Scomberoides lysan": "Ikan Doublespotted Queenfish",
        "Scomberomorus cavalla": "Ikan King Mackerel",
        "Scomberomorus commerson": "Ikan Narrow-barred Spanish Mackerel",
        "Scomberomorus maculatus": "Ikan Atlantic Spanish Mackerel",
        "Scomberomorus regalis": "Ikan Cero Mackerel",
        "Scomberomorus sierra": "Ikan Sierra Mackerel",
        "Scorpaena guttata": "Ikan California Scorpionfish",
        "Scorpaenichthys marmoratus": "Ikan Cabezon",
        "Scorpis lineolata": "Ikan Sweep",
        "Scyliorhinus canicula": "Hiu Lesser Spotted Dogfish",
        "Sebastes auriculatus": "Ikan Brown Rockfish",
        "Sebastes caurinus": "Ikan Copper Rockfish",
        "Sebastes melanops": "Ikan Black Rockfish",
        "Sebastes miniatus": "Ikan Vermilion Rockfish",
        "Sebastes mystinus": "Ikan Blue Rockfish",
        "Selar crumenophthalmus": "Ikan Bigeye Scad",
        "Selene setapinnis": "Ikan Atlantic Moonfish",
        "Selene vomer": "Ikan Lookdown",
        "Semotilus atromaculatus": "Ikan Creek Chub",
        "Semotilus corporalis": "Ikan Fallfish",
        "Seriola dumerili": "Ikan Greater Amberjack",
        "Seriola hippos": "Ikan Almaco Jack",
        "Seriola lalandi": "Ikan Yellowtail Amberjack",
        "Seriola rivoliana": "Ikan Longfin Yellowtail",
        "Serranus cabrilla": "Ikan Comber",
        "Serranus scriba": "Ikan Painted Comber",
        "Siganus guttatus": "Ikan Rabbitfish",
        "Sillaginodes punctatus": "Ikan King George Whiting",
        "Sillago ciliata": "Ikan Sand Whiting",
        "Silurus glanis": "Ikan Wels Catfish",
        "Sparisoma aurofrenatum": "Ikan Redtail Parrotfish",
        "Sparisoma chrysopterum": "Ikan Redtail Parrotfish",
        "Sparisoma cretense": "Ikan Parrotfish",
        "Sparisoma rubripinne": "Ikan Redfin Parrotfish",
        "Sparisoma viride": "Ikan Stoplight Parrotfish",
        "Sparus aurata": "Ikan Gilthead Seabream",
        "Sphoeroides maculatus": "Ikan Northern Puffer",
        "Sphoeroides testudineus": "Ikan Checkered Puffer",
        "Sphyraena argentea": "Ikan Atlantic Barracuda",
        "Sphyraena barracuda": "Ikan Great Barracuda",
        "Sphyraena jello": "Ikan Pickhandle Barracuda",
        "Sphyraena novaehollandiae": "Ikan Australian Barracuda",
        "Sphyraena obtusata": "Ikan Obtuse Barracuda",
        "Sphyraena qenie": "Ikan Blackfin Barracuda",
        "Sphyraena viridensis": "Ikan Yellowmouth Barracuda",
        "Sphyrna lewini": "Hiu Kepala Martil",
        "Sphyrna mokarran": "Hiu Martil",
        "Sphyrna tiburo": "Hiu Kepala Martil",
        "Sphyrna zygaena": "Hiu Kepala Martil",
        "Spondyliosoma cantharus": "Ikan Black Seabream",
        "Squalus acanthias": "Hiu Dogfish",
        "Stegastes leucostictus": "Ikan Beaugregory",
        "Stegastes partitus": "Ikan Bicolor Damselfish",
        "Stenotomus chrysops": "Ikan Scup",
        "Stereolepis gigas": "Ikan Giant Sea Bass",
        "Strongylura marina": "Ikan Atlantic Needlefish",
        "Symphodus cinereus": "Ikan Grey Wrasse",
        "Symphodus melops": "Ikan Corkwing Wrasse",
        "Symphodus tinca": "Ikan East Atlantic Peacock Wrasse",
        "Synodus foetens": "Ikan Inshore Lizardfish",
        "Synodus saurus": "Ikan Atlantic Lizardfish",
        "Taeniura lymma": "Ikan Bluespotted Ribbontail Ray",
        "Tandanus tandanus": "Ikan Freshwater Catfish",
        "Tautoga onitis": "Ikan Tautog",
        "Tautogolabrus adspersus": "Ikan Cunner",
        "Terapon jarbua": "Ikan Jarbua Terapon",
        "Thalassoma bifasciatum": "Ikan Bluehead Wrasse",
        "Thalassoma duperrey": "Ikan Saddle Wrasse",
        "Thalassoma lunare": "Ikan Moon Wrasse",
        "Thalassoma pavo": "Ikan Pavonine Wrasse",
        "Thalassoma purpureum": "Ikan Surge Wrasse",
        "Thunnus alalunga": "Tuna Albacore",
        "Thunnus albacares": "Tuna Sirip Kuning",
        "Thunnus atlanticus": "Tuna Atlantik",
        "Thunnus thynnus": "Tuna Sirip Biru",
        "Thymallus arcticus": "Ikan Arctic Grayling",
        "Thymallus thymallus": "Ikan European Grayling",
        "Thyrsites atun": "Ikan Snoek",
        "Tilapia sparrmanii": "Ikan Tilapia",
        "Tinca tinca": "Ikan Tench",
        "Torquigener pleurogramma": "Ikan Puffer",
        "Toxotes jaculatrix": "Ikan Archerfish",
        "Trachinotus baillonii": "Ikan Pompano",
        "Trachinotus blochii": "Ikan Snubnose Pompano",
        "Trachinotus carolinus": "Ikan Palometa",
        "Trachinotus falcatus": "Ikan Permit",
        "Trachinotus goodei": "Ikan Palometa",
        "Trachinotus ovatus": "Ikan Pompano",
        "Trachurus mediterraneus": "Ikan Mediterranean Horse Mackerel",
        "Trachurus novaezelandiae": "Ikan Jack Mackerel",
        "Trachurus symmetricus": "Ikan Pacific Jack Mackerel",
        "Trachurus trachurus": "Ikan Atlantic Horse Mackerel",
        "Triaenodon obesus": "Ikan White-tip Reef Shark",
        "Triakis semifasciata": "Hiu Leopard",
        "Trichiurus lepturus": "Ikan Cutlassfish",
        "Trinectes maculatus": "Ikan Hogchoker",
        "Trygonorrhina fasciata": "Ikan Southern Fiddler Ray",
        "Tylosurus crocodilus": "Ikan Houndfish",
        "Umbra limi": "Ikan Central Mudminnow",
        "Variola louti": "Ikan Yellow-edged Lyretail",
        "Xiphias gladius": "Ikan Pedang",
        "Xyrichtys novacula": "Ikan Pearly Razorfish"
    }

    if scientific_name in manual_translations:
        return manual_translations[scientific_name], 'manual'

    genus = normalized.split()[0].lower() if normalized else ''
    if genus and genus in genus_dict:
        descriptor = ''
        parts = normalized.split()
        if len(parts) > 1:
            descriptor = format_candidate(parts[1])
        genus_name = genus_dict[genus]
        combined = f"{genus_name} {descriptor}".strip()
        return combined, 'database_genus'

    # Default fallback
    return default_fallback_name(scientific_name), 'fallback'

def main():
    # Load scientific labels
    with open('models/classification/labels_ilmiah.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)

    existing_idx_to_name = {}
    existing_path = 'models/classification/labels_indonesian.json'
    ignore_existing = os.environ.get('IGNORE_EXISTING_LABELS') == '1'
    if ignore_existing:
        print('Mengabaikan label Indonesian existing (IGNORE_EXISTING_LABELS=1).')

    if not ignore_existing and os.path.exists(existing_path):
        try:
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_labels = json.load(f)
            existing_idx_to_name = {idx: name for name, idx in existing_labels.items()}
        except Exception:
            existing_idx_to_name = {}

    # Load database-derived names
    species_dict, genus_dict = load_ikan_database('ikan_db.csv')

    translated_labels = {}
    used_names = set()
    source_counts = defaultdict(int)
    not_found = []

    for scientific_name, idx in sorted(labels.items(), key=lambda item: item[1]):
        raw_name, source = get_indonesian_name(
            scientific_name,
            idx,
            species_dict,
            genus_dict,
            existing_idx_to_name,
        )

        formatted = apply_name_replacements(format_candidate(raw_name))
        if not formatted:
            formatted = apply_name_replacements(default_fallback_name(scientific_name))
            source = 'fallback'

        final_name = ensure_unique_name(formatted, scientific_name, used_names)
        translated_labels[final_name] = idx
        source_counts[source] += 1

        if source == 'fallback':
            not_found.append(scientific_name)

    if len(translated_labels) != len(labels):
        missing_indices = sorted(set(labels.values()) - set(translated_labels.values()))
        raise ValueError(
            f"Translation mismatch: expected {len(labels)} entries but produced {len(translated_labels)}. Missing indices: {missing_indices}"
        )

    # Save translated label files (name -> index)
    output_paths = [
        'models/classification/labels_indonesian.json',
        'models/classification/labels.json',
    ]
    for path in output_paths:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(translated_labels, f, ensure_ascii=False, indent=2)

    # Also generate id -> name mapping for convenience
    id_to_name = {str(idx): name for name, idx in translated_labels.items()}
    with open('models/classification/labels_by_id.json', 'w', encoding='utf-8') as f:
        json.dump(id_to_name, f, ensure_ascii=False, indent=2)

    print(f"Translated {len(labels)} labels")
    print("Sumber nama:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print(f"Menggunakan fallback: {len(not_found)}")

    # Save not found list for manual review
    with open('models/classification/not_found_species.txt', 'w', encoding='utf-8') as f:
        for species in not_found:
            f.write(f"{species}\n")

if __name__ == "__main__":
    main()
