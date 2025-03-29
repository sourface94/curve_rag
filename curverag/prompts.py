entity_relationship_extraction__disparate_prompt = """# DOMAIN
# GOAL
Your goal is to highlight information that is relevant to the domain and the queries that may be asked on it. Given an input document, identify all relevant entities and all relationships among them.

Examples of possible queries:
{example_queries}

# INSTRUCTIONS
1. **ENTITY IDENTIFICATION**: Identify and meticulously extract all entities mentioned in the document that belong to the provided ENTITY TYPES. For each entity, provide a concise description capturing its key features within the document's context. Use singular entity names and split compound concepts when necessary for clarity.
2. **RELATIONSHIP DISCOVERY**: Identify and describe ALL relationships between the extracted entities. Resolve pronouns to entity names for clarity. Ensure relationship descriptions clearly explain the connection between entities.
3. **ENTITY COVERAGE CHECK**: Verify that every identified entity is part of at least one relationship. If any entity is isolated, infer and add a relationship to connect it to the graph, even if the relationship is implicit.
4. **OUTPUT FORMAT: STRICTLY VALID JSON**: Output MUST be in strictly valid JSON format.  Adhere to ALL standard JSON rules. The JSON MUST contain three top-level lists: "entities", "relationships", and "other_relationships". Each list item must be a JSON object with the REQUIRED fields ("name", "type", "desc" for entities; "source", "target", "desc" for relationships), all as JSON strings enclosed in DOUBLE QUOTES ONLY.  **ABSOLUTELY NO SINGLE QUOTES, BRACKETS FOR STRINGS, TRAILING COMMAS, OR MARKDOWN FORMATTING (like triple backticks) ARE PERMITTED.**  Invalid JSON output is unacceptable.

# EXAMPLE INPUT DATA
Allowed Entity Types: [location, organization, person, communication]
Document: "Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into new media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."

# EXAMPLE OUTPUT DATA (VALID JSON - DO NOT DEVIATE)
{{
  "entities": [
    {{"name": "RADIO CITY", "type": "organization", "desc": "India's first private FM radio station"}},
    {{"name": "INDIA", "type": "location", "desc": "A country in South Asia"}},
    {{"name": "FM RADIO STATION", "type": "communication", "desc": "A radio broadcasting service using frequency modulation"}},
    {{"name": "ENGLISH", "type": "communication", "desc": "A language of global communication"}},
    {{"name": "HINDI", "type": "communication", "desc": "An Indo-Aryan language of India"}},
    {{"name": "NEW MEDIA", "type": "communication", "desc": "Digital forms of media content"}},
    {{"name": "PLANETRADIOCITY", "type": "organization", "desc": "An online platform for music and entertainment"}},
    {{"name": "MUSIC PORTAL", "type": "communication", "desc": "A website dedicated to music-related content"}},
    {{"name": "NEWS", "type": "communication", "desc": "Reports on current events, especially in music"}},
    {{"name": "VIDEO", "type": "communication", "desc": "Moving visual content, often music-related"}},
    {{"name": "SONG", "type": "communication", "desc": "A musical composition with lyrics"}}
  ],
  "relationships": [
    {{"source": "RADIO CITY", "target": "INDIA", "desc": "Radio City is geographically situated in India"}},
    {{"source": "RADIO CITY", "target": "FM RADIO STATION", "desc": "Radio City operates as a private FM radio station, launched on July 3, 2001"}},
    {{"source": "RADIO CITY", "target": "ENGLISH", "desc": "Radio City's broadcasts include songs in the English language"}},
    {{"source": "RADIO CITY", "target": "HINDI", "desc": "Radio City's broadcasts also feature songs in the Hindi language"}},
    {{"source": "RADIO CITY", "target": "PLANETRADIOCITY", "desc": "Radio City expanded into new media with PlanetRadiocity.com in May 2008"}},
    {{"source": "PLANETRADIOCITY", "target": "MUSIC PORTAL", "desc": "PlanetRadiocity.com functions as a music portal"}},
    {{"source": "PLANETRADIOCITY", "target": "NEWS", "desc": "PlanetRadiocity.com provides music-related news content"}},
    {{"source": "PLANETRADIOCITY", "target": "SONG", "desc": "PlanetRadiocity.com makes songs available to users"}}
  ],
  "other_relationships": [
    {{"source": "RADIO CITY", "target": "NEW MEDIA", "desc": "Radio City's foray into new media occurred in May 2008"}},
    {{"source": "PLANETRADIOCITY", "target": "VIDEO", "desc": "PlanetRadiocity.com includes music-related video content"}}
  ]
}}
"""