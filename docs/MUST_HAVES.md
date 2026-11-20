# Must-have fields per page

Every document built from a page is checked against this list (`ingest/spec.py`). A missing required field becomes a
quality flag (`<field>_missing`) on the document and in `data/reports/ingest_report.json`; a count below the usual target
becomes `<field>_below_target`.

## Project page (Real Estate developments, AIDA, Trump Mansions, ...)

| Field | What is captured | Usual size |
|---|---|---|
| Heading / title | Project name and tagline | 1 |
| Country, city | From the location label, JSON-LD address and menu path | 1 each |
| Descriptive text | The full "about" text (and hero text) | several paragraphs |
| Project details | Property type, status, unit type, area (sqm); min/max parsed; conflicts with the text flagged | 4 |
| Photo gallery | Every gallery image: URL, alt text, size; plus hero and property-card images | 10 to 15 |
| Key features | Title, distance/duration (for example "The Beach, 5 min"), image | 4 to 6 |
| Amenities | Every amenity name | 10 to 15 |
| Why invest | The intro text and each bullet (name plus value). Marked as developer claims | 5 to 15 bullets + 1 text |
| Location | Description, coordinates, nearby highlights | 2 to 3 points |
| FAQ | Every question and its full answer, one chunk each | 8 to 24 |
| Also stored | Starting price (null if not published), currency, partner and role, progress photos, brochure and booking links, page dates, source URL, menu path, snapshot file | |

Not scraped: location highlights "from Google and the internet" (for example distance to Dubai Marina). If added later,
they go in a separate `external_highlights` field with the source named, never mixed with developer text.

## Partner page: name, description, related projects (linked to project documents), FAQ, related articles.
## Article / press page: title, date, full body in chunks of about 250 words, description.
## Other pages (home, hospitality, insignia, one-of-one, country listings): title, text sections, the projects they list.
