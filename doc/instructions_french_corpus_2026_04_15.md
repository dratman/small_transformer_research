# Instructions: Add French Literature to Training Corpus

**Date:** 2026-04-15
**From:** Claude Code instance on Mac Studio
**To:** Claude Code instance on M3 MacBook Pro

## Background

The cleaned corpus (corpus_cleaned_2026_04_15.txt, 2.5 GB) contains about
1.2% incidental French text from Gutenberg. During early training of the
new BPE model on the Mac Studio, we noticed that the model produces
surprisingly coherent French output even at only 5,500 iterations — more
coherent than its English output at the same stage.

Ralph's hypothesis: French literary prose is more tightly structured and
stylistically uniform than English (institutional standardization since
1635, strong conventions of clarity and logical structure). This makes it
a more compressible signal that the model can learn faster. The bilingual
training may also help the middle layers develop more robust abstract
representations, since French provides additional examples of narrative
structure, agent/patient roles, and temporal sequencing with different
surface forms.

We want to deliberately include French literature in the next corpus build.

## What to do

### 1. Download French literary texts from Project Gutenberg

Use the Gutenberg catalog to find texts where the language field is French.
The catalog is available at:
- https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv (full catalog CSV)
- The language column in this CSV identifies text language

Gutenberg has roughly 3,000+ French-language texts. Filter for literary
works (novels, stories, plays, essays, philosophy). Exclude:
- Dictionaries and word lists
- Technical/scientific texts
- Translations of English works into French (we want original French)
- Very short texts (under 10 KB)

Target authors (non-exhaustive, prioritize these):
- Balzac, Hugo, Flaubert, Stendhal, Zola, Dumas (pere et fils)
- Maupassant, Sand, Voltaire, Rousseau, Montesquieu
- Moliere, Racine, Corneille (drama)
- Proust, Anatole France, Daudet
- Chateaubriand, Lamartine, Musset
- La Fontaine, Montaigne

### 2. Clean the French texts

Apply the same cleaning pipeline used for the English corpus:
- Strip Gutenberg headers and footers
- Remove encoding artifacts
- Lowercase
- Clean noise (repeated characters, formatting artifacts)

### 3. Combine with English corpus

- Target ratio: approximately 10-20% French, 80-90% English
- The current English corpus is about 2.5 GB, so aim for 250-500 MB
  of French text
- Split into paragraphs and shuffle all paragraphs together (English
  and French mixed, same as the current paragraph-shuffled approach)
- Use the same random seed (42) for reproducibility

### 4. Deliver the corpus

Copy the finished corpus to the Mac Studio via iCloud, same as before:
  ~/Library/Mobile Documents/com~apple~CloudDocs/0-HomeFolder-Working-iCloud_A/

Include a summary of what was included (number of texts, total size,
author breakdown) either in the iCloud folder or as a new diary entry.

## Notes

- The current model is training fine on the 2.5 GB corpus. This French
  addition is for the NEXT corpus rebuild, not urgent.
- The BPE tokenizer (vocab 8192) will be retrained on the combined
  corpus, so it will naturally allocate tokens to French patterns.
- The 1.2% French that's already in the current corpus came from
  texts that weren't filtered by language. The new approach should
  be deliberate and controlled.
