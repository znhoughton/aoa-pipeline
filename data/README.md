# Data

Six input files, all read through `find_file()` in Step 2 of the notebook. None of
them are produced by this repo; each is a redistribution of a published resource or a
derived table built from one. Sources and terms below.

| File | Rows | What it is |
|---|---|---|
| `cdi_items_with_aoa.csv` | 679 | Wordbank CDI (Words & Sentences) items with age-of-acquisition in months. The evaluation target. |
| `contexts_all_childes.csv` | 503,611 | CHILDES utterances, one row per (word, sentence), covering 10,837 distinct words. Used to build contextual representations. |
| `childes_freq_eng-na.csv` | 58,210 | CHILDES English–North America word frequencies (child-produced and child-directed counts, per-million, log). Supplies `childprod_logfreq`. |
| `childes_mlu.csv` | 45,598 | Mean length of utterance per word in CHILDES. Supplies `mean_sent_length`. |
| `brysbaert_concreteness.txt` | 39,954 | Brysbaert, Warriner & Kuperman (2014) concreteness norms. Tab-separated. Supplies `Conc.M`. |
| `perry_2015_babiness.csv` | 22,681 | Perry, Perlman & Lupyan (2015) ratings. Supplies `babyAVG`. |

The last four columns together form the **Braginsky-4** predictor set (frequency,
concreteness, babiness, MLU) that the model read-outs are benchmarked against.

## Provenance and terms

- **Wordbank / CDI**. Frank, Braginsky, Yurovsky & Marchman (2017), *Wordbank: an open
  repository for developmental vocabulary data*, JCL 44(3). Wordbank data is openly
  available at <http://wordbank.stanford.edu>.
- **CHILDES**. MacWhinney (2000), *The CHILDES Project*. TalkBank is released under
  CC BY-NC-SA 3.0. The frequency, MLU and context tables here are derived counts and
  extracted utterances, not the full corpora; they carry the same non-commercial terms.
- **Concreteness norms**. Brysbaert, Warriner & Kuperman (2014), *Concreteness ratings
  for 40 thousand generally known English word lemmas*, Behavior Research Methods 46.
  Distributed by the authors for research use.
- **Babiness / iconicity ratings**. Perry, Perlman & Lupyan (2015), *Iconicity in
  English and Spanish and its relation to lexical category and age of acquisition*,
  PLoS ONE 10(9). CC BY.

These files keep the terms of their original sources, which are not affected by their
being mirrored here. In particular, the CC BY-NC-SA on the CHILDES derivatives means
this data should not be used commercially. Cite the original papers, not this repo,
when you use these norms.

## Not included

`subtlex_us.tsv` was present in the original working folder but is never referenced by
the pipeline, so it was dropped during cleanup. If a later analysis needs SUBTLEX-US
frequencies, get them from <https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus>
rather than restoring the stale copy.
