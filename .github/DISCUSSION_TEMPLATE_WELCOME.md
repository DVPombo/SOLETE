<!--
This file is NOT a real GitHub Discussion — GitHub Discussions can only be turned on
via a repo setting (Settings → Features → Discussions), which this file can't do.
This is saved content for the maintainer to copy-paste into a real pinned Discussions
post once Discussions is enabled. See the Phase 4 execution prompt's manual checklist.
-->

# Welcome to SOLETE Discussions

SOLETE is a 15-month, holistic dataset of co-located meteorology, wind power, and
solar PV power from a test site (SYSLAB) at the Technical University of Denmark, along
with the code used to load, explore, and forecast with it. It started as complementary
material for a "Data in Brief" paper and a series of solar-forecasting papers, and has
since grown into a general platform for experimenting with time-series forecasting.

If you're new here, a few starting points:

- **`DATA_DICTIONARY.md`** — every column in the dataset, what it means, and where it
  comes from.
- **`KNOWN_ISSUES.md`** — data-quality issues that are already known (some fixed, some
  open) — worth checking before you report something new, or before you build an
  analysis that assumes a column is clean.
- **`QC_SCHEMA.md`** — the design behind the `<column>_qc` flag layer, if you want to
  filter or weight rows by data quality programmatically.
- **`examples/`** — four notebooks that walk through loading the data, the QC flags,
  and simple PV/wind forecasting demos on a small sample file, each with an "Open in
  Colab" badge so you can run them without installing anything locally.

## Ask here instead of by email or LinkedIn

If you have a usage question, a data question, or an idea you want to talk through, please
post it here rather than emailing or messaging the maintainer directly. Plainly: individual
emails and LinkedIn messages don't scale, and a question asked here is answered once and
stays searchable for the next person with the same question, instead of being answered
repeatedly in private. Bug reports and data-quality findings still belong in the issue
tracker (see `CONTRIBUTING.md` and the issue templates) — Discussions is for open-ended
questions and ideas, not tracked action items.

Looking forward to seeing what people build with this.
