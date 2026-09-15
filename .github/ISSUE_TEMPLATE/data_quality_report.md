---
name: Data quality report
about: Report a data-quality issue in the SOLETE dataset itself (not a code bug)
title: "[DATA] "
labels: data-quality
assignees: ''
---

<!--
This is for issues with the dataset's *content* — a suspicious value, an
out-of-range reading, an unpopulated column, something that doesn't match the
paper, etc. — not a bug in the loading/processing code. If it's a code bug,
please use the "Bug report" template instead.
-->

## Which file, column, and resolution?

<!-- e.g. SOLETE_Pombo_60min.h5, HUMIDITY[%], 60-minute resolution -->

## What did you find?

<!-- Describe the issue: what values, how many rows/what proportion, any pattern to when it occurs. -->

## How did you find it?

<!-- What you were doing when you noticed it — a plot, a summary stat, a specific analysis. -->

## Have you checked KNOWN_ISSUES.md and QC_SCHEMA.md already?

- [ ] Yes, and this isn't already documented there
- [ ] Yes, and this is related to an existing entry, but I think it needs updating (explain below)
- [ ] No, I haven't checked yet

## Anything else?

<!-- Any independent verification you did (e.g. comparison against another data source), or a hunch about root cause. -->
