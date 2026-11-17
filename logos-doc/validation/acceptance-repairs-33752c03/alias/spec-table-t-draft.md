# Proposed specification wording

Draft for root review, M0-A-S20 / #442 / AUD-210. This file does not change
the normative specification or the task ledger.

## SPEC 0.1

After the existing sentence requiring every active package to be counted,
including unselected and unimported packages, add:

> A duplicate-stem error in an unselected and unimported package MUST NOT
> abort the census; all of that package's active source-file kinds MUST
> still count. [T-0.1][R-U3]
>
> A source-header failure or mixed-package error MUST still fail the
> census. [T-0.1][R-U3]

Keep the existing tag, overlay, ownership, vendor, nested-module,
mixed-module, and structured-error requirements unchanged.

## SPEC 1.1

Replace the current broad sentence beginning “Two files in one package”
with:

> Building or importing a package with two active files whose names differ
> only by alias extension MUST fail [T-101.1].

Keep the next sentence requiring
`correctness::duplicate_stem_filename` unchanged.

## Table T, T-0.1

After the existing module-census ownership text, add:

> A duplicate-stem error in an unselected and unimported package does not
> abort the census; all its active source-file kinds still count.
> Source-header failures and mixed-package errors still reject the census.

## Table T, T-101.1

Replace only the current duplicate-stem clause with:

> Building or importing a package with two active files whose names differ
> only by alias extension: error `correctness::duplicate_stem_filename`.
> Unselected and unimported siblings still contribute their kinds to the
> module census.

The source change preserves the existing package-local rejection. The new
wording makes its interaction with the already-required module-wide census
explicit. It does not permit a mixed module or turn an unclassifiable source
into a successful census.
