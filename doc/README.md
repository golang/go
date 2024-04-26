# Release Notes

The `initial` and `next` subdirectories of this directory are for release notes.

## For developers

Release notes should be added to `next` by editing existing files or creating new files.

At the end of the development cycle, the files will be merged by being
concatenated in sorted order by pathname. Files in the directory matching the
glob "*stdlib/*minor" are treated specially. They should be in subdirectories
corresponding to standard library package paths, and headings for those package
paths will be generated automatically.

Files in this repo's `api/next` directory must have corresponding files in
`doc/next/*stdlib/*minor`.
The files should be in the subdirectory for the package with the new
API, and should be named after the issue number of the API proposal.
For example, if the directory `6-stdlib/99-minor` is present,
then an `api/next` file with the line

    pkg net/http, function F #12345

should have a corresponding file named `doc/next/6-stdlib/99-minor/net/http/12345.md`.
At a minimum, that file should contain either a full sentence or a TODO,
ideally referring to a person with the responsibility to complete the note.

Use the following forms in your markdown:

	[http.Request]                     # symbol documentation; auto-linked as in Go doc strings
	[#12345](/issue/12345)             # GitHub issues
	[CL 6789](/cl/6789)                # Gerrit changelists

## For the release team

At the start of a release development cycle, the contents of `next` should be deleted
and replaced with those of `initial`. From the repo root:

    > cd doc
    > rm -r next/*
    > cp -r initial/* next

Then edit `next/1-intro.md` to refer to the next version.

To prepare the release notes for a release, run `golang.org/x/build/cmd/relnote generate`.
That will merge the `.md` files in `next` into a single file.
