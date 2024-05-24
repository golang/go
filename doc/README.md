# Release Notes

The `initial` and `next` subdirectories of this directory are for release notes.

## For developers

Release notes should be added to `next` by editing existing files or creating
new files. **Do not add RELNOTE=yes comments in CLs.** Instead, add a file to
the CL (or ask the author to do so).

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

If your CL addresses an accepted proposal, mention the proposal issue number in
your release note in the form `/issue/NUMBER`. A link to the issue in the text
will have this form (see below). If you don't want to mention the issue in the
text, add it as a comment:
```
<!-- go.dev/issue/12345 -->
```
If an accepted proposal is mentioned in a CL but not in the release notes, it will be
flagged as a TODO by the automated tooling. That is true even for proposals that add API.

Use the following forms in your markdown:

	[http.Request]                     # symbol documentation; auto-linked as in Go doc strings
	[Request]                          # short form, for symbols in the package being documented
	[net/http]                         # package link
	[#12345](/issue/12345)             # GitHub issues
	[CL 6789](/cl/6789)                # Gerrit changelists

To preview `next` content in merged form using a local instance of the website, run:

```
go run golang.org/x/website/cmd/golangorg@latest -content='' -goroot=..
```

Then open http://localhost:6060/doc/next. Refresh the page to see your latest edits.

## For the release team

The `relnote` tool, at `golang.org/x/build/cmd/relnote`, operates on the files
in `doc/next`.

As a release cycle nears completion, run `relnote todo` to get a list of
unfinished release note work.

To prepare the release notes for a release, run `relnote generate`.
That will merge the `.md` files in `next` into a single file.
Atomically (as close to it as possible) add that file to `_content/doc` directory
of the website repository and remove the `doc/next` directory in this repository.

To begin the next release development cycle, populate the contents of `next`
with those of `initial`. From the repo root:

    > cd doc
    > cp -r initial/* next

Then edit `next/1-intro.md` to refer to the next version.
