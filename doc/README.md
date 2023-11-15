# Release Notes

The `initial` and `next` subdirectories of this directory are for release notes.

At the start of a release development cycle, the contents of `next` should be deleted
and replaced with those of `initial`.
Release notes should be added to `next` by editing existing files or creating new files.

To prepare the release notes for a release, run `golang.org/x/build/cmd/relnote generate`.
That will merge the `.md` files in `next` into a single file.

The files are merged by being concatenated in sorted order by pathname. Files in
the directory matching the glob "*stdlib/*minor" are treated specially. They should
be in subdirectories corresponding to standard library package paths, and headings
for those package paths will be generated automatically.

Files in this repo's `api/next` directory must have corresponding files in `*stdlib/*minor`.
The files should be in the subdirectory for the package with the new API, and should
be named after the issue number of the API proposal. For example, for an api/next
file with the line

    pkg net/http, function F #12345

there should be a file named net/http/12345.md.
At a minimum, that file should contain either a full sentence or a TODO,
ideally referring to a person with the responsibility to complete the note.
