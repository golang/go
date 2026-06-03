# Making Unified Intermediate Representation (UIR) Changes

For general information on export data, see [here](../../README.md).

UIR is the serial form of the compiler's intermediate representation, used to
propagate bodies of generic and/or inlined functions from one compilation unit
to another.

The Go compiler has a single, canonical UIR writer implementation in
`src/cmd/compile/internal/noder/writer.go`. When we update the byte stream that
the UIR writer writes, *all* of the UIR readers need to be reviewed and
potentially updated; a change might not be backward compatible for them. These
instructions outline the steps required to keep all UIR readers up-to-date.

## The Writer

The UIR version written by the compiler is controlled by
`src/cmd/compile/internal/noder/unified.go`. Do not change this yet. Instead:

1. Add a version flag N+1 for `MyChange` in `internal/pkgbits/version.go`.
2. Update the UIR writer in `src/cmd/compile/internal/noder/writer.go` to guard
   the writing of any fields added in N+1. Note: readers still on version N
   *must* be oblivious to this change to avoid breaking the readers on
   submission.

## The Readers

Besides the compiler itself, there are other readers in go, x/tools, and
externally. Those in x/tools and the general public exist because
`go list -export` produces export data files in this format and we support the
ability of applications to decode it.

> Note that there is an upcoming plan to decouple the compiler's IR from x/tools
> by changing the format encoded by `go list -export`; this would make UIR a
> private detail of the compiler, free to break at any time. For now, these
> instructions must still be followed.

We assume that external readers will update on their own. The necessary reader
updates in go and x/tools are detailed below.

### go

3. Update the compiler's own UIR reader in
   `src/cmd/compile/internal/noder/reader.go` to guard the reading of any fields
   added in N+1.
4. Repeat this change for the readers in `src/go/internal/gcimporter/ureader.go`
   and `src/cmd/compile/internal/importer/ureader.go`. Note that these readers
   only read data needed for type checking (in `src/go/types` and
   `src/cmd/compile/internal/types2` respectively). For instance, they do not
   read exported function bodies. Thus, it's possible that a change to UIR (such
   as the encoding of function bodies) would require no change to these readers.

### x/tools

5. Add a version flag for `MyChange` in `internal/pkgbits/version.go`. Note:
   x/tools has its own pkgbits implementation, which is intended to be an exact
   copy of the [one in go](#the-writer). Any change made to one must be
   reflected in the other.
6. Update the x/tools UIR reader in `internal/gcimporter/ureader.go` to guard
   the reading of any fields added in N+1. Call this commit C.
7. In go, take the commit hash for C and update `src/cmd/go.mod` to use x/tools@C
   per the [vendoring instructions](https://go.dev/wiki/MinorReleases#cherry-pick-cls-for-vendored-golangorgx-packages).

## Finalizing

> If this UIR change will be tested, check the [following section](#testing) and
> consider when it makes to finalize.

Only after reviewing *all* of the readers, bump the UIR version written by the
writer to N+1 in `src/cmd/compile/internal/noder/unified.go`. Because the
readers have already been updated to handle version N+1, this change is
compatible.

## Testing

If making changes related to some new feature requiring extensive testing, it's
best to postpone bumping the UIR version until *all* of the tests are in. To
commit tests incrementally, develop them with a locally-incremented UIR version
and commit *skipped* tests; don't yet bump the remote UIR version.

Once all of the required tests are in, bump the remote UIR version while turning
on all of the previously skipped tests. This minimizes churn on the UIR version
as testing uncovers any discrepancies.
