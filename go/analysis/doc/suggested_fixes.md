# Suggested Fixes in the Analysis Framework

## The Purpose of Suggested Fixes

The analysis framework is planned to add a facility to output
suggested fixes. Suggested fixes in the analysis framework
are meant to address two common use cases. The first is the
natural use case of allowing the user to quickly fix errors or issues
pointed out by analyzers through their editor or analysis tool.
An editor, when showing a diagnostic for an issue, can propose
code to fix that issue. Users can accept the proposal and have
the editor apply the fix for them. The second case is to allow
for defining refactorings. An analyzer meant to perform a
refactoring can produce suggested fixes equivalent to the diff
of the refactoring. Then, an analysis driver meant to apply
refactorings can automatically apply all the diffs that
are produced by the analysis as suggested fixes.

## Proposed Suggested Fix API

Suggested fixes will be defined using the following structs:

```go
// A SuggestedFix is a code change associated with a Diagnostic that a user can choose
// to apply to their code. Usually the SuggestedFix is meant to fix the issue flagged
// by the diagnostic.
type SuggestedFix struct {
	// A description for this suggested fix to be shown to a user deciding
	// whether to accept it.
	Message   string
	TextEdits []TextEdit
}

// A TextEdit represents the replacement of the code between Pos and End with the new text.
type TextEdit struct {
	// For a pure insertion, End can either be set to Pos or token.NoPos.
	Pos     token.Pos
	End     token.Pos
	NewText []byte
}
```

A suggested fix needs a message field so it can specify what it will do.
Some analyses may not have clear cut fixes, and a suggested fix may need
to provide additional information to help users specify whether they
should be added.

Suggested fixes are allowed to make multiple
edits in a file, because some logical changes may affect otherwise
unrelated parts of the AST.

A TextEdit specifies a Pos and End: these will usually be the Pos
and End of an AST node that will be replaced.

Finally, the replacements themselves are represented as []bytes.


Suggested fixes themselves will be added as a field in the
Diagnostic struct:

```go

type Diagnostic struct {
	...
	SuggestedFixes []SuggestedFix  // this is an optional field
}

```

### Requirements for SuggestedFixes

SuggestedFixes will be required to conform to several requirements:

* TextEdits for a SuggestedFix should not overlap.
* TextEdits for SuggestedFixes should not contain edits for other packages.
* Each TextEdit should apply to a single file.

These requirements guarantee that suggested fixes can be cleanly applied.
Because a driver may only analyze, or be able to modify, the current package,
we restrict edits to the current package. In general this restriction should
not be a big problem for users because other packages might not belong to the
same module and so will not be safe to modify in a singe change.

On the other hand, analyzers will not be required to produce gofmt-compliant
code. Analysis drivers will be expected to apply gofmt to the results of
a SuggestedFix application.

## SuggestedFix integration points

### ```checker -fix```

Singlechecker and multichecker have the ```-fix``` flag, which will automatically
apply all fixes suggested by their analysis or analyses. This is intended to
be used primarily by refactoring tools, because in general, like diagnostics,
suggested fixes will need to be examined by a human who can decide whether
they are relevent.

### gopls

Suggested fixes have been integrated into ```gopls```, and editors can choose
to display the suggested fixes to the user as they type, so that they can be
accepted to fix diagnostics immediately.

### Code Review Tools (Future Work)

Suggested fixes can be integrated into programs that are integrated with
code review systems to suggest fixes that users can apply from their code review tools.

## Alternatives

### Performing transformations directly on the AST

Even though it may be more convenient
for authors of refactorings to perform transformations directly on
the AST, allowing mutations on the AST would mean that a copy of the AST
would need to be made every time a transformation was produced, to avoid
having transformations interfere with each other.

This is primarily an issue with the current design of the Go AST and
it's possible that a new future version of the AST might make this a more
viable option.

### Supplying AST nodes directly

Another possibility would be for SuggestedFixes to supply the replacement
ASTs directly. There is one primary limitation to this: that because
comments to ASTs specify their location using token.Pos values, it's very
difficult to place any comments in the right place.

In general, it's also more difficult to generate the AST structures for
some code than to generate the text for that code. So we prefer to allow
the flexibility to do the latter.

Because users can call ```format.Node``` to produce the text for any
AST node, users will always be able to produce a SuggestedFix from AST
nodes. In future, we may choose to add a convenience method that does this for users.
