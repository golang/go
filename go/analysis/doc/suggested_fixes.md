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
	SuggestedFixes []SuggestedFix // this is an optional field
}

```

## Alternatives

# Performing transformations directly on the AST

TODO(matloob): expand on this.

Even though it may be more convienient
for authors of refactorings to perform transformations directly on
the AST, allowing mutations on the AST would mean that a copy of the AST
would need to be made every time a transformation was produced, to avoid
having transformations interfere with each other.
