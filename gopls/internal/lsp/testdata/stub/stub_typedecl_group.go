package stub

// Regression test for Issue #56825: file corrupted by insertion of
// methods after TypeSpec in a parenthesized TypeDecl.

import "io"

func newReadCloser() io.ReadCloser {
	return rdcloser{} //@suggestedfix("rd", "quickfix", "")
}

type (
	A        int
	rdcloser struct{}
	B        int
)

func _() {
	// Local types can't be stubbed as there's nowhere to put the methods.
	// The suggestedfix assertion can't express this yet. TODO(adonovan): support it.
	type local struct{}
	var _ io.ReadCloser = local{} // want error: `local type "local" cannot be stubbed`
}

type (
	C int
)
