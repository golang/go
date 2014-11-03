package issue9026

// This file appears in its own package since the assertion tests the
// per-package counter used to create fresh identifiers.

/*
typedef struct {} git_merge_file_input;

typedef struct {} git_merge_file_options;

void git_merge_file(
        git_merge_file_input *in,
        git_merge_file_options *opts) {}
*/
import "C"
import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	var in C.git_merge_file_input
	var opts *C.git_merge_file_options
	C.git_merge_file(&in, opts)

	// Test that the generated type names are deterministic.
	// (Previously this would fail about 10% of the time.)
	//
	// Brittle: the assertion may fail spuriously when the algorithm
	// changes, but should remain stable otherwise.
	got := fmt.Sprintf("%T %T", in, opts)
	want := "issue9026._Ctype_struct___0 *issue9026._Ctype_struct___1"
	if got != want {
		t.Errorf("Non-deterministic type names: got %s, want %s", got, want)
	}
}
