package cgotest

/*
typedef struct {} git_merge_file_input;

typedef struct {} git_merge_file_options;

int git_merge_file(
        git_merge_file_input *in,
        git_merge_file_options *opts) {}
*/
import "C"
import (
	"fmt"
	"testing"
)

func test9026(t *testing.T) {
	var in C.git_merge_file_input
	var opts *C.git_merge_file_options
	C.git_merge_file(&in, opts)

	// Test that the generated type names are deterministic.
	// (Previously this would fail about 10% of the time.)
	//
	// Brittle: the assertion may fail spuriously when the algorithm
	// changes, but should remain stable otherwise.
	got := fmt.Sprintf("%T %T", in, opts)
	want := "cgotest._Ctype_struct___12 *cgotest._Ctype_struct___13"
	if got != want {
		t.Errorf("Non-deterministic type names: got %s, want %s", got, want)
	}
}
