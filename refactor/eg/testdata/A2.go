// +build ignore

package A2

// This refactoring causes addition of "errors" import.
// TODO(adonovan): fix: it should also remove "fmt".

import myfmt "fmt"

func example(n int) {
	myfmt.Errorf("%s", "")
}
