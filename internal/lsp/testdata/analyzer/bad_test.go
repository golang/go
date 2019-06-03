package analyzer

import (
	"fmt"
	"sync"
	"testing"
)

func Testbad(t *testing.T) { //@diag("", "tests", "Testbad has malformed name: first letter after 'Test' must not be lowercase")
	var x sync.Mutex
	_ = x //@diag("x", "copylocks", "assignment copies lock value to _: sync.Mutex")

	printfWrapper("%s") //@diag(re`printfWrapper\(.*\)`, "printf", "printfWrapper format %s reads arg #1, but call has 0 args")
}

func printfWrapper(format string, args ...interface{}) {
	fmt.Printf(format, args...)
}
