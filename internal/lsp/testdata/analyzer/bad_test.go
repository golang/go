package analyzer

import (
	"fmt"
	"sync"
	"testing"
)

func Testbad(t *testing.T) { //@diag("", "tests", "Testbad has malformed name: first letter after 'Test' must not be lowercase", "warning")
	var x sync.Mutex
	_ = x //@diag("x", "copylocks", "assignment copies lock value to _: sync.Mutex", "warning")

	printfWrapper("%s") //@diag(re`printfWrapper\(.*\)`, "printf", "golang.org/x/tools/internal/lsp/analyzer.printfWrapper format %s reads arg #1, but call has 0 args", "warning")
}

func printfWrapper(format string, args ...interface{}) {
	fmt.Printf(format, args...)
}
