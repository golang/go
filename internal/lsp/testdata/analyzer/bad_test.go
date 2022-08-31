package analyzer

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func Testbad(t *testing.T) { //@diag("", "tests", "Testbad has malformed name: first letter after 'Test' must not be lowercase", "warning")
	var x sync.Mutex
	_ = x //@diag("x", "copylocks", "assignment copies lock value to _: sync.Mutex", "warning")

	printfWrapper("%s") //@diag(re`printfWrapper\(.*\)`, "printf", "golang.org/lsptests/analyzer.printfWrapper format %s reads arg #1, but call has 0 args", "warning")
}

func printfWrapper(format string, args ...interface{}) {
	fmt.Printf(format, args...)
}

func _() {
	now := time.Now()
	fmt.Println(now.Format("2006-02-01")) //@diag("2006-02-01", "timeformat", "2006-02-01 should be 2006-01-02", "warning")
}
