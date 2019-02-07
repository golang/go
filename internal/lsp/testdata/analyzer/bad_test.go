package analyzer

import (
	"sync"
	"testing"
)

func Testbad(t *testing.T) { //@diag("", "tests", "Testbad has malformed name: first letter after 'Test' must not be lowercase")
	var x sync.Mutex
	_ = x //@diag("x", "copylocks", "assignment copies lock value to _: sync.Mutex")
}
