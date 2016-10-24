package gccgoimporter

// This file contains testing utilities copied from $GOROOT/src/internal/testenv/testenv.go.

import (
	"runtime"
	"strings"
	"testing"
)

// HasGoBuild reports whether the current system can build programs with ``go build''
// and then run them with os.StartProcess or exec.Command.
func HasGoBuild() bool {
	switch runtime.GOOS {
	case "android", "nacl":
		return false
	case "darwin":
		if strings.HasPrefix(runtime.GOARCH, "arm") {
			return false
		}
	}
	return true
}

// MustHaveGoBuild checks that the current system can build programs with ``go build''
// and then run them with os.StartProcess or exec.Command.
// If not, MustHaveGoBuild calls t.Skip with an explanation.
func MustHaveGoBuild(t *testing.T) {
	if !HasGoBuild() {
		t.Skipf("skipping test: 'go build' not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

var testenv = struct {
	HasGoBuild      func() bool
	MustHaveGoBuild func(*testing.T)
}{
	HasGoBuild:      HasGoBuild,
	MustHaveGoBuild: MustHaveGoBuild,
}
