// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains testing utilities copied from $GOROOT/src/internal/testenv/testenv.go.

package gccgoimporter

import (
	"runtime"
	"testing"

	toolstestenv "golang.org/x/tools/internal/testenv"
)

// HasExec reports whether the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
func HasExec() bool {
	return toolstestenv.HasExec()
}

// MustHaveExec checks that the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
// If not, MustHaveExec calls t.Skip with an explanation.
func MustHaveExec(t *testing.T) {
	if !HasExec() {
		t.Skipf("skipping test: cannot exec subprocess on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

var testenv = struct {
	MustHaveExec func(*testing.T)
}{
	MustHaveExec: MustHaveExec,
}
