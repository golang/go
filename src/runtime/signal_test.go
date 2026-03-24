// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
)

func TestSignalBogus(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("No syscall.Signal on plan9")
	}

	// This test more properly belongs in os/signal, but it depends on
	// containing the only call to signal.Notify in the process, so it must
	// run as an isolated subprocess, which is simplest with testprog.
	t.Parallel()
	output := runTestProg(t, "testprog", "SignalBogus")
	want := "OK\n"
	if output != want {
		t.Fatalf("output is not %q\n%s", want, output)
	}
}
