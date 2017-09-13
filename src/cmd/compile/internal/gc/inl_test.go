// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"internal/testenv"
	"os/exec"
	"testing"
)

// TestIntendedInlining tests that specific runtime functions are inlined.
// This allows refactoring for code clarity and re-use without fear that
// changes to the compiler will cause silent performance regressions.
func TestIntendedInlining(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}
	testenv.MustHaveGoRun(t)
	t.Parallel()

	// want is the list of function names that should be inlined.
	want := []string{"tophash", "add", "(*bmap).keys", "bucketShift", "bucketMask"}

	m := make(map[string]bool, len(want))
	for _, s := range want {
		m[s] = true
	}

	cmd := testenv.CleanCmdEnv(exec.Command(testenv.GoToolPath(t), "build", "-a", "-gcflags=-m", "runtime"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
	lines := bytes.Split(out, []byte{'\n'})
	for _, x := range lines {
		f := bytes.Split(x, []byte(": can inline "))
		if len(f) < 2 {
			continue
		}
		fn := bytes.TrimSpace(f[1])
		delete(m, string(fn))
	}

	for s := range m {
		t.Errorf("function %s not inlined", s)
	}
}
