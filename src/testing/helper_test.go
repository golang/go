// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"internal/testenv"
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestTBHelper(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		testTestHelper(t)

		// Check that calling Helper from inside a top-level test function
		// has no effect.
		t.Helper()
		t.Error("8")
		return
	}

	testenv.MustHaveExec(t)
	t.Parallel()

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, exe, "-test.run=^TestTBHelper$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, _ := cmd.CombinedOutput()

	want := `--- FAIL: TestTBHelper \([^)]+\)
    helperfuncs_test.go:15: 0
    helperfuncs_test.go:47: 1
    helperfuncs_test.go:24: 2
    helperfuncs_test.go:49: 3
    helperfuncs_test.go:56: 4
    --- FAIL: TestTBHelper/sub \([^)]+\)
        helperfuncs_test.go:59: 5
        helperfuncs_test.go:24: 6
        helperfuncs_test.go:58: 7
    --- FAIL: TestTBHelper/sub2 \([^)]+\)
        helperfuncs_test.go:80: 11
    helperfuncs_test.go:84: recover 12
    helperfuncs_test.go:86: GenericFloat64
    helperfuncs_test.go:87: GenericInt
    helper_test.go:22: 8
    helperfuncs_test.go:73: 9
    helperfuncs_test.go:69: 10
`
	if !regexp.MustCompile(want).Match(out) {
		t.Errorf("got output:\n\n%s\nwant matching:\n\n%s", out, want)
	}
}

func TestTBHelperParallel(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		parallelTestHelper(t)
		return
	}

	testenv.MustHaveExec(t)
	t.Parallel()

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, exe, "-test.run=^TestTBHelperParallel$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, _ := cmd.CombinedOutput()

	t.Logf("output:\n%s", out)

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	// We expect to see one "--- FAIL" line at the start
	// of the log, five lines of "parallel" logging,
	// and a final "FAIL" line at the end of the test.
	const wantLines = 7

	if len(lines) != wantLines {
		t.Fatalf("parallelTestHelper gave %d lines of output; want %d", len(lines), wantLines)
	}
	want := "helperfuncs_test.go:24: parallel"
	if got := strings.TrimSpace(lines[1]); got != want {
		t.Errorf("got second output line %q; want %q", got, want)
	}
}

func BenchmarkTBHelper(b *testing.B) {
	f1 := func() {
		b.Helper()
	}
	f2 := func() {
		b.Helper()
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if i&1 == 0 {
			f1()
		} else {
			f2()
		}
	}
}
