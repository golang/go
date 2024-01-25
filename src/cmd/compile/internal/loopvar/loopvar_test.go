// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loopvar_test

import (
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

type testcase struct {
	lvFlag      string // ==-2, -1, 0, 1, 2
	buildExpect string // message, if any
	expectRC    int
	files       []string
}

var for_files = []string{
	"for_esc_address.go",             // address of variable
	"for_esc_closure.go",             // closure of variable
	"for_esc_minimal_closure.go",     // simple closure of variable
	"for_esc_method.go",              // method value of variable
	"for_complicated_esc_address.go", // modifies loop index in body
}

var range_files = []string{
	"range_esc_address.go",         // address of variable
	"range_esc_closure.go",         // closure of variable
	"range_esc_minimal_closure.go", // simple closure of variable
	"range_esc_method.go",          // method value of variable
}

var cases = []testcase{
	{"-1", "", 11, for_files[:1]},
	{"0", "", 0, for_files[:1]},
	{"1", "", 0, for_files[:1]},
	{"2", "loop variable i now per-iteration,", 0, for_files},

	{"-1", "", 11, range_files[:1]},
	{"0", "", 0, range_files[:1]},
	{"1", "", 0, range_files[:1]},
	{"2", "loop variable i now per-iteration,", 0, range_files},

	{"1", "", 0, []string{"for_nested.go"}},
}

// TestLoopVar checks that the GOEXPERIMENT and debug flags behave as expected.
func TestLoopVarGo1_21(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)
	tmpdir := t.TempDir()
	output := filepath.Join(tmpdir, "foo.exe")

	for i, tc := range cases {
		for _, f := range tc.files {
			source := f
			cmd := testenv.Command(t, gocmd, "build", "-o", output, "-gcflags=-lang=go1.21 -d=loopvar="+tc.lvFlag, source)
			cmd.Env = append(cmd.Env, "GOEXPERIMENT=loopvar", "HOME="+tmpdir)
			cmd.Dir = "testdata"
			t.Logf("File %s loopvar=%s expect '%s' exit code %d", f, tc.lvFlag, tc.buildExpect, tc.expectRC)
			b, e := cmd.CombinedOutput()
			if e != nil {
				t.Error(e)
			}
			if tc.buildExpect != "" {
				s := string(b)
				if !strings.Contains(s, tc.buildExpect) {
					t.Errorf("File %s test %d expected to match '%s' with \n-----\n%s\n-----", f, i, tc.buildExpect, s)
				}
			}
			// run what we just built.
			cmd = testenv.Command(t, output)
			b, e = cmd.CombinedOutput()
			if tc.expectRC != 0 {
				if e == nil {
					t.Errorf("Missing expected error, file %s, case %d", f, i)
				} else if ee, ok := (e).(*exec.ExitError); !ok || ee.ExitCode() != tc.expectRC {
					t.Error(e)
				} else {
					// okay
				}
			} else if e != nil {
				t.Error(e)
			}
		}
	}
}

func TestLoopVarInlinesGo1_21(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)
	tmpdir := t.TempDir()

	root := "cmd/compile/internal/loopvar/testdata/inlines"

	f := func(pkg string) string {
		// This disables the loopvar change, except for the specified package.
		// The effect should follow the package, even though everything (except "c")
		// is inlined.
		cmd := testenv.Command(t, gocmd, "run", "-gcflags="+root+"/...=-lang=go1.21", "-gcflags="+pkg+"=-d=loopvar=1", root)
		cmd.Env = append(cmd.Env, "GOEXPERIMENT=noloopvar", "HOME="+tmpdir)
		cmd.Dir = filepath.Join("testdata", "inlines")

		b, e := cmd.CombinedOutput()
		if e != nil {
			t.Error(e)
		}
		return string(b)
	}

	a := f(root + "/a")
	b := f(root + "/b")
	c := f(root + "/c")
	m := f(root)

	t.Logf(a)
	t.Logf(b)
	t.Logf(c)
	t.Logf(m)

	if !strings.Contains(a, "f, af, bf, abf, cf sums = 100, 45, 100, 100, 100") {
		t.Errorf("Did not see expected value of a")
	}
	if !strings.Contains(b, "f, af, bf, abf, cf sums = 100, 100, 45, 45, 100") {
		t.Errorf("Did not see expected value of b")
	}
	if !strings.Contains(c, "f, af, bf, abf, cf sums = 100, 100, 100, 100, 45") {
		t.Errorf("Did not see expected value of c")
	}
	if !strings.Contains(m, "f, af, bf, abf, cf sums = 45, 100, 100, 100, 100") {
		t.Errorf("Did not see expected value of m")
	}
}

func countMatches(s, re string) int {
	slice := regexp.MustCompile(re).FindAllString(s, -1)
	return len(slice)
}

func TestLoopVarHashes(t *testing.T) {
	// This behavior does not depend on Go version (1.21 or greater)
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)
	tmpdir := t.TempDir()

	root := "cmd/compile/internal/loopvar/testdata/inlines"

	f := func(hash string) string {
		// This disables the loopvar change, except for the specified hash pattern.
		// -trimpath is necessary so we get the same answer no matter where the
		// Go repository is checked out. This is not normally a concern since people
		// do not normally rely on the meaning of specific hashes.
		cmd := testenv.Command(t, gocmd, "run", "-trimpath", root)
		cmd.Env = append(cmd.Env, "GOCOMPILEDEBUG=loopvarhash="+hash, "HOME="+tmpdir)
		cmd.Dir = filepath.Join("testdata", "inlines")

		b, _ := cmd.CombinedOutput()
		// Ignore the error, sometimes it's supposed to fail, the output test will catch it.
		return string(b)
	}

	for _, arg := range []string{"v001100110110110010100100", "vx336ca4"} {
		m := f(arg)
		t.Logf(m)

		mCount := countMatches(m, "loopvarhash triggered cmd/compile/internal/loopvar/testdata/inlines/main.go:27:6: .* 001100110110110010100100")
		otherCount := strings.Count(m, "loopvarhash")
		if mCount < 1 {
			t.Errorf("%s: did not see triggered main.go:27:6", arg)
		}
		if mCount != otherCount {
			t.Errorf("%s: too many matches", arg)
		}
		mCount = countMatches(m, "cmd/compile/internal/loopvar/testdata/inlines/main.go:27:6: .* \\[bisect-match 0x7802e115b9336ca4\\]")
		otherCount = strings.Count(m, "[bisect-match ")
		if mCount < 1 {
			t.Errorf("%s: did not see bisect-match for main.go:27:6", arg)
		}
		if mCount != otherCount {
			t.Errorf("%s: too many matches", arg)
		}

		// This next test carefully dodges a bug-to-be-fixed with inlined locations for ir.Names.
		if !strings.Contains(m, ", 100, 100, 100, 100") {
			t.Errorf("%s: did not see expected value of m run", arg)
		}
	}
}

// TestLoopVarVersionEnableFlag checks for loopvar transformation enabled by command line flag (1.22).
func TestLoopVarVersionEnableFlag(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)

	// loopvar=3 logs info but does not change loopvarness
	cmd := testenv.Command(t, gocmd, "run", "-gcflags=-lang=go1.22 -d=loopvar=3", "opt.go")
	cmd.Dir = filepath.Join("testdata")

	b, err := cmd.CombinedOutput()
	m := string(b)

	t.Logf(m)

	yCount := strings.Count(m, "opt.go:16:6: loop variable private now per-iteration, heap-allocated (loop inlined into ./opt.go:29)")
	nCount := strings.Count(m, "shared")

	if yCount != 1 {
		t.Errorf("yCount=%d != 1", yCount)
	}
	if nCount > 0 {
		t.Errorf("nCount=%d > 0", nCount)
	}
	if err != nil {
		t.Errorf("err=%v != nil", err)
	}
}

// TestLoopVarVersionEnableGoBuild checks for loopvar transformation enabled by go:build version (1.22).
func TestLoopVarVersionEnableGoBuild(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)

	// loopvar=3 logs info but does not change loopvarness
	cmd := testenv.Command(t, gocmd, "run", "-gcflags=-lang=go1.21 -d=loopvar=3", "opt-122.go")
	cmd.Dir = filepath.Join("testdata")

	b, err := cmd.CombinedOutput()
	m := string(b)

	t.Logf(m)

	yCount := strings.Count(m, "opt-122.go:18:6: loop variable private now per-iteration, heap-allocated (loop inlined into ./opt-122.go:31)")
	nCount := strings.Count(m, "shared")

	if yCount != 1 {
		t.Errorf("yCount=%d != 1", yCount)
	}
	if nCount > 0 {
		t.Errorf("nCount=%d > 0", nCount)
	}
	if err != nil {
		t.Errorf("err=%v != nil", err)
	}
}

// TestLoopVarVersionDisableFlag checks for loopvar transformation DISABLED by command line version (1.21).
func TestLoopVarVersionDisableFlag(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)

	// loopvar=3 logs info but does not change loopvarness
	cmd := testenv.Command(t, gocmd, "run", "-gcflags=-lang=go1.21 -d=loopvar=3", "opt.go")
	cmd.Dir = filepath.Join("testdata")

	b, err := cmd.CombinedOutput()
	m := string(b)

	t.Logf(m) // expect error

	yCount := strings.Count(m, "opt.go:16:6: loop variable private now per-iteration, heap-allocated (loop inlined into ./opt.go:29)")
	nCount := strings.Count(m, "shared")

	if yCount != 0 {
		t.Errorf("yCount=%d != 0", yCount)
	}
	if nCount > 0 {
		t.Errorf("nCount=%d > 0", nCount)
	}
	if err == nil { // expect error
		t.Errorf("err=%v == nil", err)
	}
}

// TestLoopVarVersionDisableGoBuild checks for loopvar transformation DISABLED by go:build version (1.21).
func TestLoopVarVersionDisableGoBuild(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)

	// loopvar=3 logs info but does not change loopvarness
	cmd := testenv.Command(t, gocmd, "run", "-gcflags=-lang=go1.22 -d=loopvar=3", "opt-121.go")
	cmd.Dir = filepath.Join("testdata")

	b, err := cmd.CombinedOutput()
	m := string(b)

	t.Logf(m) // expect error

	yCount := strings.Count(m, "opt-121.go:18:6: loop variable private now per-iteration, heap-allocated (loop inlined into ./opt-121.go:31)")
	nCount := strings.Count(m, "shared")

	if yCount != 0 {
		t.Errorf("yCount=%d != 0", yCount)
	}
	if nCount > 0 {
		t.Errorf("nCount=%d > 0", nCount)
	}
	if err == nil { // expect error
		t.Errorf("err=%v == nil", err)
	}
}
