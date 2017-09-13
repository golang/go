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

	// want is the list of function names (by package) that should
	// be inlined.
	want := map[string][]string{
		"runtime": {
			"tophash",
			"add",
			"addb",
			"subtractb",
			"(*bmap).keys",
			"bucketShift",
			"bucketMask",
			"fastrand",
			"noescape",

			// TODO: These were modified at some point to be
			// made inlineable, but have since been broken.
			// "nextFreeFast",
		},
		"unicode/utf8": {
			"FullRune",
			"FullRuneInString",
			"RuneLen",
			"ValidRune",
		},
	}

	m := make(map[string]bool)
	pkgs := make([]string, 0, len(want))
	for pname, fnames := range want {
		pkgs = append(pkgs, pname)
		for _, fname := range fnames {
			m[pname+"."+fname] = true
		}
	}

	args := append([]string{"build", "-a", "-gcflags=-m"}, pkgs...)
	cmd := testenv.CleanCmdEnv(exec.Command(testenv.GoToolPath(t), args...))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
	lines := bytes.Split(out, []byte{'\n'})
	curPkg := ""
	for _, l := range lines {
		if bytes.HasPrefix(l, []byte("# ")) {
			curPkg = string(l[2:])
		}
		f := bytes.Split(l, []byte(": can inline "))
		if len(f) < 2 {
			continue
		}
		fn := bytes.TrimSpace(f[1])
		delete(m, curPkg+"."+string(fn))
	}

	for s := range m {
		t.Errorf("function %s not inlined", s)
	}
}
