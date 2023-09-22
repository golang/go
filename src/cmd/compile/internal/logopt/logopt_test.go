// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logopt

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const srcCode = `package x
type pair struct {a,b int}
func bar(y *pair) *int {
	return &y.b
}
var a []int
func foo(w, z *pair) *int {
	if *bar(w) > 0 {
		return bar(z)
	}
	if a[1] > 0 {
		a = a[:2]
	}
	return &a[0]
}

// address taking prevents closure inlining
func n() int {
	foo := func() int { return 1 }
	bar := &foo
	x := (*bar)() + foo()
	return x
}
`

func want(t *testing.T, out string, desired string) {
	// On Windows, Unicode escapes in the JSON output end up "normalized" elsewhere to /u....,
	// so "normalize" what we're looking for to match that.
	s := strings.ReplaceAll(desired, string(os.PathSeparator), "/")
	if !strings.Contains(out, s) {
		t.Errorf("did not see phrase %s in \n%s", s, out)
	}
}

func wantN(t *testing.T, out string, desired string, n int) {
	if strings.Count(out, desired) != n {
		t.Errorf("expected exactly %d occurrences of %s in \n%s", n, desired, out)
	}
}

func TestPathStuff(t *testing.T) {
	sep := string(filepath.Separator)
	if path, whine := parseLogPath("file:///c:foo"); path != "c:foo" || whine != "" { // good path
		t.Errorf("path='%s', whine='%s'", path, whine)
	}
	if path, whine := parseLogPath("file:///foo"); path != sep+"foo" || whine != "" { // good path
		t.Errorf("path='%s', whine='%s'", path, whine)
	}
	if path, whine := parseLogPath("foo"); path != "" || whine == "" { // BAD path
		t.Errorf("path='%s', whine='%s'", path, whine)
	}
	if sep == "\\" { // On WINDOWS ONLY
		if path, whine := parseLogPath("C:/foo"); path != "C:\\foo" || whine != "" { // good path
			t.Errorf("path='%s', whine='%s'", path, whine)
		}
		if path, whine := parseLogPath("c:foo"); path != "" || whine == "" { // BAD path
			t.Errorf("path='%s', whine='%s'", path, whine)
		}
		if path, whine := parseLogPath("/foo"); path != "" || whine == "" { // BAD path
			t.Errorf("path='%s', whine='%s'", path, whine)
		}
	} else { // ON UNIX ONLY
		if path, whine := parseLogPath("/foo"); path != sep+"foo" || whine != "" { // good path
			t.Errorf("path='%s', whine='%s'", path, whine)
		}
	}
}

func TestLogOpt(t *testing.T) {
	t.Parallel()

	testenv.MustHaveGoBuild(t)

	dir := fixSlash(t.TempDir()) // Normalize the directory name as much as possible, for Windows testing
	src := filepath.Join(dir, "file.go")
	if err := os.WriteFile(src, []byte(srcCode), 0644); err != nil {
		t.Fatal(err)
	}

	outfile := filepath.Join(dir, "file.o")

	t.Run("JSON_fails", func(t *testing.T) {
		// Test malformed flag
		out, err := testLogOpt(t, "-json=foo", src, outfile)
		if err == nil {
			t.Error("-json=foo succeeded unexpectedly")
		}
		want(t, out, "option should be")
		want(t, out, "number")

		// Test a version number that is currently unsupported (and should remain unsupported for a while)
		out, err = testLogOpt(t, "-json=9,foo", src, outfile)
		if err == nil {
			t.Error("-json=0,foo succeeded unexpectedly")
		}
		want(t, out, "version must be")

	})

	// replace d (dir)  with t ("tmpdir") and convert path separators to '/'
	normalize := func(out []byte, d, t string) string {
		s := string(out)
		s = strings.ReplaceAll(s, d, t)
		s = strings.ReplaceAll(s, string(os.PathSeparator), "/")
		return s
	}

	// Ensure that <128 byte copies are not reported and that 128-byte copies are.
	// Check at both 1 and 8-byte alignments.
	t.Run("Copy", func(t *testing.T) {
		const copyCode = `package x
func s128a1(x *[128]int8) [128]int8 {
	return *x
}
func s127a1(x *[127]int8) [127]int8 {
	return *x
}
func s16a8(x *[16]int64) [16]int64 {
	return *x
}
func s15a8(x *[15]int64) [15]int64 {
	return *x
}
`
		copy := filepath.Join(dir, "copy.go")
		if err := os.WriteFile(copy, []byte(copyCode), 0644); err != nil {
			t.Fatal(err)
		}
		outcopy := filepath.Join(dir, "copy.o")

		// On not-amd64, test the host architecture and os
		arches := []string{runtime.GOARCH}
		goos0 := runtime.GOOS
		if runtime.GOARCH == "amd64" { // Test many things with "linux" (wasm will get "js")
			arches = []string{"arm", "arm64", "386", "amd64", "mips", "mips64", "loong64", "ppc64le", "riscv64", "s390x", "wasm"}
			goos0 = "linux"
		}

		for _, arch := range arches {
			t.Run(arch, func(t *testing.T) {
				goos := goos0
				if arch == "wasm" {
					goos = "js"
				}
				_, err := testCopy(t, dir, arch, goos, copy, outcopy)
				if err != nil {
					t.Error("-json=0,file://log/opt should have succeeded")
				}
				logged, err := os.ReadFile(filepath.Join(dir, "log", "opt", "x", "copy.json"))
				if err != nil {
					t.Error("-json=0,file://log/opt missing expected log file")
				}
				slogged := normalize(logged, string(uriIfy(dir)), string(uriIfy("tmpdir")))
				t.Logf("%s", slogged)
				want(t, slogged, `{"range":{"start":{"line":3,"character":2},"end":{"line":3,"character":2}},"severity":3,"code":"copy","source":"go compiler","message":"128 bytes"}`)
				want(t, slogged, `{"range":{"start":{"line":9,"character":2},"end":{"line":9,"character":2}},"severity":3,"code":"copy","source":"go compiler","message":"128 bytes"}`)
				wantN(t, slogged, `"code":"copy"`, 2)
			})
		}
	})

	// Some architectures don't fault on nil dereference, so nilchecks are eliminated differently.
	// The N-way copy test also doesn't need to run N-ways N times.
	if runtime.GOARCH != "amd64" {
		return
	}

	t.Run("Success", func(t *testing.T) {
		// This test is supposed to succeed

		// Note 'file://' is the I-Know-What-I-Am-Doing way of specifying a file, also to deal with corner cases for Windows.
		_, err := testLogOptDir(t, dir, "-json=0,file://log/opt", src, outfile)
		if err != nil {
			t.Error("-json=0,file://log/opt should have succeeded")
		}
		logged, err := os.ReadFile(filepath.Join(dir, "log", "opt", "x", "file.json"))
		if err != nil {
			t.Error("-json=0,file://log/opt missing expected log file")
		}
		// All this delicacy with uriIfy and filepath.Join is to get this test to work right on Windows.
		slogged := normalize(logged, string(uriIfy(dir)), string(uriIfy("tmpdir")))
		t.Logf("%s", slogged)
		// below shows proper nilcheck
		want(t, slogged, `{"range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}},"severity":3,"code":"nilcheck","source":"go compiler","message":"",`+
			`"relatedInformation":[{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":4,"character":11},"end":{"line":4,"character":11}}},"message":"inlineLoc"}]}`)
		want(t, slogged, `{"range":{"start":{"line":11,"character":6},"end":{"line":11,"character":6}},"severity":3,"code":"isInBounds","source":"go compiler","message":""}`)
		want(t, slogged, `{"range":{"start":{"line":7,"character":6},"end":{"line":7,"character":6}},"severity":3,"code":"canInlineFunction","source":"go compiler","message":"cost: 35"}`)
		// escape analysis explanation
		want(t, slogged, `{"range":{"start":{"line":7,"character":13},"end":{"line":7,"character":13}},"severity":3,"code":"leak","source":"go compiler","message":"parameter z leaks to ~r0 with derefs=0",`+
			`"relatedInformation":[`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}}},"message":"escflow:    flow: y = z:"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}}},"message":"escflow:      from y := z (assign-pair)"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}}},"message":"escflow:    flow: ~r0 = y:"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":4,"character":11},"end":{"line":4,"character":11}}},"message":"inlineLoc"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}}},"message":"escflow:      from y.b (dot of pointer)"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":4,"character":11},"end":{"line":4,"character":11}}},"message":"inlineLoc"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}}},"message":"escflow:      from \u0026y.b (address-of)"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":4,"character":9},"end":{"line":4,"character":9}}},"message":"inlineLoc"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}}},"message":"escflow:      from ~r0 = \u0026y.b (assign-pair)"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":3},"end":{"line":9,"character":3}}},"message":"escflow:    flow: ~r0 = ~r0:"},`+
			`{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":9,"character":3},"end":{"line":9,"character":3}}},"message":"escflow:      from return ~r0 (return)"}]}`)
	})
}

func testLogOpt(t *testing.T, flag, src, outfile string) (string, error) {
	run := []string{testenv.GoToolPath(t), "tool", "compile", "-p=p", flag, "-o", outfile, src}
	t.Log(run)
	cmd := testenv.Command(t, run[0], run[1:]...)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	return string(out), err
}

func testLogOptDir(t *testing.T, dir, flag, src, outfile string) (string, error) {
	// Notice the specified import path "x"
	run := []string{testenv.GoToolPath(t), "tool", "compile", "-p=x", flag, "-o", outfile, src}
	t.Log(run)
	cmd := testenv.Command(t, run[0], run[1:]...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	return string(out), err
}

func testCopy(t *testing.T, dir, goarch, goos, src, outfile string) (string, error) {
	// Notice the specified import path "x"
	run := []string{testenv.GoToolPath(t), "tool", "compile", "-p=x", "-json=0,file://log/opt", "-o", outfile, src}
	t.Log(run)
	cmd := testenv.Command(t, run[0], run[1:]...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOARCH="+goarch, "GOOS="+goos)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	return string(out), err
}
