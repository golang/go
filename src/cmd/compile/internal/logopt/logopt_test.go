// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logopt

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
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
`

func want(t *testing.T, out string, desired string) {
	if !strings.Contains(out, desired) {
		t.Errorf("did not see phrase %s in \n%s", desired, out)
	}
}

func TestLogOpt(t *testing.T) {
	t.Parallel()

	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "TestLogOpt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	dir = fixSlash(dir) // Normalize the directory name as much as possible, for Windows testing
	src := filepath.Join(dir, "file.go")
	if err := ioutil.WriteFile(src, []byte(srcCode), 0644); err != nil {
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

	// Some architectures don't fault on nil dereference, so nilchecks are eliminated differently.
	if runtime.GOARCH != "amd64" {
		return
	}

	t.Run("Success", func(t *testing.T) {
		// This test is supposed to succeed

		// replace d (dir)  with t ("tmpdir") and convert path separators to '/'
		normalize := func(out []byte, d, t string) string {
			s := string(out)
			s = strings.ReplaceAll(s, d, t)
			s = strings.ReplaceAll(s, string(os.PathSeparator), "/")
			return s
		}

		// Note 'file://' is the I-Know-What-I-Am-Doing way of specifying a file, also to deal with corner cases for Windows.
		_, err := testLogOptDir(t, dir, "-json=0,file://log/opt", src, outfile)
		if err != nil {
			t.Error("-json=0,file://log/opt should have succeeded")
		}
		logged, err := ioutil.ReadFile(filepath.Join(dir, "log", "opt", "x", "file.json"))
		if err != nil {
			t.Error("-json=0,file://log/opt missing expected log file")
		}
		// All this delicacy with uriIfy and filepath.Join is to get this test to work right on Windows.
		slogged := normalize(logged, string(uriIfy(dir)), string(uriIfy("tmpdir")))
		t.Logf("%s", slogged)
		// below shows proper inlining and nilcheck
		want(t, slogged, `{"range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}},"severity":3,"code":"nilcheck","source":"go compiler","message":"","relatedInformation":[{"location":{"uri":"file://tmpdir/file.go","range":{"start":{"line":4,"character":11},"end":{"line":4,"character":11}}},"message":"inlineLoc"}]}`)
		want(t, slogged, `{"range":{"start":{"line":11,"character":6},"end":{"line":11,"character":6}},"severity":3,"code":"isInBounds","source":"go compiler","message":""}`)
		want(t, slogged, `{"range":{"start":{"line":7,"character":6},"end":{"line":7,"character":6}},"severity":3,"code":"canInlineFunction","source":"go compiler","message":"cost: 35"}`)
		want(t, slogged, `{"range":{"start":{"line":9,"character":13},"end":{"line":9,"character":13}},"severity":3,"code":"inlineCall","source":"go compiler","message":"x.bar"}`)
		want(t, slogged, `{"range":{"start":{"line":8,"character":9},"end":{"line":8,"character":9}},"severity":3,"code":"inlineCall","source":"go compiler","message":"x.bar"}`)
	})
}

func testLogOpt(t *testing.T, flag, src, outfile string) (string, error) {
	run := []string{testenv.GoToolPath(t), "tool", "compile", flag, "-o", outfile, src}
	t.Log(run)
	cmd := exec.Command(run[0], run[1:]...)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	return string(out), err
}

func testLogOptDir(t *testing.T, dir, flag, src, outfile string) (string, error) {
	// Notice the specified import path "x"
	run := []string{testenv.GoToolPath(t), "tool", "compile", "-p", "x", flag, "-o", outfile, src}
	t.Log(run)
	cmd := exec.Command(run[0], run[1:]...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	return string(out), err
}
