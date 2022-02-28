// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"text/scanner"
)

var update = flag.Bool("update", false, "update .golden files")

// gofmtFlags looks for a comment of the form
//
//	//gofmt flags
//
// within the first maxLines lines of the given file,
// and returns the flags string, if any. Otherwise it
// returns the empty string.
func gofmtFlags(filename string, maxLines int) string {
	f, err := os.Open(filename)
	if err != nil {
		return "" // ignore errors - they will be found later
	}
	defer f.Close()

	// initialize scanner
	var s scanner.Scanner
	s.Init(f)
	s.Error = func(*scanner.Scanner, string) {}       // ignore errors
	s.Mode = scanner.GoTokens &^ scanner.SkipComments // want comments

	// look for //gofmt comment
	for s.Line <= maxLines {
		switch s.Scan() {
		case scanner.Comment:
			const prefix = "//gofmt "
			if t := s.TokenText(); strings.HasPrefix(t, prefix) {
				return strings.TrimSpace(t[len(prefix):])
			}
		case scanner.EOF:
			return ""
		}
	}

	return ""
}

func runTest(t *testing.T, in, out string) {
	// process flags
	*simplifyAST = false
	*rewriteRule = ""
	info, err := os.Lstat(in)
	if err != nil {
		t.Error(err)
		return
	}
	for _, flag := range strings.Split(gofmtFlags(in, 20), " ") {
		elts := strings.SplitN(flag, "=", 2)
		name := elts[0]
		value := ""
		if len(elts) == 2 {
			value = elts[1]
		}
		switch name {
		case "":
			// no flags
		case "-r":
			*rewriteRule = value
		case "-s":
			*simplifyAST = true
		case "-stdin":
			// fake flag - pretend input is from stdin
			info = nil
		default:
			t.Errorf("unrecognized flag name: %s", name)
		}
	}

	initParserMode()
	initRewrite()

	const maxWeight = 2 << 20
	var buf, errBuf bytes.Buffer
	s := newSequencer(maxWeight, &buf, &errBuf)
	s.Add(fileWeight(in, info), func(r *reporter) error {
		return processFile(in, info, nil, r)
	})
	if errBuf.Len() > 0 {
		t.Logf("%q", errBuf.Bytes())
	}
	if s.GetExitCode() != 0 {
		t.Fail()
	}

	expected, err := os.ReadFile(out)
	if err != nil {
		t.Error(err)
		return
	}

	if got := buf.Bytes(); !bytes.Equal(got, expected) {
		if *update {
			if in != out {
				if err := os.WriteFile(out, got, 0666); err != nil {
					t.Error(err)
				}
				return
			}
			// in == out: don't accidentally destroy input
			t.Errorf("WARNING: -update did not rewrite input file %s", in)
		}

		t.Errorf("(gofmt %s) != %s (see %s.gofmt)", in, out, in)
		d, err := diffWithReplaceTempFile(expected, got, in)
		if err == nil {
			t.Errorf("%s", d)
		}
		if err := os.WriteFile(in+".gofmt", got, 0666); err != nil {
			t.Error(err)
		}
	}
}

// TestRewrite processes testdata/*.input files and compares them to the
// corresponding testdata/*.golden files. The gofmt flags used to process
// a file must be provided via a comment of the form
//
//	//gofmt flags
//
// in the processed file within the first 20 lines, if any.
func TestRewrite(t *testing.T) {
	// determine input files
	match, err := filepath.Glob("testdata/*.input")
	if err != nil {
		t.Fatal(err)
	}

	// add larger examples
	match = append(match, "gofmt.go", "gofmt_test.go")

	for _, in := range match {
		out := in // for files where input and output are identical
		if strings.HasSuffix(in, ".input") {
			out = in[:len(in)-len(".input")] + ".golden"
		}
		runTest(t, in, out)
		if in != out {
			// Check idempotence.
			runTest(t, out, out)
		}
	}
}

// Test case for issue 3961.
func TestCRLF(t *testing.T) {
	const input = "testdata/crlf.input"   // must contain CR/LF's
	const golden = "testdata/crlf.golden" // must not contain any CR's

	data, err := os.ReadFile(input)
	if err != nil {
		t.Error(err)
	}
	if !bytes.Contains(data, []byte("\r\n")) {
		t.Errorf("%s contains no CR/LF's", input)
	}

	data, err = os.ReadFile(golden)
	if err != nil {
		t.Error(err)
	}
	if bytes.Contains(data, []byte("\r")) {
		t.Errorf("%s contains CR's", golden)
	}
}

func TestBackupFile(t *testing.T) {
	dir, err := os.MkdirTemp("", "gofmt_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	name, err := backupFile(filepath.Join(dir, "foo.go"), []byte("  package main"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Created: %s", name)
}

func TestDiff(t *testing.T) {
	if _, err := exec.LookPath("diff"); err != nil {
		t.Skipf("skip test on %s: diff command is required", runtime.GOOS)
	}
	in := []byte("first\nsecond\n")
	out := []byte("first\nthird\n")
	filename := "difftest.txt"
	b, err := diffWithReplaceTempFile(in, out, filename)
	if err != nil {
		t.Fatal(err)
	}

	if runtime.GOOS == "windows" {
		b = bytes.ReplaceAll(b, []byte{'\r', '\n'}, []byte{'\n'})
	}

	bs := bytes.SplitN(b, []byte{'\n'}, 3)
	line0, line1 := bs[0], bs[1]

	if prefix := "--- difftest.txt.orig"; !bytes.HasPrefix(line0, []byte(prefix)) {
		t.Errorf("diff: first line should start with `%s`\ngot: %s", prefix, line0)
	}

	if prefix := "+++ difftest.txt"; !bytes.HasPrefix(line1, []byte(prefix)) {
		t.Errorf("diff: second line should start with `%s`\ngot: %s", prefix, line1)
	}

	want := `@@ -1,2 +1,2 @@
 first
-second
+third
`

	if got := string(bs[2]); got != want {
		t.Errorf("diff: got:\n%s\nwant:\n%s", got, want)
	}
}

func TestReplaceTempFilename(t *testing.T) {
	diff := []byte(`--- /tmp/tmpfile1	2017-02-08 00:53:26.175105619 +0900
+++ /tmp/tmpfile2	2017-02-08 00:53:38.415151275 +0900
@@ -1,2 +1,2 @@
 first
-second
+third
`)
	want := []byte(`--- path/to/file.go.orig	2017-02-08 00:53:26.175105619 +0900
+++ path/to/file.go	2017-02-08 00:53:38.415151275 +0900
@@ -1,2 +1,2 @@
 first
-second
+third
`)
	// Check path in diff output is always slash regardless of the
	// os.PathSeparator (`/` or `\`).
	sep := string(os.PathSeparator)
	filename := strings.Join([]string{"path", "to", "file.go"}, sep)
	got, err := replaceTempFilename(diff, filename)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("os.PathSeparator='%s': replacedDiff:\ngot:\n%s\nwant:\n%s", sep, got, want)
	}
}
