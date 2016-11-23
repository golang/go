// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"io/ioutil"
	"os"
	"path/filepath"
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
	stdin := false
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
			stdin = true
		default:
			t.Errorf("unrecognized flag name: %s", name)
		}
	}

	initParserMode()
	initRewrite()

	var buf bytes.Buffer
	err := processFile(in, nil, &buf, stdin)
	if err != nil {
		t.Error(err)
		return
	}

	expected, err := ioutil.ReadFile(out)
	if err != nil {
		t.Error(err)
		return
	}

	if got := buf.Bytes(); !bytes.Equal(got, expected) {
		if *update {
			if in != out {
				if err := ioutil.WriteFile(out, got, 0666); err != nil {
					t.Error(err)
				}
				return
			}
			// in == out: don't accidentally destroy input
			t.Errorf("WARNING: -update did not rewrite input file %s", in)
		}

		t.Errorf("(gofmt %s) != %s (see %s.gofmt)", in, out, in)
		d, err := diff(expected, got)
		if err == nil {
			t.Errorf("%s", d)
		}
		if err := ioutil.WriteFile(in+".gofmt", got, 0666); err != nil {
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

	data, err := ioutil.ReadFile(input)
	if err != nil {
		t.Error(err)
	}
	if !bytes.Contains(data, []byte("\r\n")) {
		t.Errorf("%s contains no CR/LF's", input)
	}

	data, err = ioutil.ReadFile(golden)
	if err != nil {
		t.Error(err)
	}
	if bytes.Contains(data, []byte("\r")) {
		t.Errorf("%s contains CR's", golden)
	}
}

func TestBackupFile(t *testing.T) {
	dir, err := ioutil.TempDir("", "gofmt_test")
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
