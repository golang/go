// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"io/ioutil"
	"path/filepath"
	"strings"
	"testing"
)

func runTest(t *testing.T, in, out, flags string) {
	// process flags
	*simplifyAST = false
	*rewriteRule = ""
	stdin := false
	for _, flag := range strings.Split(flags, " ") {
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
	initPrinterMode()
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

var tests = []struct {
	in, flags string
}{
	{"gofmt.go", ""},
	{"gofmt_test.go", ""},
	{"testdata/composites.input", "-s"},
	{"testdata/slices1.input", "-s"},
	{"testdata/slices2.input", "-s"},
	{"testdata/old.input", ""},
	{"testdata/rewrite1.input", "-r=Foo->Bar"},
	{"testdata/rewrite2.input", "-r=int->bool"},
	{"testdata/rewrite3.input", "-r=x->x"},
	{"testdata/rewrite4.input", "-r=(x)->x"},
	{"testdata/rewrite5.input", "-r=x+x->2*x"},
	{"testdata/rewrite6.input", "-r=fun(x)->Fun(x)"},
	{"testdata/rewrite7.input", "-r=fun(x...)->Fun(x)"},
	{"testdata/rewrite8.input", "-r=interface{}->int"},
	{"testdata/stdin*.input", "-stdin"},
	{"testdata/comments.input", ""},
	{"testdata/import.input", ""},
	{"testdata/crlf.input", ""},       // test case for issue 3961; see also TestCRLF
	{"testdata/typeswitch.input", ""}, // test case for issue 4470
}

func TestRewrite(t *testing.T) {
	for _, test := range tests {
		match, err := filepath.Glob(test.in)
		if err != nil {
			t.Error(err)
			continue
		}
		for _, in := range match {
			out := in
			if strings.HasSuffix(in, ".input") {
				out = in[:len(in)-len(".input")] + ".golden"
			}
			runTest(t, in, out, test.flags)
			if in != out {
				// Check idempotence.
				runTest(t, out, out, test.flags)
			}
		}
	}
}

func TestCRLF(t *testing.T) {
	const input = "testdata/crlf.input"   // must contain CR/LF's
	const golden = "testdata/crlf.golden" // must not contain any CR's

	data, err := ioutil.ReadFile(input)
	if err != nil {
		t.Error(err)
	}
	if bytes.Index(data, []byte("\r\n")) < 0 {
		t.Errorf("%s contains no CR/LF's", input)
	}

	data, err = ioutil.ReadFile(golden)
	if err != nil {
		t.Error(err)
	}
	if bytes.Index(data, []byte("\r")) >= 0 {
		t.Errorf("%s contains CR's", golden)
	}
}
