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

	if got := buf.Bytes(); bytes.Compare(got, expected) != 0 {
		t.Errorf("(gofmt %s) != %s (see %s.gofmt)", in, out, in)
		d, err := diff(expected, got)
		if err == nil {
			t.Errorf("%s", d)
		}
		ioutil.WriteFile(in+".gofmt", got, 0666)
	}
}

// TODO(gri) Add more test cases!
var tests = []struct {
	in, flags string
}{
	{"gofmt.go", ""},
	{"gofmt_test.go", ""},
	{"testdata/composites.input", "-s"},
	{"testdata/old.input", ""},
	{"testdata/rewrite1.input", "-r=Foo->Bar"},
	{"testdata/rewrite2.input", "-r=int->bool"},
	{"testdata/rewrite3.input", "-r=x->x"},
	{"testdata/stdin*.input", "-stdin"},
	{"testdata/comments.input", ""},
	{"testdata/import.input", ""},
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
