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


func runTest(t *testing.T, dirname, in, out, flags string) {
	in = filepath.Join(dirname, in)
	out = filepath.Join(dirname, out)

	// process flags
	*simplifyAST = false
	*rewriteRule = ""
	for _, flag := range strings.Split(flags, " ", -1) {
		elts := strings.Split(flag, "=", 2)
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
		default:
			t.Errorf("unrecognized flag name: %s", name)
		}
	}

	initParserMode()
	initPrinterMode()
	initRewrite()

	var buf bytes.Buffer
	err := processFile(in, nil, &buf)
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
		ioutil.WriteFile(in+".gofmt", got, 0666)
	}
}


// TODO(gri) Add more test cases!
var tests = []struct {
	dirname, in, out, flags string
}{
	{".", "gofmt.go", "gofmt.go", ""},
	{".", "gofmt_test.go", "gofmt_test.go", ""},
	{"testdata", "composites.input", "composites.golden", "-s"},
	{"testdata", "rewrite1.input", "rewrite1.golden", "-r=Foo->Bar"},
}


func TestRewrite(t *testing.T) {
	for _, test := range tests {
		runTest(t, test.dirname, test.in, test.out, test.flags)
	}
}
