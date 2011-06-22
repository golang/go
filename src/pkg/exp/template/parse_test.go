// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"testing"
)

const dumpErrors = true

type numberTest struct {
	text      string
	isInt     bool
	isUint    bool
	isFloat   bool
	imaginary bool
	int64
	uint64
	float64
}

var numberTests = []numberTest{
	// basics
	{"0", true, true, true, false, 0, 0, 0},
	{"73", true, true, true, false, 73, 73, 73},
	{"-73", true, false, true, false, -73, 0, -73},
	{"+73", true, false, true, false, 73, 0, 73},
	{"100", true, true, true, false, 100, 100, 100},
	{"1e9", true, true, true, false, 1e9, 1e9, 1e9},
	{"-1e9", true, false, true, false, -1e9, 0, -1e9},
	{"-1.2", false, false, true, false, 0, 0, -1.2},
	{"1e19", false, true, true, false, 0, 1e19, 1e19},
	{"-1e19", false, false, true, false, 0, 0, -1e19},
	{"4i", false, false, true, true, 0, 0, 4},
	// funny bases
	{"0123", true, true, true, false, 0123, 0123, 0123},
	{"-0x0", true, false, true, false, 0, 0, 0},
	{"0xdeadbeef", true, true, true, false, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef},
	// some broken syntax
	{text: "+-2"},
	{text: "0x123."},
	{text: "1e."},
	{text: "0xi."},
}

func TestNumberParse(t *testing.T) {
	for _, test := range numberTests {
		n, err := newNumber(test.text)
		ok := test.isInt || test.isUint || test.isFloat
		if ok && err != nil {
			t.Errorf("unexpected error for %q", test.text)
			continue
		}
		if !ok && err == nil {
			t.Errorf("expected error for %q", test.text)
			continue
		}
		if !ok {
			continue
		}
		if n.imaginary != test.imaginary {
			t.Errorf("imaginary incorrect for %q; should be %t", test.text, test.imaginary)
		}
		if test.isInt {
			if !n.isInt {
				t.Errorf("expected integer for %q", test.text)
			}
			if n.int64 != test.int64 {
				t.Errorf("int64 for %q should be %d is %d", test.text, test.int64, n.int64)
			}
		} else if n.isInt {
			t.Errorf("did not expect integer for %q", test.text)
		}
		if test.isUint {
			if !n.isUint {
				t.Errorf("expected unsigned integer for %q", test.text)
			}
			if n.uint64 != test.uint64 {
				t.Errorf("uint64 for %q should be %d is %d", test.text, test.uint64, n.uint64)
			}
		} else if n.isUint {
			t.Errorf("did not expect unsigned integer for %q", test.text)
		}
		if test.isFloat {
			if !n.isFloat {
				t.Errorf("expected float for %q", test.text)
			}
			if n.float64 != test.float64 {
				t.Errorf("float64 for %q should be %g is %g", test.text, test.float64, n.float64)
			}
		} else if n.isFloat {
			t.Errorf("did not expect float for %q", test.text)
		}
	}
}

func num(s string) *numberNode {
	n, err := newNumber(s)
	if err != nil {
		panic(err)
	}
	return n
}

type parseTest struct {
	name   string
	input  string
	ok     bool
	result string
}

const (
	noError  = true
	hasError = false
)

var parseTests = []parseTest{
	{"empty", "", noError,
		`[]`},
	{"spaces", " \t\n", noError,
		`[(text: " \t\n")]`},
	{"text", "some text", noError,
		`[(text: "some text")]`},
	{"emptyMeta", "{{}}", noError,
		`[(action: [])]`},
	{"simple command", "{{hello}}", noError,
		`[(action: [(command: [I=hello])])]`},
	{"multi-word command", "{{hello world}}", noError,
		`[(action: [(command: [I=hello I=world])])]`},
	{"multi-word command with number", "{{hello 80}}", noError,
		`[(action: [(command: [I=hello N=80])])]`},
	{"multi-word command with string", "{{hello `quoted text`}}", noError,
		"[(action: [(command: [I=hello S=`quoted text`])])]"},
	{"pipeline", "{{hello|world}}", noError,
		`[(action: [(command: [I=hello]) (command: [I=world])])]`},
	{"simple range", "{{range .x}}hello{{end}}", noError,
		`[({{range F=.x}} [(text: "hello")])]`},
	{"nested range", "{{range .x}}hello{{range .y}}goodbye{{end}}{{end}}", noError,
		`[({{range F=.x}} [(text: "hello")({{range F=.y}} [(text: "goodbye")])])]`},
	{"range with else", "{{range .x}}true{{else}}false{{end}}", noError,
		`[({{range F=.x}} [(text: "true")] {{else}} [(text: "false")])]`},
	// Errors.
	{"unclosed action", "hello{{range", hasError, ""},
	{"not a field", "hello{{range x}}{{end}}", hasError, ""},
	{"missing end", "hello{{range .x}}", hasError, ""},
	{"missing end after else", "hello{{range .x}}{{else}}", hasError, ""},
}

func TestParse(t *testing.T) {
	for _, test := range parseTests {
		tmpl := New(test.name)
		err := tmpl.Parse(test.input)
		switch {
		case err == nil && !test.ok:
			t.Errorf("%q: expected error; got none", test.name)
			continue
		case err != nil && test.ok:
			t.Errorf("%q: unexpected error: %v", test.name, err)
			continue
		case err != nil && !test.ok:
			// expected error, got one
			if dumpErrors {
				fmt.Printf("%s: %s\n\t%s\n", test.name, test.input, err)
			}
			continue
		}
		result := tmpl.root.String()
		if result != test.result {
			t.Errorf("%s=(%q): got\n\t%v\nexpected\n\t%v", test.name, test.input, result, test.result)
		}
	}
}
