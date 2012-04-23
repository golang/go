// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import (
	"flag"
	"fmt"
	"testing"
)

var debug = flag.Bool("debug", false, "show the errors produced by the tests")

type numberTest struct {
	text      string
	isInt     bool
	isUint    bool
	isFloat   bool
	isComplex bool
	int64
	uint64
	float64
	complex128
}

var numberTests = []numberTest{
	// basics
	{"0", true, true, true, false, 0, 0, 0, 0},
	{"-0", true, true, true, false, 0, 0, 0, 0}, // check that -0 is a uint.
	{"73", true, true, true, false, 73, 73, 73, 0},
	{"073", true, true, true, false, 073, 073, 073, 0},
	{"0x73", true, true, true, false, 0x73, 0x73, 0x73, 0},
	{"-73", true, false, true, false, -73, 0, -73, 0},
	{"+73", true, false, true, false, 73, 0, 73, 0},
	{"100", true, true, true, false, 100, 100, 100, 0},
	{"1e9", true, true, true, false, 1e9, 1e9, 1e9, 0},
	{"-1e9", true, false, true, false, -1e9, 0, -1e9, 0},
	{"-1.2", false, false, true, false, 0, 0, -1.2, 0},
	{"1e19", false, true, true, false, 0, 1e19, 1e19, 0},
	{"-1e19", false, false, true, false, 0, 0, -1e19, 0},
	{"4i", false, false, false, true, 0, 0, 0, 4i},
	{"-1.2+4.2i", false, false, false, true, 0, 0, 0, -1.2 + 4.2i},
	{"073i", false, false, false, true, 0, 0, 0, 73i}, // not octal!
	// complex with 0 imaginary are float (and maybe integer)
	{"0i", true, true, true, true, 0, 0, 0, 0},
	{"-1.2+0i", false, false, true, true, 0, 0, -1.2, -1.2},
	{"-12+0i", true, false, true, true, -12, 0, -12, -12},
	{"13+0i", true, true, true, true, 13, 13, 13, 13},
	// funny bases
	{"0123", true, true, true, false, 0123, 0123, 0123, 0},
	{"-0x0", true, true, true, false, 0, 0, 0, 0},
	{"0xdeadbeef", true, true, true, false, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0},
	// character constants
	{`'a'`, true, true, true, false, 'a', 'a', 'a', 0},
	{`'\n'`, true, true, true, false, '\n', '\n', '\n', 0},
	{`'\\'`, true, true, true, false, '\\', '\\', '\\', 0},
	{`'\''`, true, true, true, false, '\'', '\'', '\'', 0},
	{`'\xFF'`, true, true, true, false, 0xFF, 0xFF, 0xFF, 0},
	{`'パ'`, true, true, true, false, 0x30d1, 0x30d1, 0x30d1, 0},
	{`'\u30d1'`, true, true, true, false, 0x30d1, 0x30d1, 0x30d1, 0},
	{`'\U000030d1'`, true, true, true, false, 0x30d1, 0x30d1, 0x30d1, 0},
	// some broken syntax
	{text: "+-2"},
	{text: "0x123."},
	{text: "1e."},
	{text: "0xi."},
	{text: "1+2."},
	{text: "'x"},
	{text: "'xx'"},
}

func TestNumberParse(t *testing.T) {
	for _, test := range numberTests {
		// If fmt.Sscan thinks it's complex, it's complex.  We can't trust the output
		// because imaginary comes out as a number.
		var c complex128
		typ := itemNumber
		if test.text[0] == '\'' {
			typ = itemCharConstant
		} else {
			_, err := fmt.Sscan(test.text, &c)
			if err == nil {
				typ = itemComplex
			}
		}
		n, err := newNumber(test.text, typ)
		ok := test.isInt || test.isUint || test.isFloat || test.isComplex
		if ok && err != nil {
			t.Errorf("unexpected error for %q: %s", test.text, err)
			continue
		}
		if !ok && err == nil {
			t.Errorf("expected error for %q", test.text)
			continue
		}
		if !ok {
			if *debug {
				fmt.Printf("%s\n\t%s\n", test.text, err)
			}
			continue
		}
		if n.IsComplex != test.isComplex {
			t.Errorf("complex incorrect for %q; should be %t", test.text, test.isComplex)
		}
		if test.isInt {
			if !n.IsInt {
				t.Errorf("expected integer for %q", test.text)
			}
			if n.Int64 != test.int64 {
				t.Errorf("int64 for %q should be %d Is %d", test.text, test.int64, n.Int64)
			}
		} else if n.IsInt {
			t.Errorf("did not expect integer for %q", test.text)
		}
		if test.isUint {
			if !n.IsUint {
				t.Errorf("expected unsigned integer for %q", test.text)
			}
			if n.Uint64 != test.uint64 {
				t.Errorf("uint64 for %q should be %d Is %d", test.text, test.uint64, n.Uint64)
			}
		} else if n.IsUint {
			t.Errorf("did not expect unsigned integer for %q", test.text)
		}
		if test.isFloat {
			if !n.IsFloat {
				t.Errorf("expected float for %q", test.text)
			}
			if n.Float64 != test.float64 {
				t.Errorf("float64 for %q should be %g Is %g", test.text, test.float64, n.Float64)
			}
		} else if n.IsFloat {
			t.Errorf("did not expect float for %q", test.text)
		}
		if test.isComplex {
			if !n.IsComplex {
				t.Errorf("expected complex for %q", test.text)
			}
			if n.Complex128 != test.complex128 {
				t.Errorf("complex128 for %q should be %g Is %g", test.text, test.complex128, n.Complex128)
			}
		} else if n.IsComplex {
			t.Errorf("did not expect complex for %q", test.text)
		}
	}
}

type parseTest struct {
	name   string
	input  string
	ok     bool
	result string // what the user would see in an error message.
}

const (
	noError  = true
	hasError = false
)

var parseTests = []parseTest{
	{"empty", "", noError,
		``},
	{"comment", "{{/*\n\n\n*/}}", noError,
		``},
	{"spaces", " \t\n", noError,
		`" \t\n"`},
	{"text", "some text", noError,
		`"some text"`},
	{"emptyAction", "{{}}", hasError,
		`{{}}`},
	{"field", "{{.X}}", noError,
		`{{.X}}`},
	{"simple command", "{{printf}}", noError,
		`{{printf}}`},
	{"$ invocation", "{{$}}", noError,
		"{{$}}"},
	{"variable invocation", "{{with $x := 3}}{{$x 23}}{{end}}", noError,
		"{{with $x := 3}}{{$x 23}}{{end}}"},
	{"variable with fields", "{{$.I}}", noError,
		"{{$.I}}"},
	{"multi-word command", "{{printf `%d` 23}}", noError,
		"{{printf `%d` 23}}"},
	{"pipeline", "{{.X|.Y}}", noError,
		`{{.X | .Y}}`},
	{"pipeline with decl", "{{$x := .X|.Y}}", noError,
		`{{$x := .X | .Y}}`},
	{"simple if", "{{if .X}}hello{{end}}", noError,
		`{{if .X}}"hello"{{end}}`},
	{"if with else", "{{if .X}}true{{else}}false{{end}}", noError,
		`{{if .X}}"true"{{else}}"false"{{end}}`},
	{"simple range", "{{range .X}}hello{{end}}", noError,
		`{{range .X}}"hello"{{end}}`},
	{"chained field range", "{{range .X.Y.Z}}hello{{end}}", noError,
		`{{range .X.Y.Z}}"hello"{{end}}`},
	{"nested range", "{{range .X}}hello{{range .Y}}goodbye{{end}}{{end}}", noError,
		`{{range .X}}"hello"{{range .Y}}"goodbye"{{end}}{{end}}`},
	{"range with else", "{{range .X}}true{{else}}false{{end}}", noError,
		`{{range .X}}"true"{{else}}"false"{{end}}`},
	{"range over pipeline", "{{range .X|.M}}true{{else}}false{{end}}", noError,
		`{{range .X | .M}}"true"{{else}}"false"{{end}}`},
	{"range []int", "{{range .SI}}{{.}}{{end}}", noError,
		`{{range .SI}}{{.}}{{end}}`},
	{"range 1 var", "{{range $x := .SI}}{{.}}{{end}}", noError,
		`{{range $x := .SI}}{{.}}{{end}}`},
	{"range 2 vars", "{{range $x, $y := .SI}}{{.}}{{end}}", noError,
		`{{range $x, $y := .SI}}{{.}}{{end}}`},
	{"constants", "{{range .SI 1 -3.2i true false 'a'}}{{end}}", noError,
		`{{range .SI 1 -3.2i true false 'a'}}{{end}}`},
	{"template", "{{template `x`}}", noError,
		`{{template "x"}}`},
	{"template with arg", "{{template `x` .Y}}", noError,
		`{{template "x" .Y}}`},
	{"with", "{{with .X}}hello{{end}}", noError,
		`{{with .X}}"hello"{{end}}`},
	{"with with else", "{{with .X}}hello{{else}}goodbye{{end}}", noError,
		`{{with .X}}"hello"{{else}}"goodbye"{{end}}`},
	// Errors.
	{"unclosed action", "hello{{range", hasError, ""},
	{"unmatched end", "{{end}}", hasError, ""},
	{"missing end", "hello{{range .x}}", hasError, ""},
	{"missing end after else", "hello{{range .x}}{{else}}", hasError, ""},
	{"undefined function", "hello{{undefined}}", hasError, ""},
	{"undefined variable", "{{$x}}", hasError, ""},
	{"variable undefined after end", "{{with $x := 4}}{{end}}{{$x}}", hasError, ""},
	{"variable undefined in template", "{{template $v}}", hasError, ""},
	{"declare with field", "{{with $x.Y := 4}}{{end}}", hasError, ""},
	{"template with field ref", "{{template .X}}", hasError, ""},
	{"template with var", "{{template $v}}", hasError, ""},
	{"invalid punctuation", "{{printf 3, 4}}", hasError, ""},
	{"multidecl outside range", "{{with $v, $u := 3}}{{end}}", hasError, ""},
	{"too many decls in range", "{{range $u, $v, $w := 3}}{{end}}", hasError, ""},
	// Equals (and other chars) do not assignments make (yet).
	{"bug0a", "{{$x := 0}}{{$x}}", noError, "{{$x := 0}}{{$x}}"},
	{"bug0b", "{{$x = 1}}{{$x}}", hasError, ""},
	{"bug0c", "{{$x ! 2}}{{$x}}", hasError, ""},
	{"bug0d", "{{$x % 3}}{{$x}}", hasError, ""},
	// Check the parse fails for := rather than comma.
	{"bug0e", "{{range $x := $y := 3}}{{end}}", hasError, ""},
	// Another bug: variable read must ignore following punctuation.
	{"bug1a", "{{$x:=.}}{{$x!2}}", hasError, ""},                     // ! is just illegal here.
	{"bug1b", "{{$x:=.}}{{$x+2}}", hasError, ""},                     // $x+2 should not parse as ($x) (+2).
	{"bug1c", "{{$x:=.}}{{$x +2}}", noError, "{{$x := .}}{{$x +2}}"}, // It's OK with a space.
}

var builtins = map[string]interface{}{
	"printf": fmt.Sprintf,
}

func testParse(doCopy bool, t *testing.T) {
	for _, test := range parseTests {
		tmpl, err := New(test.name).Parse(test.input, "", "", make(map[string]*Tree), builtins)
		switch {
		case err == nil && !test.ok:
			t.Errorf("%q: expected error; got none", test.name)
			continue
		case err != nil && test.ok:
			t.Errorf("%q: unexpected error: %v", test.name, err)
			continue
		case err != nil && !test.ok:
			// expected error, got one
			if *debug {
				fmt.Printf("%s: %s\n\t%s\n", test.name, test.input, err)
			}
			continue
		}
		var result string
		if doCopy {
			result = tmpl.Root.Copy().String()
		} else {
			result = tmpl.Root.String()
		}
		if result != test.result {
			t.Errorf("%s=(%q): got\n\t%v\nexpected\n\t%v", test.name, test.input, result, test.result)
		}
	}
}

func TestParse(t *testing.T) {
	testParse(false, t)
}

// Same as TestParse, but we copy the node first
func TestParseCopy(t *testing.T) {
	testParse(true, t)
}

type isEmptyTest struct {
	name  string
	input string
	empty bool
}

var isEmptyTests = []isEmptyTest{
	{"empty", ``, true},
	{"nonempty", `hello`, false},
	{"spaces only", " \t\n \t\n", true},
	{"definition", `{{define "x"}}something{{end}}`, true},
	{"definitions and space", "{{define `x`}}something{{end}}\n\n{{define `y`}}something{{end}}\n\n", true},
	{"definitions and text", "{{define `x`}}something{{end}}\nx\n{{define `y`}}something{{end}}\ny\n}}", false},
	{"definition and action", "{{define `x`}}something{{end}}{{if 3}}foo{{end}}", false},
}

func TestIsEmpty(t *testing.T) {
	if !IsEmptyTree(nil) {
		t.Errorf("nil tree is not empty")
	}
	for _, test := range isEmptyTests {
		tree, err := New("root").Parse(test.input, "", "", make(map[string]*Tree), nil)
		if err != nil {
			t.Errorf("%q: unexpected error: %v", test.name, err)
			continue
		}
		if empty := IsEmptyTree(tree.Root); empty != test.empty {
			t.Errorf("%q: expected %t got %t", test.name, test.empty, empty)
		}
	}
}
