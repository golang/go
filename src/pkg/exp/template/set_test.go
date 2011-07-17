// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"testing"
)

type setParseTest struct {
	name    string
	input   string
	ok      bool
	names   []string
	results []string
}

var setParseTests = []setParseTest{
	{"empty", "", noError,
		nil,
		nil},
	{"one", `{{define "foo"}} FOO {{end}}`, noError,
		[]string{"foo"},
		[]string{`[(text: " FOO ")]`}},
	{"two", `{{define "foo"}} FOO {{end}}{{define "bar"}} BAR {{end}}`, noError,
		[]string{"foo", "bar"},
		[]string{`[(text: " FOO ")]`, `[(text: " BAR ")]`}},
	// errors
	{"missing end", `{{define "foo"}} FOO `, hasError,
		nil,
		nil},
	{"malformed name", `{{define "foo}} FOO `, hasError,
		nil,
		nil},
}

func TestSetParse(t *testing.T) {
	for _, test := range setParseTests {
		set := new(Set)
		err := set.Parse(test.input)
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
		if len(set.tmpl) != len(test.names) {
			t.Errorf("%s: wrong number of templates; wanted %d got %d", test.name, len(test.names), len(set.tmpl))
			continue
		}
		for i, name := range test.names {
			tmpl, ok := set.tmpl[name]
			if !ok {
				t.Errorf("%s: can't find template %q", test.name, name)
				continue
			}
			result := tmpl.root.String()
			if result != test.results[i] {
				t.Errorf("%s=(%q): got\n\t%v\nexpected\n\t%v", test.name, test.input, result, test.results[i])
			}
		}
	}
}

var setExecTests = []execTest{
	{"empty", "", "", nil, true},
	{"text", "some text", "some text", nil, true},
	{"invoke x", `{{template "x" .SI}}`, "TEXT", tVal, true},
	{"invoke x no args", `{{template "x"}}`, "TEXT", tVal, true},
	{"invoke dot int", `{{template "dot" .I}}`, "17", tVal, true},
	{"invoke dot []int", `{{template "dot" .SI}}`, "[3 4 5]", tVal, true},
	{"invoke dotV", `{{template "dotV" .U}}`, "v", tVal, true},
	{"invoke nested int", `{{template "nested" .I}}`, "17", tVal, true},
	{"variable declared by template", `{{template "nested" $x=.SI}},{{index $x 1}}`, "[3 4 5],4", tVal, true},

	// User-defined function: test argument evaluator.
	{"testFunc literal", `{{oneArg "joe"}}`, "oneArg=joe", tVal, true},
	{"testFunc .", `{{oneArg .}}`, "oneArg=joe", "joe", true},
}

const setText1 = `
	{{define "x"}}TEXT{{end}}
	{{define "dotV"}}{{.V}}{{end}}
`

const setText2 = `
	{{define "dot"}}{{.}}{{end}}
	{{define "nested"}}{{template "dot" .}}{{end}}
`

func TestSetExecute(t *testing.T) {
	// Declare a set with a couple of templates first.
	set := new(Set)
	err := set.Parse(setText1)
	if err != nil {
		t.Fatalf("error parsing set: %s", err)
	}
	err = set.Parse(setText2)
	if err != nil {
		t.Fatalf("error parsing set: %s", err)
	}
	testExecute(setExecTests, set, t)
}
