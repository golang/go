// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
)

// T has lots of interesting pieces to use to test execution.
type T struct {
	// Basics
	I           int
	U16         uint16
	X           string
	FloatZero   float64
	ComplexZero float64
	// Nested structs.
	U *U
	// Slices
	SI      []int
	SIEmpty []int
	SB      []bool
	// Maps
	MSI      map[string]int
	MSIone   map[string]int // one element, for deterministic output
	MSIEmpty map[string]int
}

// Simple methods with and without arguments.
func (t *T) Method0() string {
	return "resultOfMethod0"
}

func (t *T) Method1(a int) int {
	return a
}

func (t *T) Method2(a uint16, b string) string {
	return fmt.Sprintf("Method2: %d %s", a, b)
}

func (t *T) MAdd(a int, b []int) []int {
	v := make([]int, len(b))
	for i, x := range b {
		v[i] = x + a
	}
	return v
}

// MSort is used to sort map keys for stable output. (Nice trick!)
func (t *T) MSort(m map[string]int) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.SortStrings(keys)
	return keys
}

// EPERM returns a value and an os.Error according to its argument.
func (t *T) EPERM(error bool) (bool, os.Error) {
	if error {
		return true, os.EPERM
	}
	return false, nil
}

type U struct {
	V string
}

var tVal = &T{
	I:      17,
	U16:    16,
	X:      "x",
	U:      &U{"v"},
	SI:     []int{3, 4, 5},
	SB:     []bool{true, false},
	MSI:    map[string]int{"one": 1, "two": 2, "three": 3},
	MSIone: map[string]int{"one": 1},
}

type execTest struct {
	name   string
	input  string
	output string
	data   interface{}
	ok     bool
}

var execTests = []execTest{
	// Trivial cases.
	{"empty", "", "", nil, true},
	{"text", "some text", "some text", nil, true},
	// Fields of structs.
	{".X", "-{{.X}}-", "-x-", tVal, true},
	{".U.V", "-{{.U.V}}-", "-v-", tVal, true},
	// Dots of all kinds to test basic evaluation.
	{"dot int", "<{{.}}>", "<13>", 13, true},
	{"dot uint", "<{{.}}>", "<14>", uint(14), true},
	{"dot float", "<{{.}}>", "<15.1>", 15.1, true},
	{"dot bool", "<{{.}}>", "<true>", true, true},
	{"dot complex", "<{{.}}>", "<(16.2-17i)>", 16.2 - 17i, true},
	{"dot string", "<{{.}}>", "<hello>", "hello", true},
	{"dot slice", "<{{.}}>", "<[-1 -2 -3]>", []int{-1, -2, -3}, true},
	{"dot map", "<{{.}}>", "<map[two:22 one:11]>", map[string]int{"one": 11, "two": 22}, true},
	{"dot struct", "<{{.}}>", "<{7 seven}>", struct {
		a int
		b string
	}{7, "seven"}, true},
	// Method calls.
	{".Method0", "-{{.Method0}}-", "-resultOfMethod0-", tVal, true},
	{".Method1(1234)", "-{{.Method1 1234}}-", "-1234-", tVal, true},
	{".Method1(.I)", "-{{.Method1 .I}}-", "-17-", tVal, true},
	{".Method2(3, .X)", "-{{.Method2 3 .X}}-", "-Method2: 3 x-", tVal, true},
	{".Method2(.U16, `str`)", "-{{.Method2 .U16 `str`}}-", "-Method2: 16 str-", tVal, true},
	// Pipelines.
	{"pipeline", "-{{.Method0 | .Method2 .U16}}-", "-Method2: 16 resultOfMethod0-", tVal, true},
	// If.
	{"if true", "{{if true}}TRUE{{end}}", "TRUE", tVal, true},
	{"if false", "{{if false}}TRUE{{else}}FALSE{{end}}", "FALSE", tVal, true},
	{"if 1", "{{if 1}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0", "{{if 0}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if 1.5", "{{if 1.5}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0.0", "{{if .FloatZero}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if 1.5i", "{{if 1.5i}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0.0i", "{{if .ComplexZero}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if emptystring", "{{if ``}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if string", "{{if `notempty`}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if emptyslice", "{{if .SIEmpty}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if slice", "{{if .SI}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if emptymap", "{{if .MSIEmpty}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if map", "{{if .MSI}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	// Printf.
	{"printf", `{{printf "hello, printf"}}`, "hello, printf", tVal, true},
	{"printf int", `{{printf "%04x" 127}}`, "007f", tVal, true},
	{"printf float", `{{printf "%g" 3.5}}`, "3.5", tVal, true},
	{"printf complex", `{{printf "%g" 1+7i}}`, "(1+7i)", tVal, true},
	{"printf string", `{{printf "%s" "hello"}}`, "hello", tVal, true},
	{"printf function", `{{printf "%#q" gopher}}`, "`gopher`", tVal, true},
	{"printf field", `{{printf "%s" .U.V}}`, "v", tVal, true},
	{"printf method", `{{printf "%s" .Method0}}`, "resultOfMethod0", tVal, true},
	{"printf lots", `{{printf "%d %s %g %s" 127 "hello" 7-3i .Method0}}`, "127 hello (7-3i) resultOfMethod0", tVal, true},
	// HTML.
	{"html", `{{html "<script>alert(\"XSS\");</script>"}}`,
		"&lt;script&gt;alert(&#34;XSS&#34;);&lt;/script&gt;", nil, true},
	{"html pipeline", `{{printf "<script>alert(\"XSS\");</script>" | html}}`,
		"&lt;script&gt;alert(&#34;XSS&#34;);&lt;/script&gt;", nil, true},
	// Booleans
	{"not", "{{not true}} {{not false}}", "false true", nil, true},
	{"and", "{{and 0 0}} {{and 1 0}} {{and 0 1}} {{and 1 1}}", "false false false true", nil, true},
	{"or", "{{or 0 0}} {{or 1 0}} {{or 0 1}} {{or 1 1}}", "false true true true", nil, true},
	{"boolean if", "{{if and true 1 `hi`}}TRUE{{else}}FALSE{{end}}", "TRUE", tVal, true},
	{"boolean if not", "{{if and true 1 `hi` | not}}TRUE{{else}}FALSE{{end}}", "FALSE", nil, true},
	// With.
	{"with true", "{{with true}}{{.}}{{end}}", "true", tVal, true},
	{"with false", "{{with false}}{{.}}{{else}}FALSE{{end}}", "FALSE", tVal, true},
	{"with 1", "{{with 1}}{{.}}{{else}}ZERO{{end}}", "1", tVal, true},
	{"with 0", "{{with 0}}{{.}}{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"with 1.5", "{{with 1.5}}{{.}}{{else}}ZERO{{end}}", "1.5", tVal, true},
	{"with 0.0", "{{with .FloatZero}}{{.}}{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"with 1.5i", "{{with 1.5i}}{{.}}{{else}}ZERO{{end}}", "(0+1.5i)", tVal, true},
	{"with 0.0i", "{{with .ComplexZero}}{{.}}{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"with emptystring", "{{with ``}}{{.}}{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"with string", "{{with `notempty`}}{{.}}{{else}}EMPTY{{end}}", "notempty", tVal, true},
	{"with emptyslice", "{{with .SIEmpty}}{{.}}{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"with slice", "{{with .SI}}{{.}}{{else}}EMPTY{{end}}", "[3 4 5]", tVal, true},
	{"with emptymap", "{{with .MSIEmpty}}{{.}}{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"with map", "{{with .MSIone}}{{.}}{{else}}EMPTY{{end}}", "map[one:1]", tVal, true},
	// Range.
	{"range []int", "{{range .SI}}-{{.}}-{{end}}", "-3--4--5-", tVal, true},
	{"range empty no else", "{{range .SIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range []int else", "{{range .SI}}-{{.}}-{{else}}EMPTY{{end}}", "-3--4--5-", tVal, true},
	{"range empty else", "{{range .SIEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"range []bool", "{{range .SB}}-{{.}}-{{end}}", "-true--false-", tVal, true},
	{"range []int method", "{{range .SI | .MAdd .I}}-{{.}}-{{end}}", "-20--21--22-", tVal, true},
	{"range map", "{{range .MSI | .MSort}}-{{.}}-{{end}}", "-one--three--two-", tVal, true},
	{"range empty map no else", "{{range .MSIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range map else", "{{range .MSI | .MSort}}-{{.}}-{{else}}EMPTY{{end}}", "-one--three--two-", tVal, true},
	{"range empty map else", "{{range .MSIEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	// Error handling.
	{"error method, error", "{{.EPERM true}}", "", tVal, false},
	{"error method, no error", "{{.EPERM false}}", "false", tVal, true},
}

func gopher() string {
	return "gopher"
}

func testExecute(execTests []execTest, set *Set, t *testing.T) {
	b := new(bytes.Buffer)
	funcs := FuncMap{"gopher": gopher}
	for _, test := range execTests {
		tmpl := New(test.name).Funcs(funcs)
		err := tmpl.Parse(test.input)
		if err != nil {
			t.Errorf("%s: parse error: %s", test.name, err)
			continue
		}
		b.Reset()
		err = tmpl.ExecuteInSet(b, test.data, set)
		switch {
		case !test.ok && err == nil:
			t.Errorf("%s: expected error; got none", test.name)
			continue
		case test.ok && err != nil:
			t.Errorf("%s: unexpected execute error: %s", test.name, err)
			continue
		case !test.ok && err != nil:
			continue
		}
		result := b.String()
		if result != test.output {
			t.Errorf("%s: expected\n\t%q\ngot\n\t%q", test.name, test.output, result)
		}
	}
}

func TestExecute(t *testing.T) {
	testExecute(execTests, nil, t)
}

// Check that an error from a method flows back to the top.
func TestExecuteError(t *testing.T) {
	b := new(bytes.Buffer)
	tmpl := New("error")
	err := tmpl.Parse("{{.EPERM true}}")
	if err != nil {
		t.Fatalf("parse error: %s", err)
	}
	err = tmpl.Execute(b, tVal)
	if err == nil {
		t.Errorf("expected error; got none")
	} else if !strings.Contains(err.String(), os.EPERM.String()) {
		t.Errorf("expected os.EPERM; got %s", err)
	}
}
