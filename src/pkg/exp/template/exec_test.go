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
	I   int
	U16 uint16
	X   string
	// Nested structs.
	U *U
	// Slices
	SI     []int
	SEmpty []int
	SB     []bool
	// Maps
	MSI      map[string]int
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
	I:   17,
	U16: 16,
	X:   "x",
	U:   &U{"v"},
	SI:  []int{3, 4, 5},
	SB:  []bool{true, false},
	MSI: map[string]int{"one": 1, "two": 2, "three": 3},
}

type execTest struct {
	name   string
	input  string
	output string
	data   interface{}
	ok     bool
}

var execTests = []execTest{
	{"empty", "", "", nil, true},
	{"text", "some text", "some text", nil, true},
	{".X", "-{{.X}}-", "-x-", tVal, true},
	{".U.V", "-{{.U.V}}-", "-v-", tVal, true},
	{".Method0", "-{{.Method0}}-", "-resultOfMethod0-", tVal, true},
	{".Method1(1234)", "-{{.Method1 1234}}-", "-1234-", tVal, true},
	{".Method1(.I)", "-{{.Method1 .I}}-", "-17-", tVal, true},
	{".Method2(3, .X)", "-{{.Method2 3 .X}}-", "-Method2: 3 x-", tVal, true},
	{".Method2(.U16, `str`)", "-{{.Method2 .U16 `str`}}-", "-Method2: 16 str-", tVal, true},
	{"pipeline", "-{{.Method0 | .Method2 .U16}}-", "-Method2: 16 resultOfMethod0-", tVal, true},
	{"range []int", "{{range .SI}}-{{.}}-{{end}}", "-3--4--5-", tVal, true},
	{"range empty no else", "{{range .SEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range []int else", "{{range .SI}}-{{.}}-{{else}}EMPTY{{end}}", "-3--4--5-", tVal, true},
	{"range empty else", "{{range .SEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"range []bool", "{{range .SB}}-{{.}}-{{end}}", "-true--false-", tVal, true},
	{"range []int method", "{{range .SI | .MAdd .I}}-{{.}}-{{end}}", "-20--21--22-", tVal, true},
	{"range map", "{{range .MSI | .MSort}}-{{.}}-{{end}}", "-one--three--two-", tVal, true},
	{"range empty map no else", "{{range .MSIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range map else", "{{range .MSI | .MSort}}-{{.}}-{{else}}EMPTY{{end}}", "-one--three--two-", tVal, true},
	{"range empty map else", "{{range .MSIEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"error method, error", "{{.EPERM true}}", "", tVal, false},
	{"error method, no error", "{{.EPERM false}}", "false", tVal, true},
}

func TestExecute(t *testing.T) {
	b := new(bytes.Buffer)
	for _, test := range execTests {
		tmpl := New(test.name)
		err := tmpl.Parse(test.input)
		if err != nil {
			t.Errorf("%s: parse error: %s", test.name, err)
			continue
		}
		b.Reset()
		err = tmpl.Execute(b, test.data)
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
