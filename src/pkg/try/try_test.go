// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package try

import (
	"bytes"
	"regexp" // Used as the package to try.
	"testing"
)

// The global functions in package regexp at time of writing.
// Doesn't need to be updated unless the entries in this list become invalid.
var functions = map[string]interface{}{
	"Compile":     regexp.Compile,
	"Match":       regexp.Match,
	"MatchString": regexp.MatchString,
	"MustCompile": regexp.MustCompile,
	"QuoteMeta":   regexp.QuoteMeta,
}

// A wraps arguments to make the test cases nicer to read.
func A(args ...interface{}) []interface{} {
	return args
}

type Test struct {
	firstArg string // only needed if there is exactly one argument
	result   string // minus final newline; might be just the godoc string
	args     []interface{}
}

var testRE = regexp.MustCompile("a(.)(.)d")

var tests = []Test{
	// A simple expression.  The final value is a slice in case the expression is multivalue.
	{"3+4", "3+4 = 7", A([]interface{}{7})},
	// A search for a function.
	{"", "regexp QuoteMeta", A("([])", `\(\[\]\)`)},
	// A search for a function with multiple return values.
	{"", "regexp MatchString", A("abc", "xabcd", true, nil)},
	// Searches for methods.
	{"", "regexp MatchString", A(testRE, "xabcde", true)},
	{"", "regexp NumSubexp", A(testRE, 2)},
}

func TestAll(t *testing.T) {
	re := regexp.MustCompile(".*// godoc ")
	for _, test := range tests {
		b := new(bytes.Buffer)
		output = b
		Main("regexp", test.firstArg, functions, test.args)
		expect := test.result + "\n"
		got := re.ReplaceAllString(b.String(), "")
		if got != expect {
			t.Errorf("expected %q; got %q", expect, got)
		}
	}
}
