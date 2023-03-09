// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysisutil_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
)

func TestExtractDoc(t *testing.T) {
	const multi = `// Copyright

//+build tag

// Package foo
//
// # Irrelevant heading
//
// This is irrelevant doc.
//
// # Analyzer nocolon
//
// This one has the wrong form for this line.
//
// # Analyzer food
//
// food: reports dining opportunities
//
// This is the doc for analyzer 'food'.
//
// # Analyzer foo
//
// foo: reports diagnostics
//
// This is the doc for analyzer 'foo'.
//
// # Analyzer bar
//
// bar: reports drinking opportunities
//
// This is the doc for analyzer 'bar'.
package blah

var x = syntax error
`

	for _, test := range []struct {
		content, name string
		want          string // doc or "error: %w" string
	}{
		{"", "foo",
			"error: empty Go source file"},
		{"//foo", "foo",
			"error: not a Go source file"},
		{"//foo\npackage foo", "foo",
			"error: package doc comment contains no 'Analyzer foo' heading"},
		{multi, "foo",
			"reports diagnostics\n\nThis is the doc for analyzer 'foo'."},
		{multi, "bar",
			"reports drinking opportunities\n\nThis is the doc for analyzer 'bar'."},
		{multi, "food",
			"reports dining opportunities\n\nThis is the doc for analyzer 'food'."},
		{multi, "nope",
			"error: package doc comment contains no 'Analyzer nope' heading"},
		{multi, "nocolon",
			"error: 'Analyzer nocolon' heading not followed by 'nocolon: summary...' line"},
	} {
		got, err := analysisutil.ExtractDoc(test.content, test.name)
		if err != nil {
			got = "error: " + err.Error()
		}
		if test.want != got {
			t.Errorf("ExtractDoc(%q) returned <<%s>>, want <<%s>>, given input <<%s>>",
				test.name, got, test.want, test.content)
		}
	}
}
