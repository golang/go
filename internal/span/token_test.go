// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"fmt"
	"go/token"
	"testing"

	"golang.org/x/tools/internal/span"
)

var testdata = []struct {
	uri     string
	content []byte
}{
	{"/a.go", []byte(`
// file a.go
package test
`)},
	{"/b.go", []byte(`
//
//
// file b.go
package test
`)},
}

var tokenTests = []span.Span{
	{span.FileURI("/a.go"), span.Point{}, span.Point{}},
	{span.FileURI("/a.go"), span.Point{3, 7, 20}, span.Point{3, 7, 20}},
	{span.FileURI("/b.go"), span.Point{4, 9, 15}, span.Point{4, 13, 19}},
}

func TestToken(t *testing.T) {
	fset := token.NewFileSet()
	files := map[span.URI]*token.File{}
	for _, f := range testdata {
		file := fset.AddFile(f.uri, -1, len(f.content))
		file.SetLinesForContent(f.content)
		files[span.FileURI(f.uri)] = file
	}
	for _, test := range tokenTests {
		f := files[test.URI]
		c := span.NewTokenConverter(fset, f)
		checkToken(t, c, span.Span{
			URI:   test.URI,
			Start: span.Point{Line: test.Start.Line, Column: test.Start.Column},
			End:   span.Point{Line: test.End.Line, Column: test.End.Column},
		}, test)
		checkToken(t, c, span.Span{
			URI:   test.URI,
			Start: span.Point{Offset: test.Start.Offset},
			End:   span.Point{Offset: test.End.Offset},
		}, test)
	}
}

func checkToken(t *testing.T, c *span.TokenConverter, in, expect span.Span) {
	rng := in.Range(c)
	gotLoc := rng.Span()
	expected := fmt.Sprintf("%+v", expect)
	got := fmt.Sprintf("%+v", gotLoc)
	if expected != got {
		t.Errorf("Expected %q got %q", expected, got)
	}
}
