// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"fmt"
	"go/token"
	"path"
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
package test`)},
	{"/c.go", []byte(`
// file c.go
package test`)},
}

var tokenTests = []span.Span{
	span.New(span.URIFromPath("/a.go"), span.NewPoint(1, 1, 0), span.Point{}),
	span.New(span.URIFromPath("/a.go"), span.NewPoint(3, 7, 20), span.NewPoint(3, 7, 20)),
	span.New(span.URIFromPath("/b.go"), span.NewPoint(4, 9, 15), span.NewPoint(4, 13, 19)),
	span.New(span.URIFromPath("/c.go"), span.NewPoint(4, 1, 26), span.Point{}),
}

func TestToken(t *testing.T) {
	fset := token.NewFileSet()
	files := map[span.URI]*token.File{}
	for _, f := range testdata {
		file := fset.AddFile(f.uri, -1, len(f.content))
		file.SetLinesForContent(f.content)
		files[span.URIFromPath(f.uri)] = file
	}
	for _, test := range tokenTests {
		f := files[test.URI()]
		c := span.NewTokenConverter(fset, f)
		t.Run(path.Base(f.Name()), func(t *testing.T) {
			checkToken(t, c, span.New(
				test.URI(),
				span.NewPoint(test.Start().Line(), test.Start().Column(), 0),
				span.NewPoint(test.End().Line(), test.End().Column(), 0),
			), test)
			checkToken(t, c, span.New(
				test.URI(),
				span.NewPoint(0, 0, test.Start().Offset()),
				span.NewPoint(0, 0, test.End().Offset()),
			), test)
		})
	}
}

func checkToken(t *testing.T, c *span.TokenConverter, in, expect span.Span) {
	rng, err := in.Range(c)
	if err != nil {
		t.Error(err)
	}
	gotLoc, err := rng.Span()
	if err != nil {
		t.Error(err)
	}
	expected := fmt.Sprintf("%+v", expect)
	got := fmt.Sprintf("%+v", gotLoc)
	if expected != got {
		t.Errorf("For %v expected %q got %q", in, expected, got)
	}
}
