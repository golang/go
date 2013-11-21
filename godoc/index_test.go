// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"bytes"
	"reflect"
	"sort"
	"strings"
	"testing"

	"code.google.com/p/go.tools/godoc/vfs/mapfs"
)

func newCorpus(t *testing.T) *Corpus {
	c := NewCorpus(mapfs.New(map[string]string{
		"src/pkg/foo/foo.go": `// Package foo is an example.
package foo

import "bar"

const Pi = 3.1415

var Foos []Foo

// Foo is stuff.
type Foo struct{}

func New() *Foo {
   return new(Foo)
}
`,
		"src/pkg/bar/bar.go": `// Package bar is another example to test races.
package bar
`,
		"src/pkg/other/bar/bar.go": `// Package bar is another bar package.
package bar
func X() {}
`,
		"src/pkg/skip/skip.go": `// Package skip should be skipped.
package skip
func Skip() {}
`,
	}))
	c.IndexEnabled = true
	c.IndexDirectory = func(dir string) bool {
		return !strings.Contains(dir, "skip")
	}

	if err := c.Init(); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestIndex(t *testing.T) {
	c := newCorpus(t)
	c.UpdateIndex()
	ix, _ := c.CurrentIndex()
	if ix == nil {
		t.Fatal("no index")
	}
	t.Logf("Got: %#v", ix)
	testIndex(t, ix)
}

func TestIndexWriteRead(t *testing.T) {
	c := newCorpus(t)
	c.UpdateIndex()
	ix, _ := c.CurrentIndex()
	if ix == nil {
		t.Fatal("no index")
	}

	var buf bytes.Buffer
	nw, err := ix.WriteTo(&buf)
	if err != nil {
		t.Fatalf("Index.WriteTo: %v", err)
	}

	ix2 := new(Index)
	nr, err := ix2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("Index.ReadFrom: %v", err)
	}
	if nr != nw {
		t.Errorf("Wrote %d bytes to index but read %d", nw, nr)
	}
	testIndex(t, ix2)
}

func testIndex(t *testing.T, ix *Index) {
	wantStats := Statistics{Bytes: 291, Files: 3, Lines: 20, Words: 8, Spots: 12}
	if !reflect.DeepEqual(ix.Stats(), wantStats) {
		t.Errorf("Stats = %#v; want %#v", ix.Stats(), wantStats)
	}

	if _, ok := ix.words["Skip"]; ok {
		t.Errorf("the word Skip was found; expected it to be skipped")
	}

	if got, want := ix.ImportCount(), map[string]int{
		"bar": 1,
	}; !reflect.DeepEqual(got, want) {
		t.Errorf("ImportCount = %v; want %v", got, want)
	}

	if got, want := ix.PackagePath(), map[string]map[string]bool{
		"foo": map[string]bool{
			"foo": true,
		},
		"bar": map[string]bool{
			"bar":       true,
			"other/bar": true,
		},
	}; !reflect.DeepEqual(got, want) {
		t.Errorf("PackagePath = %v; want %v", got, want)
	}

	if got, want := ix.Exports(), map[string]map[string]SpotKind{
		"foo": map[string]SpotKind{
			"Pi":   ConstDecl,
			"Foos": VarDecl,
			"Foo":  TypeDecl,
			"New":  FuncDecl,
		},
		"other/bar": map[string]SpotKind{
			"X": FuncDecl,
		},
	}; !reflect.DeepEqual(got, want) {
		t.Errorf("Exports = %v; want %v", got, want)
	}

	if got, want := ix.Idents(), map[SpotKind]map[string][]Ident{
		ConstDecl: map[string][]Ident{
			"Pi": []Ident{{"/src/pkg/foo", "foo", "Pi", ""}},
		},
		VarDecl: map[string][]Ident{
			"Foos": []Ident{{"/src/pkg/foo", "foo", "Foos", ""}},
		},
		TypeDecl: map[string][]Ident{
			"Foo": []Ident{{"/src/pkg/foo", "foo", "Foo", "Foo is stuff."}},
		},
		FuncDecl: map[string][]Ident{
			"New": []Ident{{"/src/pkg/foo", "foo", "New", ""}},
			"X":   []Ident{{"/src/pkg/other/bar", "bar", "X", ""}},
		},
	}; !reflect.DeepEqual(got, want) {
		t.Errorf("Idents = %v; want %v", got, want)
	}
}

func TestIdentResultSort(t *testing.T) {
	for _, tc := range []struct {
		ir  []Ident
		exp []Ident
	}{
		{
			ir: []Ident{
				{"/a/b/pkg2", "pkg2", "MyFunc2", ""},
				{"/b/d/pkg3", "pkg3", "MyFunc3", ""},
				{"/a/b/pkg1", "pkg1", "MyFunc1", ""},
			},
			exp: []Ident{
				{"/a/b/pkg1", "pkg1", "MyFunc1", ""},
				{"/a/b/pkg2", "pkg2", "MyFunc2", ""},
				{"/b/d/pkg3", "pkg3", "MyFunc3", ""},
			},
		},
	} {
		if sort.Sort(byPackage(tc.ir)); !reflect.DeepEqual(tc.ir, tc.exp) {
			t.Errorf("got: %v, want %v", tc.ir, tc.exp)
		}
	}
}

func TestIdentPackageFilter(t *testing.T) {
	for _, tc := range []struct {
		ir  []Ident
		pak string
		exp []Ident
	}{
		{
			ir: []Ident{
				{"/a/b/pkg2", "pkg2", "MyFunc2", ""},
				{"/b/d/pkg3", "pkg3", "MyFunc3", ""},
				{"/a/b/pkg1", "pkg1", "MyFunc1", ""},
			},
			pak: "pkg2",
			exp: []Ident{
				{"/a/b/pkg2", "pkg2", "MyFunc2", ""},
			},
		},
	} {
		if res := byPackage(tc.ir).filter(tc.pak); !reflect.DeepEqual(res, tc.exp) {
			t.Errorf("got: %v, want %v", res, tc.exp)
		}
	}
}
