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

	"golang.org/x/tools/godoc/vfs/mapfs"
)

func newCorpus(t *testing.T) *Corpus {
	c := NewCorpus(mapfs.New(map[string]string{
		"src/foo/foo.go": `// Package foo is an example.
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
		"src/bar/bar.go": `// Package bar is another example to test races.
package bar
`,
		"src/other/bar/bar.go": `// Package bar is another bar package.
package bar
func X() {}
`,
		"src/skip/skip.go": `// Package skip should be skipped.
package skip
func Skip() {}
`,
		"src/bar/readme.txt": `Whitelisted text file.
`,
		"src/bar/baz.zzz": `Text file not whitelisted.
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
	for _, docs := range []bool{true, false} {
		for _, goCode := range []bool{true, false} {
			for _, fullText := range []bool{true, false} {
				c := newCorpus(t)
				c.IndexDocs = docs
				c.IndexGoCode = goCode
				c.IndexFullText = fullText
				c.UpdateIndex()
				ix, _ := c.CurrentIndex()
				if ix == nil {
					t.Fatal("no index")
				}
				t.Logf("docs, goCode, fullText = %v,%v,%v", docs, goCode, fullText)
				testIndex(t, c, ix)
			}
		}
	}
}

func TestIndexWriteRead(t *testing.T) {
	type key struct {
		docs, goCode, fullText bool
	}
	type val struct {
		buf *bytes.Buffer
		c   *Corpus
	}
	m := map[key]val{}

	for _, docs := range []bool{true, false} {
		for _, goCode := range []bool{true, false} {
			for _, fullText := range []bool{true, false} {
				k := key{docs, goCode, fullText}
				c := newCorpus(t)
				c.IndexDocs = docs
				c.IndexGoCode = goCode
				c.IndexFullText = fullText
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
				m[k] = val{bytes.NewBuffer(buf.Bytes()), c}
				ix2 := new(Index)
				nr, err := ix2.ReadFrom(&buf)
				if err != nil {
					t.Fatalf("Index.ReadFrom: %v", err)
				}
				if nr != nw {
					t.Errorf("Wrote %d bytes to index but read %d", nw, nr)
				}
				testIndex(t, c, ix)
			}
		}
	}
	// Test CompatibleWith
	for k1, v1 := range m {
		ix := new(Index)
		if _, err := ix.ReadFrom(v1.buf); err != nil {
			t.Fatalf("Index.ReadFrom: %v", err)
		}
		for k2, v2 := range m {
			if got, want := ix.CompatibleWith(v2.c), k1 == k2; got != want {
				t.Errorf("CompatibleWith = %v; want %v for %v, %v", got, want, k1, k2)
			}
		}
	}
}

func testIndex(t *testing.T, c *Corpus, ix *Index) {
	if _, ok := ix.words["Skip"]; ok {
		t.Errorf("the word Skip was found; expected it to be skipped")
	}
	checkStats(t, c, ix)
	checkImportCount(t, c, ix)
	checkPackagePath(t, c, ix)
	checkExports(t, c, ix)
	checkIdents(t, c, ix)
}

// checkStats checks the Index's statistics.
// Some statistics are only set when we're indexing Go code.
func checkStats(t *testing.T, c *Corpus, ix *Index) {
	want := Statistics{}
	if c.IndexFullText {
		want.Bytes = 314
		want.Files = 4
		want.Lines = 21
	} else if c.IndexDocs || c.IndexGoCode {
		want.Bytes = 291
		want.Files = 3
		want.Lines = 20
	}
	if c.IndexGoCode {
		want.Words = 8
		want.Spots = 12
	}
	if got := ix.Stats(); !reflect.DeepEqual(got, want) {
		t.Errorf("Stats = %#v; want %#v", got, want)
	}
}

// checkImportCount checks the Index's import count map.
// It is only set when we're indexing Go code.
func checkImportCount(t *testing.T, c *Corpus, ix *Index) {
	want := map[string]int{}
	if c.IndexGoCode {
		want = map[string]int{
			"bar": 1,
		}
	}
	if got := ix.ImportCount(); !reflect.DeepEqual(got, want) {
		t.Errorf("ImportCount = %v; want %v", got, want)
	}
}

// checkPackagePath checks the Index's package path map.
// It is set if at least one of the indexing options is enabled.
func checkPackagePath(t *testing.T, c *Corpus, ix *Index) {
	want := map[string]map[string]bool{}
	if c.IndexDocs || c.IndexGoCode || c.IndexFullText {
		want = map[string]map[string]bool{
			"foo": {
				"foo": true,
			},
			"bar": {
				"bar":       true,
				"other/bar": true,
			},
		}
	}
	if got := ix.PackagePath(); !reflect.DeepEqual(got, want) {
		t.Errorf("PackagePath = %v; want %v", got, want)
	}
}

// checkExports checks the Index's exports map.
// It is only set when we're indexing Go code.
func checkExports(t *testing.T, c *Corpus, ix *Index) {
	want := map[string]map[string]SpotKind{}
	if c.IndexGoCode {
		want = map[string]map[string]SpotKind{
			"foo": {
				"Pi":   ConstDecl,
				"Foos": VarDecl,
				"Foo":  TypeDecl,
				"New":  FuncDecl,
			},
			"other/bar": {
				"X": FuncDecl,
			},
		}
	}
	if got := ix.Exports(); !reflect.DeepEqual(got, want) {
		t.Errorf("Exports = %v; want %v", got, want)
	}
}

// checkIdents checks the Index's indents map.
// It is only set when we're indexing documentation.
func checkIdents(t *testing.T, c *Corpus, ix *Index) {
	want := map[SpotKind]map[string][]Ident{}
	if c.IndexDocs {
		want = map[SpotKind]map[string][]Ident{
			PackageClause: {
				"bar": {
					{"bar", "bar", "bar", "Package bar is another example to test races."},
					{"other/bar", "bar", "bar", "Package bar is another bar package."},
				},
				"foo":   {{"foo", "foo", "foo", "Package foo is an example."}},
				"other": {{"other/bar", "bar", "bar", "Package bar is another bar package."}},
			},
			ConstDecl: {
				"Pi": {{"foo", "foo", "Pi", ""}},
			},
			VarDecl: {
				"Foos": {{"foo", "foo", "Foos", ""}},
			},
			TypeDecl: {
				"Foo": {{"foo", "foo", "Foo", "Foo is stuff."}},
			},
			FuncDecl: {
				"New": {{"foo", "foo", "New", ""}},
				"X":   {{"other/bar", "bar", "X", ""}},
			},
		}
	}
	if got := ix.Idents(); !reflect.DeepEqual(got, want) {
		t.Errorf("Idents = %v; want %v", got, want)
	}
}

func TestIdentResultSort(t *testing.T) {
	ic := map[string]int{
		"/a/b/pkg1": 10,
		"/a/b/pkg2": 2,
		"/b/d/pkg3": 20,
	}
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
				{"/b/d/pkg3", "pkg3", "MyFunc3", ""},
				{"/a/b/pkg1", "pkg1", "MyFunc1", ""},
				{"/a/b/pkg2", "pkg2", "MyFunc2", ""},
			},
		},
		{
			ir: []Ident{
				{"/a/a/pkg1", "pkg1", "MyFunc1", ""},
				{"/a/b/pkg1", "pkg1", "MyFunc1", ""},
			},
			exp: []Ident{
				{"/a/b/pkg1", "pkg1", "MyFunc1", ""},
				{"/a/a/pkg1", "pkg1", "MyFunc1", ""},
			},
		},
	} {
		if sort.Sort(byImportCount{tc.ir, ic}); !reflect.DeepEqual(tc.ir, tc.exp) {
			t.Errorf("got: %v, want %v", tc.ir, tc.exp)
		}
	}
}

func TestIdentFilter(t *testing.T) {
	ic := map[string]int{}
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
		res := byImportCount{tc.ir, ic}.filter(tc.pak)
		if !reflect.DeepEqual(res, tc.exp) {
			t.Errorf("got: %v, want %v", res, tc.exp)
		}
	}
}
