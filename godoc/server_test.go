// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"go/doc"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sort"
	"strings"
	"testing"
	"text/template"

	"golang.org/x/tools/godoc/vfs/mapfs"
	"golang.org/x/tools/internal/typeparams"
)

// TestIgnoredGoFiles tests the scenario where a folder has no .go or .c files,
// but has an ignored go file.
func TestIgnoredGoFiles(t *testing.T) {
	packagePath := "github.com/package"
	packageComment := "main is documented in an ignored .go file"

	c := NewCorpus(mapfs.New(map[string]string{
		"src/" + packagePath + "/ignored.go": `// +build ignore

// ` + packageComment + `
package main`}))
	srv := &handlerServer{
		p: &Presentation{
			Corpus: c,
		},
		c: c,
	}
	pInfo := srv.GetPageInfo("/src/"+packagePath, packagePath, NoFiltering, "linux", "amd64")

	if pInfo.PDoc == nil {
		t.Error("pInfo.PDoc = nil; want non-nil.")
	} else {
		if got, want := pInfo.PDoc.Doc, packageComment+"\n"; got != want {
			t.Errorf("pInfo.PDoc.Doc = %q; want %q.", got, want)
		}
		if got, want := pInfo.PDoc.Name, "main"; got != want {
			t.Errorf("pInfo.PDoc.Name = %q; want %q.", got, want)
		}
		if got, want := pInfo.PDoc.ImportPath, packagePath; got != want {
			t.Errorf("pInfo.PDoc.ImportPath = %q; want %q.", got, want)
		}
	}
	if pInfo.FSet == nil {
		t.Error("pInfo.FSet = nil; want non-nil.")
	}
}

func TestIssue5247(t *testing.T) {
	const packagePath = "example.com/p"
	c := NewCorpus(mapfs.New(map[string]string{
		"src/" + packagePath + "/p.go": `package p

//line notgen.go:3
// F doc //line 1 should appear
// line 2 should appear
func F()
//line foo.go:100`})) // No newline at end to check corner cases.

	srv := &handlerServer{
		p: &Presentation{Corpus: c},
		c: c,
	}
	pInfo := srv.GetPageInfo("/src/"+packagePath, packagePath, 0, "linux", "amd64")
	if got, want := pInfo.PDoc.Funcs[0].Doc, "F doc //line 1 should appear\nline 2 should appear\n"; got != want {
		t.Errorf("pInfo.PDoc.Funcs[0].Doc = %q; want %q", got, want)
	}
}

func testServeBody(t *testing.T, p *Presentation, path, body string) {
	t.Helper()
	r := &http.Request{URL: &url.URL{Path: path}}
	rw := httptest.NewRecorder()
	p.ServeFile(rw, r)
	if rw.Code != 200 || !strings.Contains(rw.Body.String(), body) {
		t.Fatalf("GET %s: expected 200 w/ %q: got %d w/ body:\n%s",
			path, body, rw.Code, rw.Body)
	}
}

func TestRedirectAndMetadata(t *testing.T) {
	c := NewCorpus(mapfs.New(map[string]string{
		"doc/y/index.html": "Hello, y.",
		"doc/x/index.html": `<!--{
		"Path": "/doc/x/"
}-->

Hello, x.
`}))
	c.updateMetadata()
	p := &Presentation{
		Corpus:    c,
		GodocHTML: template.Must(template.New("").Parse(`{{printf "%s" .Body}}`)),
	}

	// Test that redirect is sent back correctly.
	// Used to panic. See golang.org/issue/40665.
	for _, elem := range []string{"x", "y"} {
		dir := "/doc/" + elem + "/"

		r := &http.Request{URL: &url.URL{Path: dir + "index.html"}}
		rw := httptest.NewRecorder()
		p.ServeFile(rw, r)
		loc := rw.Result().Header.Get("Location")
		if rw.Code != 301 || loc != dir {
			t.Errorf("GET %s: expected 301 -> %q, got %d -> %q", r.URL.Path, dir, rw.Code, loc)
		}

		testServeBody(t, p, dir, "Hello, "+elem)
	}
}

func TestMarkdown(t *testing.T) {
	p := &Presentation{
		Corpus: NewCorpus(mapfs.New(map[string]string{
			"doc/test.md":  "**bold**",
			"doc/test2.md": `{{"*template*"}}`,
		})),
		GodocHTML: template.Must(template.New("").Parse(`{{printf "%s" .Body}}`)),
	}

	testServeBody(t, p, "/doc/test.html", "<strong>bold</strong>")
	testServeBody(t, p, "/doc/test2.html", "<em>template</em>")
}

func TestGenerics(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("type params are not enabled at this Go version")
	}

	c := NewCorpus(mapfs.New(map[string]string{
		"blah/blah.go": `package blah

var A AStruct[int]

type AStruct[T any] struct {
	A string
	X T
}

func (a *AStruct[T]) Method() T {
	return a.X
}

func (a AStruct[T]) NonPointerMethod() T {
	return a.X
}

func NewAStruct[T any](arg T) *AStruct[T] {
	return &AStruct[T]{ X: arg }
}

type NonGenericStruct struct {
	B int
}

func (b *NonGenericStruct) NonGenericMethod() int {
	return b.B
}

func NewNonGenericStruct(arg int) *NonGenericStruct {
	return &NonGenericStruct{arg}
}

type Pair[K, V any] struct {
	K K
	V V
}

func (p Pair[K, V]) Apply(kf func(K) K, vf func(V) V) Pair[K, V] {
	return &Pair{ K: kf(p.K), V: vf(p.V) }
}

func (p *Pair[K, V]) Set(k K, v V) {
	p.K = k
	p.V = v
}

func NewPair[K, V any](k K, v V) Pair[K, V] {
	return Pair[K, V]{ k, v }
}
`}))

	srv := &handlerServer{
		p: &Presentation{
			Corpus: c,
		},
		c: c,
	}
	pInfo := srv.GetPageInfo("/blah/", "", NoFiltering, "linux", "amd64")
	t.Logf("%v\n", pInfo)

	findType := func(name string) *doc.Type {
		for _, typ := range pInfo.PDoc.Types {
			if typ.Name == name {
				return typ
			}
		}
		return nil
	}

	assertFuncs := func(typ *doc.Type, typFuncs []*doc.Func, funcs ...string) {
		typfuncs := make([]string, len(typFuncs))
		for i := range typFuncs {
			typfuncs[i] = typFuncs[i].Name
		}
		sort.Strings(typfuncs)
		sort.Strings(funcs)
		if len(typfuncs) != len(funcs) {
			t.Errorf("function mismatch for type %q, got: %q, want: %q", typ.Name, typfuncs, funcs)
			return
		}
		for i := range funcs {
			if funcs[i] != typfuncs[i] {
				t.Errorf("function mismatch for type %q: got: %q, want: %q", typ.Name, typfuncs, funcs)
				return
			}
		}
	}

	aStructType := findType("AStruct")
	assertFuncs(aStructType, aStructType.Funcs, "NewAStruct")
	assertFuncs(aStructType, aStructType.Methods, "Method", "NonPointerMethod")

	nonGenericStructType := findType("NonGenericStruct")
	assertFuncs(nonGenericStructType, nonGenericStructType.Funcs, "NewNonGenericStruct")
	assertFuncs(nonGenericStructType, nonGenericStructType.Methods, "NonGenericMethod")

	pairType := findType("Pair")
	assertFuncs(pairType, pairType.Funcs, "NewPair")
	assertFuncs(pairType, pairType.Methods, "Apply", "Set")

	if len(pInfo.PDoc.Funcs) > 0 {
		t.Errorf("unexpected functions in package documentation")
	}
}
