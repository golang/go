// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"text/template"

	"golang.org/x/tools/godoc/vfs/mapfs"
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
	r := &http.Request{URL: &url.URL{}}

	// Test that redirect is sent back correctly.
	// Used to panic. See golang.org/issue/40665.
	for _, elem := range []string{"x", "y"} {
		dir := "/doc/" + elem + "/"
		r.URL.Path = dir + "index.html"
		rw := httptest.NewRecorder()
		p.ServeFile(rw, r)
		loc := rw.Result().Header.Get("Location")
		if rw.Code != 301 || loc != dir {
			t.Errorf("GET %s: expected 301 -> %q, got %d -> %q", r.URL.Path, dir, rw.Code, loc)
		}

		r.URL.Path = dir
		rw = httptest.NewRecorder()
		p.ServeFile(rw, r)
		if rw.Code != 200 || !strings.Contains(rw.Body.String(), "Hello, "+elem) {
			t.Fatalf("GET %s: expected 200 w/ Hello, %s: got %d w/ body:\n%s",
				r.URL.Path, elem, rw.Code, rw.Body)
		}
	}
}
