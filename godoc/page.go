// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

// Page describes the contents of the top-level godoc webpage.
type Page struct {
	Title    string
	Tabtitle string
	Subtitle string
	Query    string
	Body     []byte
	Share    bool

	// filled in by servePage
	SearchBox  bool
	Playground bool
	Version    string
}

func (p *Presentation) ServePage(w http.ResponseWriter, page Page) {
	if page.Tabtitle == "" {
		page.Tabtitle = page.Title
	}
	page.SearchBox = p.Corpus.IndexEnabled
	page.Playground = p.ShowPlayground
	page.Version = runtime.Version()
	applyTemplateToResponseWriter(w, p.GodocHTML, page)
}

func (p *Presentation) ServeError(w http.ResponseWriter, r *http.Request, relpath string, err error) {
	w.WriteHeader(http.StatusNotFound)
	if perr, ok := err.(*os.PathError); ok {
		rel, err := filepath.Rel(runtime.GOROOT(), perr.Path)
		if err != nil {
			perr.Path = "REDACTED"
		} else {
			perr.Path = filepath.Join("$GOROOT", rel)
		}
	}
	p.ServePage(w, Page{
		Title:    "File " + relpath,
		Subtitle: relpath,
		Body:     applyTemplate(p.ErrorHTML, "errorHTML", err),
		Share:    allowShare(r),
	})
}

var onAppengine = false // overriden in appengine.go when on app engine

func allowShare(r *http.Request) bool {
	if !onAppengine {
		return true
	}
	switch r.Header.Get("X-AppEngine-Country") {
	case "", "ZZ", "CN":
		return false
	}
	return true
}
