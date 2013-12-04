// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"net/http"
	"regexp"
	"sync"
	"text/template"

	"code.google.com/p/go.tools/godoc/vfs/httpfs"
)

// Presentation generates output from a corpus.
type Presentation struct {
	Corpus *Corpus

	mux        *http.ServeMux
	fileServer http.Handler
	cmdHandler handlerServer
	pkgHandler handlerServer

	DirlistHTML,
	ErrorHTML,
	ExampleHTML,
	GodocHTML,
	PackageHTML,
	PackageText,
	SearchHTML,
	SearchText,
	SearchDescXML *template.Template

	// TabWidth optionally specifies the tab width.
	TabWidth int

	ShowTimestamps bool
	ShowPlayground bool
	ShowExamples   bool
	DeclLinks      bool

	// SrcMode outputs source code instead of documentation in command-line mode.
	SrcMode bool
	// HTMLMode outputs HTML instead of plain text in command-line mode.
	HTMLMode bool

	// NotesRx optionally specifies a regexp to match
	// notes to render in the output.
	NotesRx *regexp.Regexp

	// AdjustPageInfoMode optionally specifies a function to
	// modify the PageInfoMode of a request. The default chosen
	// value is provided.
	AdjustPageInfoMode func(req *http.Request, mode PageInfoMode) PageInfoMode

	// URLForSrc optionally specifies a function that takes a source file and
	// returns a URL for it.
	// The source file argument has the form /src/pkg/<path>/<filename>.
	URLForSrc func(src string) string

	// URLForSrcPos optionally specifies a function to create a URL given a
	// source file, a line from the source file (1-based), and low & high offset
	// positions (0-based, bytes from beginning of file). Ideally, the returned
	// URL will be for the specified line of the file, while the high & low
	// positions will be used to highlight a section of the file.
	// The source file argument has the form /src/pkg/<path>/<filename>.
	URLForSrcPos func(src string, line, low, high int) string

	// URLForSrcQuery optionally specifies a function to create a URL given a
	// source file, a query string, and a line from the source file (1-based).
	// The source file argument has the form /src/pkg/<path>/<filename>.
	// The query argument will be escaped for the purposes of embedding in a URL
	// query parameter.
	// Ideally, the returned URL will be for the specified line of the file with
	// the query string highlighted.
	URLForSrcQuery func(src, query string, line int) string

	initFuncMapOnce sync.Once
	funcMap         template.FuncMap
	templateFuncs   template.FuncMap
}

// NewPresentation returns a new Presentation from a corpus.
func NewPresentation(c *Corpus) *Presentation {
	if c == nil {
		panic("nil Corpus")
	}
	p := &Presentation{
		Corpus:     c,
		mux:        http.NewServeMux(),
		fileServer: http.FileServer(httpfs.New(c.fs)),

		TabWidth:     4,
		ShowExamples: true,
		DeclLinks:    true,
	}
	p.cmdHandler = handlerServer{p, c, "/cmd/", "/src/cmd"}
	p.pkgHandler = handlerServer{p, c, "/pkg/", "/src/pkg"}
	p.cmdHandler.registerWithMux(p.mux)
	p.pkgHandler.registerWithMux(p.mux)
	p.mux.HandleFunc("/", p.ServeFile)
	p.mux.HandleFunc("/search", p.HandleSearch)
	p.mux.HandleFunc("/opensearch.xml", p.serveSearchDesc)
	return p
}

func (p *Presentation) FileServer() http.Handler {
	return p.fileServer
}

func (p *Presentation) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p.mux.ServeHTTP(w, r)
}

func (p *Presentation) PkgFSRoot() string {
	return p.pkgHandler.fsRoot
}

func (p *Presentation) CmdFSRoot() string {
	return p.cmdHandler.fsRoot
}

// TODO(bradfitz): move this to be a method on Corpus. Just moving code around for now,
// but this doesn't feel right.
func (p *Presentation) GetPkgPageInfo(abspath, relpath string, mode PageInfoMode) *PageInfo {
	return p.pkgHandler.GetPageInfo(abspath, relpath, mode)
}

// TODO(bradfitz): move this to be a method on Corpus. Just moving code around for now,
// but this doesn't feel right.
func (p *Presentation) GetCmdPageInfo(abspath, relpath string, mode PageInfoMode) *PageInfo {
	return p.cmdHandler.GetPageInfo(abspath, relpath, mode)
}
