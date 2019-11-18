// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"go/format"
	"log"
	"net/http"
	"text/template"

	"golang.org/x/tools/godoc"
	"golang.org/x/tools/godoc/redirect"
	"golang.org/x/tools/godoc/vfs"
)

// This package registers "/compile" and "/share" handlers
// that redirect to the golang.org playground.
import _ "golang.org/x/tools/playground"

var (
	pres *godoc.Presentation
	fs   = vfs.NameSpace{}
)

func registerHandlers(pres *godoc.Presentation) {
	if pres == nil {
		panic("nil Presentation")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/" {
			http.Redirect(w, req, "/pkg/", http.StatusFound)
			return
		}
		pres.ServeHTTP(w, req)
	})
	mux.Handle("/pkg/C/", redirect.Handler("/cmd/cgo/"))
	mux.HandleFunc("/fmt", fmtHandler)
	redirect.Register(mux)

	http.Handle("/", mux)
}

func readTemplate(name string) *template.Template {
	if pres == nil {
		panic("no global Presentation set yet")
	}
	path := "lib/godoc/" + name

	// use underlying file system fs to read the template file
	// (cannot use template ParseFile functions directly)
	data, err := vfs.ReadFile(fs, path)
	if err != nil {
		log.Fatal("readTemplate: ", err)
	}
	// be explicit with errors (for app engine use)
	t, err := template.New(name).Funcs(pres.FuncMap()).Parse(string(data))
	if err != nil {
		log.Fatal("readTemplate: ", err)
	}
	return t
}

func readTemplates(p *godoc.Presentation) {
	p.CallGraphHTML = readTemplate("callgraph.html")
	p.DirlistHTML = readTemplate("dirlist.html")
	p.ErrorHTML = readTemplate("error.html")
	p.ExampleHTML = readTemplate("example.html")
	p.GodocHTML = readTemplate("godoc.html")
	p.ImplementsHTML = readTemplate("implements.html")
	p.MethodSetHTML = readTemplate("methodset.html")
	p.PackageHTML = readTemplate("package.html")
	p.PackageRootHTML = readTemplate("packageroot.html")
	p.SearchHTML = readTemplate("search.html")
	p.SearchDocHTML = readTemplate("searchdoc.html")
	p.SearchCodeHTML = readTemplate("searchcode.html")
	p.SearchTxtHTML = readTemplate("searchtxt.html")
}

type fmtResponse struct {
	Body  string
	Error string
}

// fmtHandler takes a Go program in its "body" form value, formats it with
// standard gofmt formatting, and writes a fmtResponse as a JSON object.
func fmtHandler(w http.ResponseWriter, r *http.Request) {
	resp := new(fmtResponse)
	body, err := format.Source([]byte(r.FormValue("body")))
	if err != nil {
		resp.Error = err.Error()
	} else {
		resp.Body = string(body)
	}
	w.Header().Set("Content-type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(resp)
}
