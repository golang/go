// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The /doc/codewalk/ tree is synthesized from codewalk descriptions,
// files named $GOROOT/doc/codewalk/*.xml.
// For an example and a description of the format, see
// http://golang.org/doc/codewalk/codewalk or run godoc -http=:6060
// and see http://localhost:6060/doc/codewalk/codewalk .
// That page is itself a codewalk; the source code for it is
// $GOROOT/doc/codewalk/codewalk.xml.

package main

import (
	"log"
	"net/http"
	"text/template"

	"code.google.com/p/go.tools/godoc"
	"code.google.com/p/go.tools/godoc/redirect"
	"code.google.com/p/go.tools/godoc/vfs"
)

var (
	pres *godoc.Presentation
	fs   = vfs.NameSpace{}
)

func registerHandlers(pres *godoc.Presentation) {
	if pres == nil {
		panic("nil Presentation")
	}
	http.HandleFunc("/doc/codewalk/", codewalk)
	http.Handle("/doc/play/", pres.FileServer())
	http.Handle("/robots.txt", pres.FileServer())
	http.Handle("/", pres)
	redirect.Register(nil)
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

func readTemplates(p *godoc.Presentation, html bool) {
	p.PackageText = readTemplate("package.txt")
	p.SearchText = readTemplate("search.txt")

	if html || p.HTMLMode {
		codewalkHTML = readTemplate("codewalk.html")
		codewalkdirHTML = readTemplate("codewalkdir.html")
		p.DirlistHTML = readTemplate("dirlist.html")
		p.ErrorHTML = readTemplate("error.html")
		p.ExampleHTML = readTemplate("example.html")
		p.GodocHTML = readTemplate("godoc.html")
		p.PackageHTML = readTemplate("package.html")
		p.SearchHTML = readTemplate("search.html")
		p.SearchDescXML = readTemplate("opensearch.xml")
	}
}
