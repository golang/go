// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"code.google.com/p/go.tools/present"
)

func init() {
	http.HandleFunc("/", dirHandler)
}

// dirHandler serves a directory listing for the requested path, rooted at basePath.
func dirHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/favicon.ico" {
		http.Error(w, "not found", 404)
		return
	}
	const base = "."
	name := filepath.Join(base, r.URL.Path)
	if isDoc(name) {
		err := renderDoc(w, basePath, name)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), 500)
		}
		return
	}
	if isDir, err := dirList(w, name); err != nil {
		log.Println(err)
		http.Error(w, err.Error(), 500)
		return
	} else if isDir {
		return
	}
	http.FileServer(http.Dir(base)).ServeHTTP(w, r)
}

// extensions maps the presentable file extensions to the name of the
// template to be executed.
var extensions = map[string]string{
	".slide":   "slides.tmpl",
	".article": "article.tmpl",
}

func isDoc(path string) bool {
	_, ok := extensions[filepath.Ext(path)]
	return ok
}

// renderDoc reads the present file, builds its template representation,
// and executes the template, sending output to w.
func renderDoc(w io.Writer, base, docFile string) error {
	// Read the input and build the doc structure.
	doc, err := parse(docFile, 0)
	if err != nil {
		return err
	}

	// Find which template should be executed.
	ext := filepath.Ext(docFile)
	contentTmpl, ok := extensions[ext]
	if !ok {
		return fmt.Errorf("no template for extension %v", ext)
	}

	// Locate the template file.
	actionTmpl := filepath.Join(base, "templates/action.tmpl")
	contentTmpl = filepath.Join(base, "templates", contentTmpl)

	// Read and parse the input.
	tmpl := present.Template()
	tmpl = tmpl.Funcs(template.FuncMap{"playable": playable})
	if _, err := tmpl.ParseFiles(actionTmpl, contentTmpl); err != nil {
		return err
	}

	// Execute the template.
	return doc.Render(w, tmpl)
}

func parse(name string, mode present.ParseMode) (*present.Doc, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return present.Parse(f, name, 0)
}

// dirList scans the given path and writes a directory listing to w.
// It parses the first part of each .slide file it encounters to display the
// presentation title in the listing.
// If the given path is not a directory, it returns (isDir == false, err == nil)
// and writes nothing to w.
func dirList(w io.Writer, name string) (isDir bool, err error) {
	f, err := os.Open(name)
	if err != nil {
		return false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return false, err
	}
	if isDir = fi.IsDir(); !isDir {
		return false, nil
	}
	fis, err := f.Readdir(0)
	if err != nil {
		return false, err
	}
	d := &dirListData{Path: name}
	for _, fi := range fis {
		// skip the pkg directory
		if name == "." && fi.Name() == "pkg" {
			continue
		}
		e := dirEntry{
			Name: fi.Name(),
			Path: filepath.ToSlash(filepath.Join(name, fi.Name())),
		}
		if fi.IsDir() && showDir(e.Name) {
			d.Dirs = append(d.Dirs, e)
			continue
		}
		if isDoc(e.Name) {
			if p, err := parse(e.Path, present.TitlesOnly); err != nil {
				log.Println(err)
			} else {
				e.Title = p.Title
			}
			switch filepath.Ext(e.Path) {
			case ".article":
				d.Articles = append(d.Articles, e)
			case ".slide":
				d.Slides = append(d.Slides, e)
			}
		} else if showFile(e.Name) {
			d.Other = append(d.Other, e)
		}
	}
	if d.Path == "." {
		d.Path = ""
	}
	sort.Sort(d.Dirs)
	sort.Sort(d.Slides)
	sort.Sort(d.Articles)
	sort.Sort(d.Other)
	return true, dirListTemplate.Execute(w, d)
}

// showFile reports whether the given file should be displayed in the list.
func showFile(n string) bool {
	switch filepath.Ext(n) {
	case ".pdf":
	case ".html":
	case ".go":
	default:
		return isDoc(n)
	}
	return true
}

// showDir reports whether the given directory should be displayed in the list.
func showDir(n string) bool {
	if len(n) > 0 && (n[0] == '.' || n[0] == '_') || n == "present" {
		return false
	}
	return true
}

type dirListData struct {
	Path                          string
	Dirs, Slides, Articles, Other dirEntrySlice
}

type dirEntry struct {
	Name, Path, Title string
}

type dirEntrySlice []dirEntry

func (s dirEntrySlice) Len() int           { return len(s) }
func (s dirEntrySlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s dirEntrySlice) Less(i, j int) bool { return s[i].Name < s[j].Name }

var dirListTemplate = template.Must(template.New("").Parse(dirListHTML))

const dirListHTML = `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <title>Talks - The Go Programming Language</title>
  <link type="text/css" rel="stylesheet" href="/static/dir.css">
  <script src="/static/dir.js"></script>
</head>
<body>

<div id="topbar"><div class="container">

<form method="GET" action="http://golang.org/search">
<div id="menu">
<a href="http://golang.org/doc/">Documents</a>
<a href="http://golang.org/ref/">References</a>
<a href="http://golang.org/pkg/">Packages</a>
<a href="http://golang.org/project/">The Project</a>
<a href="http://golang.org/help/">Help</a>
<input type="text" id="search" name="q" class="inactive" value="Search">
</div>
<div id="heading"><a href="/">The Go Programming Language</a></div>
</form>

</div></div>

<div id="page">

  <h1>Go talks</h1>

  {{with .Path}}<h2>{{.}}</h2>{{end}}

  {{with .Articles}}
  <h4>Articles:</h4>
  <dl>
  {{range .}}
  <dd><a href="/{{.Path}}">{{.Name}}</a>: {{.Title}}</dd>
  {{end}}
  </dl>
  {{end}}

  {{with .Slides}}
  <h4>Slide decks:</h4>
  <dl>
  {{range .}}
  <dd><a href="/{{.Path}}">{{.Name}}</a>: {{.Title}}</dd>
  {{end}}
  </dl>
  {{end}}

  {{with .Other}}
  <h4>Files:</h4>
  <dl>
  {{range .}}
  <dd><a href="/{{.Path}}">{{.Name}}</a></dd>
  {{end}}
  </dl>
  {{end}}

  {{with .Dirs}}
  <h4>Sub-directories:</h4>
  <dl>
  {{range .}}
  <dd><a href="/{{.Path}}">{{.Name}}</a></dd>
  {{end}}
  </dl>
  {{end}}

</div>

<div id="footer">
Except as <a href="https://developers.google.com/site-policies#restrictions">noted</a>,
the content of this page is licensed under the
Creative Commons Attribution 3.0 License,
and code is licensed under a <a href="http://golang.org/LICENSE">BSD license</a>.<br>
<a href="http://golang.org/doc/tos.html">Terms of Service</a> | 
<a href="http://www.google.com/intl/en/policies/privacy/">Privacy Policy</a>
</div>

</body>
</html>`
