// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"html/template"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"golang.org/x/tools/present"
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
		err := renderDoc(w, name)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), 500)
		}
		return
	}
	if isDir, err := dirList(w, name); err != nil {
		addr, _, e := net.SplitHostPort(r.RemoteAddr)
		if e != nil {
			addr = r.RemoteAddr
		}
		log.Printf("request from %s: %s", addr, err)
		http.Error(w, err.Error(), 500)
		return
	} else if isDir {
		return
	}
	http.FileServer(http.Dir(base)).ServeHTTP(w, r)
}

func isDoc(path string) bool {
	_, ok := contentTemplate[filepath.Ext(path)]
	return ok
}

var (
	// dirListTemplate holds the front page template.
	dirListTemplate *template.Template

	// contentTemplate maps the presentable file extensions to the
	// template to be executed.
	contentTemplate map[string]*template.Template
)

func initTemplates(base string) error {
	// Locate the template file.
	actionTmpl := filepath.Join(base, "templates/action.tmpl")

	contentTemplate = make(map[string]*template.Template)

	for ext, contentTmpl := range map[string]string{
		".slide":   "slides.tmpl",
		".article": "article.tmpl",
	} {
		contentTmpl = filepath.Join(base, "templates", contentTmpl)

		// Read and parse the input.
		tmpl := present.Template()
		tmpl = tmpl.Funcs(template.FuncMap{"playable": playable})
		if _, err := tmpl.ParseFiles(actionTmpl, contentTmpl); err != nil {
			return err
		}
		contentTemplate[ext] = tmpl
	}

	var err error
	dirListTemplate, err = template.ParseFiles(filepath.Join(base, "templates/dir.tmpl"))
	return err
}

// renderDoc reads the present file, gets its template representation,
// and executes the template, sending output to w.
func renderDoc(w io.Writer, docFile string) error {
	// Read the input and build the doc structure.
	doc, err := parse(docFile, 0)
	if err != nil {
		return err
	}

	// Find which template should be executed.
	tmpl := contentTemplate[filepath.Ext(docFile)]

	// Execute the template.
	return doc.Render(w, tmpl)
}

func parse(name string, mode present.ParseMode) (*present.Doc, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return present.Parse(f, name, mode)
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
		// skip the golang.org directory
		if name == "." && fi.Name() == "golang.org" {
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
