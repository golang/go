// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	htmlpkg "html"
	"log"
	"net/http"
	"net/url"
	pathpkg "path"
	"regexp"
	"runtime"
	"strings"
	"text/template"

	"code.google.com/p/go.tools/godoc"
	"code.google.com/p/go.tools/godoc/util"
	"code.google.com/p/go.tools/godoc/vfs"
)

// ----------------------------------------------------------------------------
// Globals

func flagBool(b *bool, name string, value bool, usage string) interface{} {
	flag.BoolVar(b, name, value, usage)
	return nil
}

func flagInt(v *int, name string, value int, usage string) interface{} {
	flag.IntVar(v, name, value, usage)
	return nil
}

func flagString(v *string, name string, value string, usage string) interface{} {
	flag.StringVar(v, name, value, usage)
	return nil
}

func flagFloat64(v *float64, name string, value float64, usage string) interface{} {
	flag.Float64Var(v, name, value, usage)
	return nil
}

var (
	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot = flag.String("goroot", runtime.GOROOT(), "Go root directory")

	// layout control
	_           = flagInt(&godoc.TabWidth, "tabwidth", 4, "tab width")
	_           = flagBool(&godoc.ShowTimestamps, "timestamps", false, "show timestamps with directory listings")
	templateDir = flag.String("templates", "", "directory containing alternate template files")
	_           = flagBool(&godoc.ShowPlayground, "play", false, "enable playground in web interface")
	_           = flagBool(&godoc.ShowExamples, "ex", false, "show examples in command line mode")
	_           = flagBool(&godoc.DeclLinks, "links", true, "link identifiers to their declarations")

	// search index
	indexEnabled = flag.Bool("index", false, "enable search index")
	indexFiles   = flag.String("index_files", "", "glob pattern specifying index files;"+
		"if not empty, the index is read from these files in sorted order")
	_ = flagInt(&godoc.MaxResults, "maxresults", 10000, "maximum number of full text search results shown")
	_ = flagFloat64(&godoc.IndexThrottle, "index_throttle", 0.75, "index throttle value; 0.0 = no time allocated, 1.0 = full throttle")

	// source code notes
	_ = flagString(&godoc.NotesRx, "notes", "BUG", "regular expression matching note markers to show")
)

var pres *godoc.Presentation

func registerPublicHandlers(mux *http.ServeMux) {
	godoc.CmdHandler.RegisterWithMux(mux)
	godoc.PkgHandler.RegisterWithMux(mux)
	mux.HandleFunc("/doc/codewalk/", codewalk)
	mux.Handle("/doc/play/", godoc.FileServer)
	mux.HandleFunc("/search", search)
	mux.Handle("/robots.txt", godoc.FileServer)
	mux.HandleFunc("/opensearch.xml", serveSearchDesc)
	mux.HandleFunc("/", serveFile)
}

// ----------------------------------------------------------------------------
// Templates

func readTemplate(name string) *template.Template {
	path := "lib/godoc/" + name

	// use underlying file system fs to read the template file
	// (cannot use template ParseFile functions directly)
	data, err := vfs.ReadFile(godoc.FS, path)
	if err != nil {
		log.Fatal("readTemplate: ", err)
	}
	// be explicit with errors (for app engine use)
	t, err := template.New(name).Funcs(godoc.FuncMap).Parse(string(data))
	if err != nil {
		log.Fatal("readTemplate: ", err)
	}
	return t
}

var codewalkHTML, codewalkdirHTML *template.Template

func readTemplates() {
	// have to delay until after flags processing since paths depend on goroot
	codewalkHTML = readTemplate("codewalk.html")
	codewalkdirHTML = readTemplate("codewalkdir.html")
	godoc.DirlistHTML = readTemplate("dirlist.html")
	godoc.ErrorHTML = readTemplate("error.html")
	godoc.ExampleHTML = readTemplate("example.html")
	godoc.GodocHTML = readTemplate("godoc.html")
	godoc.PackageHTML = readTemplate("package.html")
	godoc.PackageText = readTemplate("package.txt")
	godoc.SearchHTML = readTemplate("search.html")
	godoc.SearchText = readTemplate("search.txt")
	godoc.SearchDescXML = readTemplate("opensearch.xml")
}

// ----------------------------------------------------------------------------
// Files

func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}

func redirect(w http.ResponseWriter, r *http.Request) (redirected bool) {
	canonical := pathpkg.Clean(r.URL.Path)
	if !strings.HasSuffix(canonical, "/") {
		canonical += "/"
	}
	if r.URL.Path != canonical {
		url := *r.URL
		url.Path = canonical
		http.Redirect(w, r, url.String(), http.StatusMovedPermanently)
		redirected = true
	}
	return
}

func redirectFile(w http.ResponseWriter, r *http.Request) (redirected bool) {
	c := pathpkg.Clean(r.URL.Path)
	c = strings.TrimRight(c, "/")
	if r.URL.Path != c {
		url := *r.URL
		url.Path = c
		http.Redirect(w, r, url.String(), http.StatusMovedPermanently)
		redirected = true
	}
	return
}

func serveTextFile(w http.ResponseWriter, r *http.Request, abspath, relpath, title string) {
	src, err := vfs.ReadFile(godoc.FS, abspath)
	if err != nil {
		log.Printf("ReadFile: %s", err)
		pres.ServeError(w, r, relpath, err)
		return
	}

	if r.FormValue("m") == "text" {
		pres.ServeText(w, src)
		return
	}

	var buf bytes.Buffer
	buf.WriteString("<pre>")
	godoc.FormatText(&buf, src, 1, pathpkg.Ext(abspath) == ".go", r.FormValue("h"), godoc.RangeSelection(r.FormValue("s")))
	buf.WriteString("</pre>")
	fmt.Fprintf(&buf, `<p><a href="/%s?m=text">View as plain text</a></p>`, htmlpkg.EscapeString(relpath))

	pres.ServePage(w, godoc.Page{
		Title:    title + " " + relpath,
		Tabtitle: relpath,
		Body:     buf.Bytes(),
	})
}

func serveDirectory(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	if redirect(w, r) {
		return
	}

	list, err := godoc.FS.ReadDir(abspath)
	if err != nil {
		pres.ServeError(w, r, relpath, err)
		return
	}

	pres.ServePage(w, godoc.Page{
		Title:    "Directory " + relpath,
		Tabtitle: relpath,
		Body:     applyTemplate(godoc.DirlistHTML, "dirlistHTML", list),
	})
}

func serveFile(w http.ResponseWriter, r *http.Request) {
	relpath := r.URL.Path

	// Check to see if we need to redirect or serve another file.
	if m := godoc.MetadataFor(relpath); m != nil {
		if m.Path != relpath {
			// Redirect to canonical path.
			http.Redirect(w, r, m.Path, http.StatusMovedPermanently)
			return
		}
		// Serve from the actual filesystem path.
		relpath = m.FilePath()
	}

	abspath := relpath
	relpath = relpath[1:] // strip leading slash

	switch pathpkg.Ext(relpath) {
	case ".html":
		if strings.HasSuffix(relpath, "/index.html") {
			// We'll show index.html for the directory.
			// Use the dir/ version as canonical instead of dir/index.html.
			http.Redirect(w, r, r.URL.Path[0:len(r.URL.Path)-len("index.html")], http.StatusMovedPermanently)
			return
		}
		pres.ServeHTMLDoc(w, r, abspath, relpath)
		return

	case ".go":
		serveTextFile(w, r, abspath, relpath, "Source file")
		return
	}

	dir, err := godoc.FS.Lstat(abspath)
	if err != nil {
		log.Print(err)
		pres.ServeError(w, r, relpath, err)
		return
	}

	if dir != nil && dir.IsDir() {
		if redirect(w, r) {
			return
		}
		if index := pathpkg.Join(abspath, "index.html"); util.IsTextFile(godoc.FS, index) {
			pres.ServeHTMLDoc(w, r, index, index)
			return
		}
		serveDirectory(w, r, abspath, relpath)
		return
	}

	if util.IsTextFile(godoc.FS, abspath) {
		if redirectFile(w, r) {
			return
		}
		serveTextFile(w, r, abspath, relpath, "Text file")
		return
	}

	godoc.FileServer.ServeHTTP(w, r)
}

func serveSearchDesc(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/opensearchdescription+xml")
	data := map[string]interface{}{
		"BaseURL": fmt.Sprintf("http://%s", r.Host),
	}
	if err := godoc.SearchDescXML.Execute(w, &data); err != nil && err != http.ErrBodyNotAllowed {
		// Only log if there's an error that's not about writing on HEAD requests.
		// See Issues 5451 and 5454.
		log.Printf("searchDescXML.Execute: %s", err)
	}
}

// ----------------------------------------------------------------------------
// Packages

// remoteSearchURL returns the search URL for a given query as needed by
// remoteSearch. If html is set, an html result is requested; otherwise
// the result is in textual form.
// Adjust this function as necessary if modeNames or FormValue parameters
// change.
func remoteSearchURL(query string, html bool) string {
	s := "/search?m=text&q="
	if html {
		s = "/search?q="
	}
	return s + url.QueryEscape(query)
}

// ----------------------------------------------------------------------------
// Search

type SearchResult struct {
	Query string
	Alert string // error or warning message

	// identifier matches
	Pak godoc.HitList       // packages matching Query
	Hit *godoc.LookupResult // identifier matches of Query
	Alt *godoc.AltWords     // alternative identifiers to look for

	// textual matches
	Found    int               // number of textual occurrences found
	Textual  []godoc.FileLines // textual matches of Query
	Complete bool              // true if all textual occurrences of Query are reported
}

func lookup(query string) (result SearchResult) {
	result.Query = query

	index, timestamp := godoc.SearchIndex.Get()
	if index != nil {
		index := index.(*godoc.Index)

		// identifier search
		var err error
		result.Pak, result.Hit, result.Alt, err = index.Lookup(query)
		if err != nil && godoc.MaxResults <= 0 {
			// ignore the error if full text search is enabled
			// since the query may be a valid regular expression
			result.Alert = "Error in query string: " + err.Error()
			return
		}

		// full text search
		if godoc.MaxResults > 0 && query != "" {
			rx, err := regexp.Compile(query)
			if err != nil {
				result.Alert = "Error in query regular expression: " + err.Error()
				return
			}
			// If we get maxResults+1 results we know that there are more than
			// maxResults results and thus the result may be incomplete (to be
			// precise, we should remove one result from the result set, but
			// nobody is going to count the results on the result page).
			result.Found, result.Textual = index.LookupRegexp(rx, godoc.MaxResults+1)
			result.Complete = result.Found <= godoc.MaxResults
			if !result.Complete {
				result.Found-- // since we looked for maxResults+1
			}
		}
	}

	// is the result accurate?
	if pres.Corpus.IndexEnabled {
		if _, ts := godoc.FSModified.Get(); timestamp.Before(ts) {
			// The index is older than the latest file system change under godoc's observation.
			result.Alert = "Indexing in progress: result may be inaccurate"
		}
	} else {
		result.Alert = "Search index disabled: no results available"
	}

	return
}

func search(w http.ResponseWriter, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	result := lookup(query)

	if godoc.GetPageInfoMode(r)&godoc.NoHTML != 0 {
		pres.ServeText(w, applyTemplate(godoc.SearchText, "searchText", result))
		return
	}

	var title string
	if result.Hit != nil || len(result.Textual) > 0 {
		title = fmt.Sprintf(`Results for query %q`, query)
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	pres.ServePage(w, godoc.Page{
		Title:    title,
		Tabtitle: query,
		Query:    query,
		Body:     applyTemplate(godoc.SearchHTML, "searchHTML", result),
	})
}
