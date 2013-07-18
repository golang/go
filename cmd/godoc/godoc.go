// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"
	pathpkg "path"
	"regexp"
	"runtime"
	"strings"
	"text/template"

	"code.google.com/p/go.tools/godoc"
	"code.google.com/p/go.tools/godoc/vfs"
)

var (
	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot = flag.String("goroot", runtime.GOROOT(), "Go root directory")

	// layout control
	tabWidth       = flag.Int("tabwidth", 4, "tab width")
	showTimestamps = flag.Bool("timestamps", false, "show timestamps with directory listings")
	templateDir    = flag.String("templates", "", "directory containing alternate template files")
	showPlayground = flag.Bool("play", false, "enable playground in web interface")
	showExamples   = flag.Bool("ex", false, "show examples in command line mode")
	declLinks      = flag.Bool("links", true, "link identifiers to their declarations")

	// search index
	indexEnabled = flag.Bool("index", false, "enable search index")
	indexFiles   = flag.String("index_files", "", "glob pattern specifying index files;"+
		"if not empty, the index is read from these files in sorted order")
	maxResults    = flag.Int("maxresults", 10000, "maximum number of full text search results shown")
	indexThrottle = flag.Float64("index_throttle", 0.75, "index throttle value; 0.0 = no time allocated, 1.0 = full throttle")

	// source code notes
	notesRx = flag.String("notes", "BUG", "regular expression matching note markers to show")
)

var (
	pres *godoc.Presentation
	fs   = vfs.NameSpace{}
)

func registerPublicHandlers(mux *http.ServeMux) {
	if pres == nil {
		panic("nil Presentation")
	}
	godoc.CmdHandler.RegisterWithMux(mux)
	godoc.PkgHandler.RegisterWithMux(mux)
	mux.HandleFunc("/doc/codewalk/", codewalk)
	mux.Handle("/doc/play/", godoc.FileServer)
	mux.HandleFunc("/search", search)
	mux.Handle("/robots.txt", godoc.FileServer)
	mux.HandleFunc("/opensearch.xml", serveSearchDesc)
	mux.HandleFunc("/", pres.ServeFile)
}

// ----------------------------------------------------------------------------
// Templates

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

	corp := pres.Corpus
	index, timestamp := corp.CurrentIndex()
	if index != nil {
		// identifier search
		var err error
		result.Pak, result.Hit, result.Alt, err = index.Lookup(query)
		if err != nil && corp.MaxResults <= 0 {
			// ignore the error if full text search is enabled
			// since the query may be a valid regular expression
			result.Alert = "Error in query string: " + err.Error()
			return
		}

		// full text search
		if corp.MaxResults > 0 && query != "" {
			rx, err := regexp.Compile(query)
			if err != nil {
				result.Alert = "Error in query regular expression: " + err.Error()
				return
			}
			// If we get maxResults+1 results we know that there are more than
			// maxResults results and thus the result may be incomplete (to be
			// precise, we should remove one result from the result set, but
			// nobody is going to count the results on the result page).
			result.Found, result.Textual = index.LookupRegexp(rx, corp.MaxResults+1)
			result.Complete = result.Found <= corp.MaxResults
			if !result.Complete {
				result.Found-- // since we looked for maxResults+1
			}
		}
	}

	// is the result accurate?
	if pres.Corpus.IndexEnabled {
		if ts := pres.Corpus.FSModifiedTime(); timestamp.Before(ts) {
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
