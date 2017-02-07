// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/		main landing page
//	http://godoc/doc/	serve from $GOROOT/doc - spec, mem, etc.
//	http://godoc/src/	serve files from $GOROOT/src; .go gets pretty-printed
//	http://godoc/cmd/	serve documentation about commands
//	http://godoc/pkg/	serve documentation about packages
//				(idea is if you say import "compress/zlib", you go to
//				http://godoc/pkg/compress/zlib)
//
// Command-line interface:
//
//	godoc packagepath [name ...]
//
//	godoc compress/zlib
//		- prints doc for package compress/zlib
//	godoc crypto/block Cipher NewCMAC
//		- prints doc for Cipher and NewCMAC in package crypto/block

// +build !appengine

package main

import (
	"archive/zip"
	_ "expvar" // to serve /debug/vars
	"flag"
	"fmt"
	"go/build"
	"log"
	"net/http"
	"net/http/httptest"
	_ "net/http/pprof" // to serve /debug/pprof/*
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"golang.org/x/tools/godoc"
	"golang.org/x/tools/godoc/analysis"
	"golang.org/x/tools/godoc/static"
	"golang.org/x/tools/godoc/vfs"
	"golang.org/x/tools/godoc/vfs/gatefs"
	"golang.org/x/tools/godoc/vfs/mapfs"
	"golang.org/x/tools/godoc/vfs/zipfs"
)

const (
	defaultAddr = ":6060" // default webserver address
	toolsPath   = "golang.org/x/tools/cmd/"
)

var (
	// file system to serve
	// (with e.g.: zip -r go.zip $GOROOT -i \*.go -i \*.html -i \*.css -i \*.js -i \*.txt -i \*.c -i \*.h -i \*.s -i \*.png -i \*.jpg -i \*.sh -i favicon.ico)
	zipfile = flag.String("zip", "", "zip file providing the file system to serve; disabled if empty")

	// file-based index
	writeIndex = flag.Bool("write_index", false, "write index to a file; the file name must be specified with -index_files")

	analysisFlag = flag.String("analysis", "", `comma-separated list of analyses to perform (supported: type, pointer). See http://golang.org/lib/godoc/analysis/help.html`)

	// network
	httpAddr   = flag.String("http", "", "HTTP service address (e.g., '"+defaultAddr+"')")
	serverAddr = flag.String("server", "", "webserver address for command line searches")

	// layout control
	html    = flag.Bool("html", false, "print HTML in command-line mode")
	srcMode = flag.Bool("src", false, "print (exported) source in command-line mode")
	urlFlag = flag.String("url", "", "print HTML for named URL")

	// command-line searches
	query = flag.Bool("q", false, "arguments are considered search queries")

	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot = flag.String("goroot", runtime.GOROOT(), "Go root directory")

	// layout control
	tabWidth       = flag.Int("tabwidth", 4, "tab width")
	showTimestamps = flag.Bool("timestamps", false, "show timestamps with directory listings")
	templateDir    = flag.String("templates", "", "load templates/JS/CSS from disk in this directory")
	showPlayground = flag.Bool("play", false, "enable playground in web interface")
	showExamples   = flag.Bool("ex", false, "show examples in command line mode")
	declLinks      = flag.Bool("links", true, "link identifiers to their declarations")

	// search index
	indexEnabled  = flag.Bool("index", false, "enable search index")
	indexFiles    = flag.String("index_files", "", "glob pattern specifying index files; if not empty, the index is read from these files in sorted order")
	indexInterval = flag.Duration("index_interval", 0, "interval of indexing; 0 for default (5m), negative to only index once at startup")
	maxResults    = flag.Int("maxresults", 10000, "maximum number of full text search results shown")
	indexThrottle = flag.Float64("index_throttle", 0.75, "index throttle value; 0.0 = no time allocated, 1.0 = full throttle")

	// source code notes
	notesRx = flag.String("notes", "BUG", "regular expression matching note markers to show")
)

func usage() {
	fmt.Fprintf(os.Stderr,
		"usage: godoc package [name ...]\n"+
			"	godoc -http="+defaultAddr+"\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		log.Printf("%s\t%s", req.RemoteAddr, req.URL)
		h.ServeHTTP(w, req)
	})
}

func handleURLFlag() {
	// Try up to 10 fetches, following redirects.
	urlstr := *urlFlag
	for i := 0; i < 10; i++ {
		// Prepare request.
		u, err := url.Parse(urlstr)
		if err != nil {
			log.Fatal(err)
		}
		req := &http.Request{
			URL: u,
		}

		// Invoke default HTTP handler to serve request
		// to our buffering httpWriter.
		w := httptest.NewRecorder()
		http.DefaultServeMux.ServeHTTP(w, req)

		// Return data, error, or follow redirect.
		switch w.Code {
		case 200: // ok
			os.Stdout.Write(w.Body.Bytes())
			return
		case 301, 302, 303, 307: // redirect
			redirect := w.HeaderMap.Get("Location")
			if redirect == "" {
				log.Fatalf("HTTP %d without Location header", w.Code)
			}
			urlstr = redirect
		default:
			log.Fatalf("HTTP error %d", w.Code)
		}
	}
	log.Fatalf("too many redirects")
}

func main() {
	flag.Usage = usage
	flag.Parse()

	playEnabled = *showPlayground

	// Check usage: server and no args.
	if (*httpAddr != "" || *urlFlag != "") && (flag.NArg() > 0) {
		fmt.Fprintln(os.Stderr, "can't use -http with args.")
		usage()
	}

	// Check usage: command line args or index creation mode.
	if (*httpAddr != "" || *urlFlag != "") != (flag.NArg() == 0) && !*writeIndex {
		fmt.Fprintln(os.Stderr, "missing args.")
		usage()
	}

	var fsGate chan bool
	fsGate = make(chan bool, 20)

	// Determine file system to use.
	if *zipfile == "" {
		// use file system of underlying OS
		rootfs := gatefs.New(vfs.OS(*goroot), fsGate)
		fs.Bind("/", rootfs, "/", vfs.BindReplace)
	} else {
		// use file system specified via .zip file (path separator must be '/')
		rc, err := zip.OpenReader(*zipfile)
		if err != nil {
			log.Fatalf("%s: %s\n", *zipfile, err)
		}
		defer rc.Close() // be nice (e.g., -writeIndex mode)
		fs.Bind("/", zipfs.New(rc, *zipfile), *goroot, vfs.BindReplace)
	}
	if *templateDir != "" {
		fs.Bind("/lib/godoc", vfs.OS(*templateDir), "/", vfs.BindBefore)
	} else {
		fs.Bind("/lib/godoc", mapfs.New(static.Files), "/", vfs.BindReplace)
	}

	// Bind $GOPATH trees into Go root.
	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		fs.Bind("/src", gatefs.New(vfs.OS(p), fsGate), "/src", vfs.BindAfter)
	}

	httpMode := *httpAddr != ""

	var typeAnalysis, pointerAnalysis bool
	if *analysisFlag != "" {
		for _, a := range strings.Split(*analysisFlag, ",") {
			switch a {
			case "type":
				typeAnalysis = true
			case "pointer":
				pointerAnalysis = true
			default:
				log.Fatalf("unknown analysis: %s", a)
			}
		}
	}

	corpus := godoc.NewCorpus(fs)
	corpus.Verbose = *verbose
	corpus.MaxResults = *maxResults
	corpus.IndexEnabled = *indexEnabled && httpMode
	if *maxResults == 0 {
		corpus.IndexFullText = false
	}
	corpus.IndexFiles = *indexFiles
	corpus.IndexDirectory = indexDirectoryDefault
	corpus.IndexThrottle = *indexThrottle
	corpus.IndexInterval = *indexInterval
	if *writeIndex {
		corpus.IndexThrottle = 1.0
		corpus.IndexEnabled = true
	}
	if *writeIndex || httpMode || *urlFlag != "" {
		if err := corpus.Init(); err != nil {
			log.Fatal(err)
		}
	}

	pres = godoc.NewPresentation(corpus)
	pres.TabWidth = *tabWidth
	pres.ShowTimestamps = *showTimestamps
	pres.ShowPlayground = *showPlayground
	pres.ShowExamples = *showExamples
	pres.DeclLinks = *declLinks
	pres.SrcMode = *srcMode
	pres.HTMLMode = *html
	if *notesRx != "" {
		pres.NotesRx = regexp.MustCompile(*notesRx)
	}

	readTemplates(pres, httpMode || *urlFlag != "")
	registerHandlers(pres)

	if *writeIndex {
		// Write search index and exit.
		if *indexFiles == "" {
			log.Fatal("no index file specified")
		}

		log.Println("initialize file systems")
		*verbose = true // want to see what happens

		corpus.UpdateIndex()

		log.Println("writing index file", *indexFiles)
		f, err := os.Create(*indexFiles)
		if err != nil {
			log.Fatal(err)
		}
		index, _ := corpus.CurrentIndex()
		_, err = index.WriteTo(f)
		if err != nil {
			log.Fatal(err)
		}

		log.Println("done")
		return
	}

	// Print content that would be served at the URL *urlFlag.
	if *urlFlag != "" {
		handleURLFlag()
		return
	}

	if httpMode {
		// HTTP server mode.
		var handler http.Handler = http.DefaultServeMux
		if *verbose {
			log.Printf("Go Documentation Server")
			log.Printf("version = %s", runtime.Version())
			log.Printf("address = %s", *httpAddr)
			log.Printf("goroot = %s", *goroot)
			log.Printf("tabwidth = %d", *tabWidth)
			switch {
			case !*indexEnabled:
				log.Print("search index disabled")
			case *maxResults > 0:
				log.Printf("full text index enabled (maxresults = %d)", *maxResults)
			default:
				log.Print("identifier search index enabled")
			}
			fs.Fprint(os.Stderr)
			handler = loggingHandler(handler)
		}

		// Initialize search index.
		if *indexEnabled {
			go corpus.RunIndexer()
		}

		// Start type/pointer analysis.
		if typeAnalysis || pointerAnalysis {
			go analysis.Run(pointerAnalysis, &corpus.Analysis)
		}

		if serveAutoCertHook != nil {
			go func() {
				if err := serveAutoCertHook(handler); err != nil {
					log.Fatalf("ListenAndServe TLS: %v", err)
				}
			}()
		}

		// Start http server.
		if err := http.ListenAndServe(*httpAddr, handler); err != nil {
			log.Fatalf("ListenAndServe %s: %v", *httpAddr, err)
		}

		return
	}

	if *query {
		handleRemoteSearch()
		return
	}

	if err := godoc.CommandLine(os.Stdout, fs, pres, flag.Args()); err != nil {
		log.Print(err)
	}
}

// serveAutoCertHook if non-nil specifies a function to listen on port 443.
// See autocert.go.
var serveAutoCertHook func(http.Handler) error
