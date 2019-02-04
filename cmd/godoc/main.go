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

package main

import (
	"archive/zip"
	"bytes"
	_ "expvar" // to serve /debug/vars
	"flag"
	"fmt"
	"go/build"
	"log"
	"net/http"
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

const defaultAddr = "localhost:6060" // default webserver address

var (
	// file system to serve
	// (with e.g.: zip -r go.zip $GOROOT -i \*.go -i \*.html -i \*.css -i \*.js -i \*.txt -i \*.c -i \*.h -i \*.s -i \*.png -i \*.jpg -i \*.sh -i favicon.ico)
	zipfile = flag.String("zip", "", "zip file providing the file system to serve; disabled if empty")

	// file-based index
	writeIndex = flag.Bool("write_index", false, "write index to a file; the file name must be specified with -index_files")

	analysisFlag = flag.String("analysis", "", `comma-separated list of analyses to perform (supported: type, pointer). See http://golang.org/lib/godoc/analysis/help.html`)

	// network
	httpAddr = flag.String("http", defaultAddr, "HTTP service address")

	// layout control
	urlFlag = flag.String("url", "", "print HTML for named URL")

	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot = flag.String("goroot", findGOROOT(), "Go root directory")

	// layout control
	showTimestamps = flag.Bool("timestamps", false, "show timestamps with directory listings")
	templateDir    = flag.String("templates", "", "load templates/JS/CSS from disk in this directory")
	showPlayground = flag.Bool("play", false, "enable playground")
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

// An httpResponseRecorder is an http.ResponseWriter
type httpResponseRecorder struct {
	body   *bytes.Buffer
	header http.Header
	code   int
}

func (w *httpResponseRecorder) Header() http.Header         { return w.header }
func (w *httpResponseRecorder) Write(b []byte) (int, error) { return w.body.Write(b) }
func (w *httpResponseRecorder) WriteHeader(code int)        { w.code = code }

func usage() {
	fmt.Fprintf(os.Stderr, "usage: godoc -http="+defaultAddr+"\n")
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
		w := &httpResponseRecorder{code: 200, header: make(http.Header), body: new(bytes.Buffer)}
		http.DefaultServeMux.ServeHTTP(w, req)

		// Return data, error, or follow redirect.
		switch w.code {
		case 200: // ok
			os.Stdout.Write(w.body.Bytes())
			return
		case 301, 302, 303, 307: // redirect
			redirect := w.header.Get("Location")
			if redirect == "" {
				log.Fatalf("HTTP %d without Location header", w.code)
			}
			urlstr = redirect
		default:
			log.Fatalf("HTTP error %d", w.code)
		}
	}
	log.Fatalf("too many redirects")
}

func initCorpus(corpus *godoc.Corpus) {
	err := corpus.Init()
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	flag.Usage = usage
	flag.Parse()

	if certInit != nil {
		certInit()
	}

	playEnabled = *showPlayground

	// Check usage.
	if flag.NArg() > 0 {
		fmt.Fprintln(os.Stderr, `Unexpected arguments. Use "go doc" for command-line help output instead. For example, "go doc fmt.Printf".`)
		usage()
	}
	if *httpAddr == "" && *urlFlag == "" && !*writeIndex {
		fmt.Fprintln(os.Stderr, "At least one of -http, -url, or -write_index must be set to a non-zero value.")
		usage()
	}

	// Set the resolved goroot.
	vfs.GOROOT = *goroot

	fsGate := make(chan bool, 20)

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
	corpus.IndexEnabled = *indexEnabled
	if *maxResults == 0 {
		corpus.IndexFullText = false
	}
	corpus.IndexFiles = *indexFiles
	corpus.IndexDirectory = indexDirectoryDefault
	corpus.IndexThrottle = *indexThrottle
	corpus.IndexInterval = *indexInterval
	if *writeIndex || *urlFlag != "" {
		corpus.IndexThrottle = 1.0
		corpus.IndexEnabled = true
		initCorpus(corpus)
	} else {
		go initCorpus(corpus)
	}

	// Initialize the version info before readTemplates, which saves
	// the map value in a method value.
	corpus.InitVersionInfo()

	pres = godoc.NewPresentation(corpus)
	pres.ShowTimestamps = *showTimestamps
	pres.ShowPlayground = *showPlayground
	pres.DeclLinks = *declLinks
	if *notesRx != "" {
		pres.NotesRx = regexp.MustCompile(*notesRx)
	}

	readTemplates(pres)
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

	var handler http.Handler = http.DefaultServeMux
	if *verbose {
		log.Printf("Go Documentation Server")
		log.Printf("version = %s", runtime.Version())
		log.Printf("address = %s", *httpAddr)
		log.Printf("goroot = %s", *goroot)
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

	if runHTTPS != nil {
		go func() {
			if err := runHTTPS(handler); err != nil {
				log.Fatalf("ListenAndServe TLS: %v", err)
			}
		}()
	}

	// Start http server.
	if *verbose {
		log.Println("starting HTTP server")
	}
	if wrapHTTPMux != nil {
		handler = wrapHTTPMux(handler)
	}
	if err := http.ListenAndServe(*httpAddr, handler); err != nil {
		log.Fatalf("ListenAndServe %s: %v", *httpAddr, err)
	}
}

// Hooks that are set non-nil in autocert.go if the "autocert" build tag
// is used.
var (
	certInit    func()
	runHTTPS    func(http.Handler) error
	wrapHTTPMux func(http.Handler) http.Handler
)
