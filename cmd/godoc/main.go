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
	"bytes"
	_ "expvar" // to serve /debug/vars
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/printer"
	"log"
	"net/http"
	"net/http/httptest"
	_ "net/http/pprof" // to serve /debug/pprof/*
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"code.google.com/p/go.tools/godoc"
	"code.google.com/p/go.tools/godoc/static"
	"code.google.com/p/go.tools/godoc/vfs"
	"code.google.com/p/go.tools/godoc/vfs/mapfs"
	"code.google.com/p/go.tools/godoc/vfs/zipfs"
)

const (
	defaultAddr  = ":6060" // default webserver address
	templatePath = "code.google.com/p/go.tools/cmd/godoc/template"
)

var (
	// file system to serve
	// (with e.g.: zip -r go.zip $GOROOT -i \*.go -i \*.html -i \*.css -i \*.js -i \*.txt -i \*.c -i \*.h -i \*.s -i \*.png -i \*.jpg -i \*.sh -i favicon.ico)
	zipfile = flag.String("zip", "", "zip file providing the file system to serve; disabled if empty")

	// file-based index
	writeIndex = flag.Bool("write_index", false, "write index to a file; the file name must be specified with -index_files")

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

// Does s look like a regular expression?
func isRegexp(s string) bool {
	return strings.IndexAny(s, ".(|)*+?^$[]") >= 0
}

// Make a regular expression of the form
// names[0]|names[1]|...names[len(names)-1].
// Returns nil if the regular expression is illegal.
func makeRx(names []string) (rx *regexp.Regexp) {
	if len(names) > 0 {
		s := ""
		for i, name := range names {
			if i > 0 {
				s += "|"
			}
			if isRegexp(name) {
				s += name
			} else {
				s += "^" + name + "$" // must match exactly
			}
		}
		rx, _ = regexp.Compile(s) // rx is nil if there's a compilation error
	}
	return
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

	// Check usage: either server and no args, command line and args, or index creation mode
	if (*httpAddr != "" || *urlFlag != "") != (flag.NArg() == 0) && !*writeIndex {
		usage()
	}

	// Determine file system to use.
	if *zipfile == "" {
		// use file system of underlying OS
		fs.Bind("/", vfs.OS(*goroot), "/", vfs.BindReplace)
		if *templateDir != "" {
			fs.Bind("/lib/godoc", vfs.OS(*templateDir), "/", vfs.BindBefore)
		} else {
			fs.Bind("/lib/godoc", mapfs.New(static.Files), "/", vfs.BindReplace)
		}
	} else {
		// use file system specified via .zip file (path separator must be '/')
		rc, err := zip.OpenReader(*zipfile)
		if err != nil {
			log.Fatalf("%s: %s\n", *zipfile, err)
		}
		defer rc.Close() // be nice (e.g., -writeIndex mode)
		fs.Bind("/", zipfs.New(rc, *zipfile), *goroot, vfs.BindReplace)
	}

	// Bind $GOPATH trees into Go root.
	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		fs.Bind("/src/pkg", vfs.OS(p), "/src", vfs.BindAfter)
	}

	httpMode := *httpAddr != ""

	corpus := godoc.NewCorpus(fs)
	corpus.Verbose = *verbose
	corpus.IndexEnabled = *indexEnabled && httpMode
	corpus.IndexFiles = *indexFiles
	corpus.IndexThrottle = *indexThrottle
	if *writeIndex {
		corpus.IndexThrottle = 1.0
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
	if *notesRx != "" {
		pres.NotesRx = regexp.MustCompile(*notesRx)
	}

	readTemplates(pres, httpMode || *urlFlag != "" || *html)
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
		err = index.Write(f)
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

		// Start http server.
		if err := http.ListenAndServe(*httpAddr, handler); err != nil {
			log.Fatalf("ListenAndServe %s: %v", *httpAddr, err)
		}

		return
	}

	packageText := pres.PackageText

	// Command line mode.
	if *html {
		packageText = pres.PackageHTML
	}

	if *query {
		handleRemoteSearch()
		return
	}

	// Determine paths.
	//
	// If we are passed an operating system path like . or ./foo or /foo/bar or c:\mysrc,
	// we need to map that path somewhere in the fs name space so that routines
	// like getPageInfo will see it.  We use the arbitrarily-chosen virtual path "/target"
	// for this.  That is, if we get passed a directory like the above, we map that
	// directory so that getPageInfo sees it as /target.
	const target = "/target"
	const cmdPrefix = "cmd/"
	path := flag.Arg(0)
	var forceCmd bool
	var abspath, relpath string
	if filepath.IsAbs(path) {
		fs.Bind(target, vfs.OS(path), "/", vfs.BindReplace)
		abspath = target
	} else if build.IsLocalImport(path) {
		cwd, _ := os.Getwd() // ignore errors
		path = filepath.Join(cwd, path)
		fs.Bind(target, vfs.OS(path), "/", vfs.BindReplace)
		abspath = target
	} else if strings.HasPrefix(path, cmdPrefix) {
		path = strings.TrimPrefix(path, cmdPrefix)
		forceCmd = true
	} else if bp, _ := build.Import(path, "", build.FindOnly); bp.Dir != "" && bp.ImportPath != "" {
		fs.Bind(target, vfs.OS(bp.Dir), "/", vfs.BindReplace)
		abspath = target
		relpath = bp.ImportPath
	} else {
		abspath = pathpkg.Join(pres.PkgFSRoot(), path)
	}
	if relpath == "" {
		relpath = abspath
	}

	var mode godoc.PageInfoMode
	if relpath == "builtin" {
		// the fake built-in package contains unexported identifiers
		mode = godoc.NoFiltering | godoc.NoFactoryFuncs
	}
	if *srcMode {
		// only filter exports if we don't have explicit command-line filter arguments
		if flag.NArg() > 1 {
			mode |= godoc.NoFiltering
		}
		mode |= godoc.ShowSource
	}

	// first, try as package unless forced as command
	var info *godoc.PageInfo
	if !forceCmd {
		info = pres.GetPkgPageInfo(abspath, relpath, mode)
	}

	// second, try as command unless the path is absolute
	// (the go command invokes godoc w/ absolute paths; don't override)
	var cinfo *godoc.PageInfo
	if !filepath.IsAbs(path) {
		abspath = pathpkg.Join(pres.CmdFSRoot(), path)
		cinfo = pres.GetCmdPageInfo(abspath, relpath, mode)
	}

	// determine what to use
	if info == nil || info.IsEmpty() {
		if cinfo != nil && !cinfo.IsEmpty() {
			// only cinfo exists - switch to cinfo
			info = cinfo
		}
	} else if cinfo != nil && !cinfo.IsEmpty() {
		// both info and cinfo exist - use cinfo if info
		// contains only subdirectory information
		if info.PAst == nil && info.PDoc == nil {
			info = cinfo
		} else {
			fmt.Printf("use 'godoc %s%s' for documentation on the %s command \n\n", cmdPrefix, relpath, relpath)
		}
	}

	if info == nil {
		log.Fatalf("%s: no such directory or package", flag.Arg(0))
	}
	if info.Err != nil {
		log.Fatalf("%v", info.Err)
	}

	if info.PDoc != nil && info.PDoc.ImportPath == target {
		// Replace virtual /target with actual argument from command line.
		info.PDoc.ImportPath = flag.Arg(0)
	}

	// If we have more than one argument, use the remaining arguments for filtering.
	if flag.NArg() > 1 {
		args := flag.Args()[1:]
		rx := makeRx(args)
		if rx == nil {
			log.Fatalf("illegal regular expression from %v", args)
		}

		filter := func(s string) bool { return rx.MatchString(s) }
		switch {
		case info.PAst != nil:
			cmap := ast.NewCommentMap(info.FSet, info.PAst, info.PAst.Comments)
			ast.FilterFile(info.PAst, filter)
			// Special case: Don't use templates for printing
			// so we only get the filtered declarations without
			// package clause or extra whitespace.
			for i, d := range info.PAst.Decls {
				// determine the comments associated with d only
				comments := cmap.Filter(d).Comments()
				cn := &printer.CommentedNode{Node: d, Comments: comments}
				if i > 0 {
					fmt.Println()
				}
				if *html {
					var buf bytes.Buffer
					pres.WriteNode(&buf, info.FSet, cn)
					godoc.FormatText(os.Stdout, buf.Bytes(), -1, true, "", nil)
				} else {
					pres.WriteNode(os.Stdout, info.FSet, cn)
				}
				fmt.Println()
			}
			return

		case info.PDoc != nil:
			info.PDoc.Filter(filter)
		}
	}

	if err := packageText.Execute(os.Stdout, info); err != nil {
		log.Printf("packageText.Execute: %s", err)
	}
}
