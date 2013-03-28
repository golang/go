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
	"errors"
	_ "expvar" // to serve /debug/vars
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/printer"
	"io"
	"log"
	"net/http"
	_ "net/http/pprof" // to serve /debug/pprof/*
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

const defaultAddr = ":6060" // default webserver address

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
)

func serveError(w http.ResponseWriter, r *http.Request, relpath string, err error) {
	w.WriteHeader(http.StatusNotFound)
	servePage(w, Page{
		Title:    "File " + relpath,
		Subtitle: relpath,
		Body:     applyTemplate(errorHTML, "errorHTML", err), // err may contain an absolute path!
	})
}

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

func remoteSearch(query string) (res *http.Response, err error) {
	// list of addresses to try
	var addrs []string
	if *serverAddr != "" {
		// explicit server address - only try this one
		addrs = []string{*serverAddr}
	} else {
		addrs = []string{
			defaultAddr,
			"golang.org",
		}
	}

	// remote search
	search := remoteSearchURL(query, *html)
	for _, addr := range addrs {
		url := "http://" + addr + search
		res, err = http.Get(url)
		if err == nil && res.StatusCode == http.StatusOK {
			break
		}
	}

	if err == nil && res.StatusCode != http.StatusOK {
		err = errors.New(res.Status)
	}

	return
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

func main() {
	flag.Usage = usage
	flag.Parse()

	// Check usage: either server and no args, command line and args, or index creation mode
	if (*httpAddr != "" || *urlFlag != "") != (flag.NArg() == 0) && !*writeIndex {
		usage()
	}

	if *tabwidth < 0 {
		log.Fatalf("negative tabwidth %d", *tabwidth)
	}

	// Determine file system to use.
	// TODO(gri) - fs and fsHttp should really be the same. Try to unify.
	//           - fsHttp doesn't need to be set up in command-line mode,
	//             same is true for the http handlers in initHandlers.
	if *zipfile == "" {
		// use file system of underlying OS
		fs.Bind("/", OS(*goroot), "/", bindReplace)
		if *templateDir != "" {
			fs.Bind("/lib/godoc", OS(*templateDir), "/", bindBefore)
		}
	} else {
		// use file system specified via .zip file (path separator must be '/')
		rc, err := zip.OpenReader(*zipfile)
		if err != nil {
			log.Fatalf("%s: %s\n", *zipfile, err)
		}
		defer rc.Close() // be nice (e.g., -writeIndex mode)
		fs.Bind("/", NewZipFS(rc, *zipfile), *goroot, bindReplace)
	}

	// Bind $GOPATH trees into Go root.
	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		fs.Bind("/src/pkg", OS(p), "/src", bindAfter)
	}

	readTemplates()
	initHandlers()

	if *writeIndex {
		// Write search index and exit.
		if *indexFiles == "" {
			log.Fatal("no index file specified")
		}

		log.Println("initialize file systems")
		*verbose = true // want to see what happens
		initFSTree()

		*indexThrottle = 1
		updateIndex()

		log.Println("writing index file", *indexFiles)
		f, err := os.Create(*indexFiles)
		if err != nil {
			log.Fatal(err)
		}
		index, _ := searchIndex.get()
		err = index.(*Index).Write(f)
		if err != nil {
			log.Fatal(err)
		}

		log.Println("done")
		return
	}

	// Print content that would be served at the URL *urlFlag.
	if *urlFlag != "" {
		registerPublicHandlers(http.DefaultServeMux)
		initFSTree()
		updateMetadata()
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
			w := &httpWriter{h: http.Header{}, code: 200}
			http.DefaultServeMux.ServeHTTP(w, req)

			// Return data, error, or follow redirect.
			switch w.code {
			case 200: // ok
				os.Stdout.Write(w.Bytes())
				return
			case 301, 302, 303, 307: // redirect
				redirect := w.h.Get("Location")
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

	if *httpAddr != "" {
		// HTTP server mode.
		var handler http.Handler = http.DefaultServeMux
		if *verbose {
			log.Printf("Go Documentation Server")
			log.Printf("version = %s", runtime.Version())
			log.Printf("address = %s", *httpAddr)
			log.Printf("goroot = %s", *goroot)
			log.Printf("tabwidth = %d", *tabwidth)
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

		registerPublicHandlers(http.DefaultServeMux)
		registerPlaygroundHandlers(http.DefaultServeMux)

		// Initialize default directory tree with corresponding timestamp.
		// (Do it in a goroutine so that launch is quick.)
		go initFSTree()

		// Immediately update metadata.
		updateMetadata()
		// Periodically refresh metadata.
		go refreshMetadataLoop()

		// Initialize search index.
		if *indexEnabled {
			go indexer()
		}

		// Start http server.
		if err := http.ListenAndServe(*httpAddr, handler); err != nil {
			log.Fatalf("ListenAndServe %s: %v", *httpAddr, err)
		}

		return
	}

	// Command line mode.
	if *html {
		packageText = packageHTML
		searchText = packageHTML
	}

	if *query {
		// Command-line queries.
		for i := 0; i < flag.NArg(); i++ {
			res, err := remoteSearch(flag.Arg(i))
			if err != nil {
				log.Fatalf("remoteSearch: %s", err)
			}
			io.Copy(os.Stdout, res.Body)
		}
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
		fs.Bind(target, OS(path), "/", bindReplace)
		abspath = target
	} else if build.IsLocalImport(path) {
		cwd, _ := os.Getwd() // ignore errors
		path = filepath.Join(cwd, path)
		fs.Bind(target, OS(path), "/", bindReplace)
		abspath = target
	} else if strings.HasPrefix(path, cmdPrefix) {
		path = strings.TrimPrefix(path, cmdPrefix)
		forceCmd = true
	} else if bp, _ := build.Import(path, "", build.FindOnly); bp.Dir != "" && bp.ImportPath != "" {
		fs.Bind(target, OS(bp.Dir), "/", bindReplace)
		abspath = target
		relpath = bp.ImportPath
	} else {
		abspath = pathpkg.Join(pkgHandler.fsRoot, path)
	}
	if relpath == "" {
		relpath = abspath
	}

	var mode PageInfoMode
	if relpath == builtinPkgPath {
		// the fake built-in package contains unexported identifiers
		mode = noFiltering
	}
	if *srcMode {
		// only filter exports if we don't have explicit command-line filter arguments
		if flag.NArg() > 1 {
			mode |= noFiltering
		}
		mode |= showSource
	}

	// first, try as package unless forced as command
	var info *PageInfo
	if !forceCmd {
		info = pkgHandler.getPageInfo(abspath, relpath, mode)
	}

	// second, try as command unless the path is absolute
	// (the go command invokes godoc w/ absolute paths; don't override)
	var cinfo *PageInfo
	if !filepath.IsAbs(path) {
		abspath = pathpkg.Join(cmdHandler.fsRoot, path)
		cinfo = cmdHandler.getPageInfo(abspath, relpath, mode)
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
					writeNode(&buf, info.FSet, cn)
					FormatText(os.Stdout, buf.Bytes(), -1, true, "", nil)
				} else {
					writeNode(os.Stdout, info.FSet, cn)
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

// An httpWriter is an http.ResponseWriter writing to a bytes.Buffer.
type httpWriter struct {
	bytes.Buffer
	h    http.Header
	code int
}

func (w *httpWriter) Header() http.Header  { return w.h }
func (w *httpWriter) WriteHeader(code int) { w.code = code }
