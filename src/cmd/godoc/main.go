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
	"io"
	"log"
	"net/http"
	_ "net/http/pprof" // to serve /debug/pprof/*
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

const defaultAddr = ":6060" // default webserver address

var (
	// file system to serve
	// (with e.g.: zip -r go.zip $GOROOT -i \*.go -i \*.html -i \*.css -i \*.js -i \*.txt -i \*.c -i \*.h -i \*.s -i \*.png -i \*.jpg -i \*.sh -i favicon.ico)
	zipfile = flag.String("zip", "", "zip file providing the file system to serve; disabled if empty")

	// file-based index
	writeIndex = flag.Bool("write_index", false, "write index to a file; the file name must be specified with -index_files")

	// periodic sync
	syncCmd   = flag.String("sync", "", "sync command; disabled if empty")
	syncMin   = flag.Int("sync_minutes", 0, "sync interval in minutes; disabled if <= 0")
	syncDelay delayTime // actual sync interval in minutes; usually syncDelay == syncMin, but syncDelay may back off exponentially

	// network
	httpAddr   = flag.String("http", "", "HTTP service address (e.g., '"+defaultAddr+"')")
	serverAddr = flag.String("server", "", "webserver address for command line searches")

	// layout control
	html    = flag.Bool("html", false, "print HTML in command-line mode")
	srcMode = flag.Bool("src", false, "print (exported) source in command-line mode")

	// command-line searches
	query = flag.Bool("q", false, "arguments are considered search queries")
)

func serveError(w http.ResponseWriter, r *http.Request, relpath string, err error) {
	contents := applyTemplate(errorHTML, "errorHTML", err) // err may contain an absolute path!
	w.WriteHeader(http.StatusNotFound)
	servePage(w, "File "+relpath, "", "", contents)
}

func exec(rw http.ResponseWriter, args []string) (status int) {
	r, w, err := os.Pipe()
	if err != nil {
		log.Printf("os.Pipe(): %v", err)
		return 2
	}

	bin := args[0]
	fds := []*os.File{nil, w, w}
	if *verbose {
		log.Printf("executing %v", args)
	}
	p, err := os.StartProcess(bin, args, &os.ProcAttr{Files: fds, Dir: *goroot})
	defer r.Close()
	w.Close()
	if err != nil {
		log.Printf("os.StartProcess(%q): %v", bin, err)
		return 2
	}

	var buf bytes.Buffer
	io.Copy(&buf, r)
	wait, err := p.Wait()
	if err != nil {
		os.Stderr.Write(buf.Bytes())
		log.Printf("os.Wait(%d, 0): %v", p.Pid, err)
		return 2
	}
	if !wait.Success() {
		os.Stderr.Write(buf.Bytes())
		log.Printf("executing %v failed", args)
		status = 1 // See comment in default case in dosync.
		return
	}

	if *verbose {
		os.Stderr.Write(buf.Bytes())
	}
	if rw != nil {
		rw.Header().Set("Content-Type", "text/plain; charset=utf-8")
		rw.Write(buf.Bytes())
	}

	return
}

func dosync(w http.ResponseWriter, r *http.Request) {
	args := []string{"/bin/sh", "-c", *syncCmd}
	switch exec(w, args) {
	case 0:
		// sync succeeded and some files have changed;
		// update package tree.
		// TODO(gri): The directory tree may be temporarily out-of-sync.
		//            Consider keeping separate time stamps so the web-
		//            page can indicate this discrepancy.
		initFSTree()
		fallthrough
	case 1:
		// sync failed because no files changed;
		// don't change the package tree
		syncDelay.set(time.Duration(*syncMin) * time.Minute) //  revert to regular sync schedule
	default:
		// TODO(r): this cannot happen now, since Wait has a boolean exit condition,
		// not an integer.
		// sync failed because of an error - back off exponentially, but try at least once a day
		syncDelay.backoff(24 * time.Hour)
	}
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
	if (*httpAddr != "") != (flag.NArg() == 0) && !*writeIndex {
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
		*goroot = filepath.Clean(*goroot) // normalize path separator
		fs = OS
		fsHttp = http.Dir(*goroot)
	} else {
		// use file system specified via .zip file (path separator must be '/')
		rc, err := zip.OpenReader(*zipfile)
		if err != nil {
			log.Fatalf("%s: %s\n", *zipfile, err)
		}
		defer rc.Close()                  // be nice (e.g., -writeIndex mode)
		*goroot = path.Join("/", *goroot) // fsHttp paths are relative to '/'
		fs = NewZipFS(rc)
		fsHttp = NewHttpZipFS(rc, *goroot)
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
		initDirTrees()

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
			if !fsMap.IsEmpty() {
				log.Print("user-defined mapping:")
				fsMap.Fprint(os.Stderr)
			}
			handler = loggingHandler(handler)
		}

		registerPublicHandlers(http.DefaultServeMux)
		if *syncCmd != "" {
			http.Handle("/debug/sync", http.HandlerFunc(dosync))
		}

		// Initialize default directory tree with corresponding timestamp.
		// (Do it in a goroutine so that launch is quick.)
		go initFSTree()

		// Initialize directory trees for user-defined file systems (-path flag).
		initDirTrees()

		// Start sync goroutine, if enabled.
		if *syncCmd != "" && *syncMin > 0 {
			syncDelay.set(*syncMin) // initial sync delay
			go func() {
				for {
					dosync(nil, nil)
					delay, _ := syncDelay.get()
					dt := delay.(time.Duration)
					if *verbose {
						log.Printf("next sync in %s", dt)
					}
					time.Sleep(dt)
				}
			}()
		}

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

	// determine paths
	const cmdPrefix = "cmd/"
	path := flag.Arg(0)
	var forceCmd bool
	if strings.HasPrefix(path, ".") {
		// assume cwd; don't assume -goroot
		cwd, _ := os.Getwd() // ignore errors
		path = filepath.Join(cwd, path)
	} else if strings.HasPrefix(path, cmdPrefix) {
		path = path[len(cmdPrefix):]
		forceCmd = true
	}
	relpath := path
	abspath := path
	if bp, _ := build.Import(path, "", build.FindOnly); bp.Dir != "" && bp.ImportPath != "" {
		relpath = bp.ImportPath
		abspath = bp.Dir
	} else if !filepath.IsAbs(path) {
		abspath = absolutePath(path, pkgHandler.fsRoot)
	} else {
		relpath = relativeURL(path)
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
	// TODO(gri): Provide a mechanism (flag?) to select a package
	//            if there are multiple packages in a directory.

	// first, try as package unless forced as command
	var info PageInfo
	if !forceCmd {
		info = pkgHandler.getPageInfo(abspath, relpath, "", mode)
	}

	// second, try as command unless the path is absolute
	// (the go command invokes godoc w/ absolute paths; don't override)
	var cinfo PageInfo
	if !filepath.IsAbs(path) {
		abspath = absolutePath(path, cmdHandler.fsRoot)
		cinfo = cmdHandler.getPageInfo(abspath, relpath, "", mode)
	}

	// determine what to use
	if info.IsEmpty() {
		if !cinfo.IsEmpty() {
			// only cinfo exists - switch to cinfo
			info = cinfo
		}
	} else if !cinfo.IsEmpty() {
		// both info and cinfo exist - use cinfo if info
		// contains only subdirectory information
		if info.PAst == nil && info.PDoc == nil {
			info = cinfo
		} else {
			fmt.Printf("use 'godoc %s%s' for documentation on the %s command \n\n", cmdPrefix, relpath, relpath)
		}
	}

	if info.Err != nil {
		log.Fatalf("%v", info.Err)
	}

	// If we have more than one argument, use the remaining arguments for filtering
	if flag.NArg() > 1 {
		args := flag.Args()[1:]
		rx := makeRx(args)
		if rx == nil {
			log.Fatalf("illegal regular expression from %v", args)
		}

		filter := func(s string) bool { return rx.MatchString(s) }
		switch {
		case info.PAst != nil:
			ast.FilterFile(info.PAst, filter)
			// Special case: Don't use templates for printing
			// so we only get the filtered declarations without
			// package clause or extra whitespace.
			for i, d := range info.PAst.Decls {
				if i > 0 {
					fmt.Println()
				}
				if *html {
					var buf bytes.Buffer
					writeNode(&buf, info.FSet, d)
					FormatText(os.Stdout, buf.Bytes(), -1, true, "", nil)
				} else {
					writeNode(os.Stdout, info.FSet, d)
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
