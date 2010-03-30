// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/		main landing page
//	http://godoc/doc/	serve from $GOROOT/doc - spec, mem, tutorial, etc.
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
	"bytes"
	_ "expvar" // to serve /debug/vars
	"flag"
	"fmt"
	"go/ast"
	"http"
	_ "http/pprof" // to serve /debug/pprof/*
	"io"
	"log"
	"os"
	pathutil "path"
	"regexp"
	"runtime"
	"strings"
	"time"
)

const defaultAddr = ":6060" // default webserver address

var (
	// periodic sync
	syncCmd   = flag.String("sync", "", "sync command; disabled if empty")
	syncMin   = flag.Int("sync_minutes", 0, "sync interval in minutes; disabled if <= 0")
	syncDelay delayTime // actual sync delay in minutes; usually syncDelay == syncMin, but delay may back off exponentially

	// network
	httpAddr   = flag.String("http", "", "HTTP service address (e.g., '"+defaultAddr+"')")
	serverAddr = flag.String("server", "", "webserver address for command line searches")

	// layout control
	html    = flag.Bool("html", false, "print HTML in command-line mode")
	srcMode = flag.Bool("src", false, "print (exported) source in command-line mode")

	// command-line searches
	query = flag.Bool("q", false, "arguments are considered search queries")
)


func serveError(c *http.Conn, r *http.Request, relpath string, err os.Error) {
	contents := applyTemplate(errorHTML, "errorHTML", err) // err may contain an absolute path!
	servePage(c, "File "+relpath, "", contents)
}


func exec(c *http.Conn, args []string) (status int) {
	r, w, err := os.Pipe()
	if err != nil {
		log.Stderrf("os.Pipe(): %v\n", err)
		return 2
	}

	bin := args[0]
	fds := []*os.File{nil, w, w}
	if *verbose {
		log.Stderrf("executing %v", args)
	}
	pid, err := os.ForkExec(bin, args, os.Environ(), *goroot, fds)
	defer r.Close()
	w.Close()
	if err != nil {
		log.Stderrf("os.ForkExec(%q): %v\n", bin, err)
		return 2
	}

	var buf bytes.Buffer
	io.Copy(&buf, r)
	wait, err := os.Wait(pid, 0)
	if err != nil {
		os.Stderr.Write(buf.Bytes())
		log.Stderrf("os.Wait(%d, 0): %v\n", pid, err)
		return 2
	}
	status = wait.ExitStatus()
	if !wait.Exited() || status > 1 {
		os.Stderr.Write(buf.Bytes())
		log.Stderrf("executing %v failed (exit status = %d)", args, status)
		return
	}

	if *verbose {
		os.Stderr.Write(buf.Bytes())
	}
	if c != nil {
		c.SetHeader("content-type", "text/plain; charset=utf-8")
		c.Write(buf.Bytes())
	}

	return
}


// Maximum directory depth, adjust as needed.
const maxDirDepth = 24

func dosync(c *http.Conn, r *http.Request) {
	args := []string{"/bin/sh", "-c", *syncCmd}
	switch exec(c, args) {
	case 0:
		// sync succeeded and some files have changed;
		// update package tree.
		// TODO(gri): The directory tree may be temporarily out-of-sync.
		//            Consider keeping separate time stamps so the web-
		//            page can indicate this discrepancy.
		fsTree.set(newDirectory(*goroot, maxDirDepth))
		fallthrough
	case 1:
		// sync failed because no files changed;
		// don't change the package tree
		syncDelay.set(*syncMin) //  revert to regular sync schedule
	default:
		// sync failed because of an error - back off exponentially, but try at least once a day
		syncDelay.backoff(24 * 60)
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
	return http.HandlerFunc(func(c *http.Conn, req *http.Request) {
		log.Stderrf("%s\t%s", c.RemoteAddr, req.URL)
		h.ServeHTTP(c, req)
	})
}


func remoteSearch(query string) (res *http.Response, err os.Error) {
	search := "/search?f=text&q=" + http.URLEscape(query)

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
	for _, addr := range addrs {
		url := "http://" + addr + search
		res, _, err = http.Get(url)
		if err == nil && res.StatusCode == http.StatusOK {
			break
		}
	}

	if err == nil && res.StatusCode != http.StatusOK {
		err = os.NewError(res.Status)
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

	// Check usage: either server and no args, or command line and args
	if (*httpAddr != "") != (flag.NArg() == 0) {
		usage()
	}

	if *tabwidth < 0 {
		log.Exitf("negative tabwidth %d", *tabwidth)
	}

	initHandlers()
	readTemplates()

	if *httpAddr != "" {
		// HTTP server mode.
		var handler http.Handler = http.DefaultServeMux
		if *verbose {
			log.Stderrf("Go Documentation Server\n")
			log.Stderrf("version = %s\n", runtime.Version())
			log.Stderrf("address = %s\n", *httpAddr)
			log.Stderrf("goroot = %s\n", *goroot)
			log.Stderrf("tabwidth = %d\n", *tabwidth)
			if !fsMap.IsEmpty() {
				log.Stderr("user-defined mapping:")
				fsMap.Fprint(os.Stderr)
			}
			handler = loggingHandler(handler)
		}

		registerPublicHandlers(http.DefaultServeMux)
		if *syncCmd != "" {
			http.Handle("/debug/sync", http.HandlerFunc(dosync))
		}

		// Initialize directory tree with corresponding timestamp.
		// Do it in two steps:
		// 1) set timestamp right away so that the indexer is kicked on
		fsTree.set(nil)
		// 2) compute initial directory tree in a goroutine so that launch is quick
		go func() { fsTree.set(newDirectory(*goroot, maxDirDepth)) }()

		// Start sync goroutine, if enabled.
		if *syncCmd != "" && *syncMin > 0 {
			syncDelay.set(*syncMin) // initial sync delay
			go func() {
				for {
					dosync(nil, nil)
					delay, _ := syncDelay.get()
					if *verbose {
						log.Stderrf("next sync in %dmin", delay.(int))
					}
					time.Sleep(int64(delay.(int)) * 60e9)
				}
			}()
		}

		// Start indexing goroutine.
		go indexer()

		// Start http server.
		if err := http.ListenAndServe(*httpAddr, handler); err != nil {
			log.Exitf("ListenAndServe %s: %v", *httpAddr, err)
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
				log.Exitf("remoteSearch: %s", err)
			}
			io.Copy(os.Stdout, res.Body)
		}
		return
	}

	// determine paths
	path := flag.Arg(0)
	if len(path) > 0 && path[0] == '.' {
		// assume cwd; don't assume -goroot
		cwd, _ := os.Getwd() // ignore errors
		path = pathutil.Join(cwd, path)
	}
	relpath := path
	abspath := path
	if len(path) > 0 && path[0] != '/' {
		abspath = absolutePath(path, pkgHandler.fsRoot)
	} else {
		relpath = relativePath(path)
	}

	var mode PageInfoMode
	if *srcMode {
		// only filter exports if we don't have explicit command-line filter arguments
		if flag.NArg() == 1 {
			mode |= exportsOnly
		}
	} else {
		mode = exportsOnly | genDoc
	}
	// TODO(gri): Provide a mechanism (flag?) to select a package
	//            if there are multiple packages in a directory.
	info := pkgHandler.getPageInfo(abspath, relpath, "", mode|tryMode)

	if info.PAst == nil && info.PDoc == nil && info.Dirs == nil {
		// try again, this time assume it's a command
		if len(path) > 0 && path[0] != '/' {
			abspath = absolutePath(path, cmdHandler.fsRoot)
		}
		info = cmdHandler.getPageInfo(abspath, relpath, "", mode)
	}

	// If we have more than one argument, use the remaining arguments for filtering
	if flag.NArg() > 1 {
		args := flag.Args()[1:]
		rx := makeRx(args)
		if rx == nil {
			log.Exitf("illegal regular expression from %v", args)
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
				writeAny(os.Stdout, d, *html)
				fmt.Println()
			}
			return

		case info.PDoc != nil:
			info.PDoc.Filter(filter)
		}
	}

	if err := packageText.Execute(info, os.Stdout); err != nil {
		log.Stderrf("packageText.Execute: %s", err)
	}
}
