// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/		main landing page
//	http://godoc/doc/	serve from $GOROOT/doc - spec, mem, tutorial, etc.
//	http://godoc/src/	serve files from $GOROOT/src; .go gets pretty-printed
//	http://godoc/cmd/	serve documentation about commands (TODO)
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
	"flag"
	"fmt"
	"http"
	"io"
	"log"
	"os"
	"time"
)

var (
	// periodic sync
	syncCmd             = flag.String("sync", "", "sync command; disabled if empty")
	syncMin             = flag.Int("sync_minutes", 0, "sync interval in minutes; disabled if <= 0")
	syncDelay delayTime // actual sync delay in minutes; usually syncDelay == syncMin, but delay may back off exponentially

	// server control
	httpaddr = flag.String("http", "", "HTTP service address (e.g., ':6060')")

	// layout control
	html = flag.Bool("html", false, "print HTML in command-line mode")
)


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
	pid, err := os.ForkExec(bin, args, os.Environ(), goroot, fds)
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
		fsTree.set(newDirectory(".", maxDirDepth))
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
			"	godoc -http=:6060\n")
	flag.PrintDefaults()
	os.Exit(2)
}


func loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(c *http.Conn, req *http.Request) {
		log.Stderrf("%s\t%s", c.RemoteAddr, req.URL)
		h.ServeHTTP(c, req)
	})
}


func main() {
	flag.Usage = usage
	flag.Parse()

	// Check usage: either server and no args, or command line and args
	if (*httpaddr != "") != (flag.NArg() == 0) {
		usage()
	}

	if *tabwidth < 0 {
		log.Exitf("negative tabwidth %d", *tabwidth)
	}

	if err := os.Chdir(goroot); err != nil {
		log.Exitf("chdir %s: %v", goroot, err)
	}

	readTemplates()

	if *httpaddr != "" {
		// HTTP server mode.
		var handler http.Handler = http.DefaultServeMux
		if *verbose {
			log.Stderrf("Go Documentation Server\n")
			log.Stderrf("address = %s\n", *httpaddr)
			log.Stderrf("goroot = %s\n", goroot)
			log.Stderrf("cmdroot = %s\n", *cmdroot)
			log.Stderrf("pkgroot = %s\n", *pkgroot)
			log.Stderrf("tmplroot = %s\n", *tmplroot)
			log.Stderrf("tabwidth = %d\n", *tabwidth)
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
		go func() { fsTree.set(newDirectory(".", maxDirDepth)) }()

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

		// The server may have been restarted; always wait 1sec to
		// give the forking server a chance to shut down and release
		// the http port.
		// TODO(gri): Do we still need this?
		time.Sleep(1e9)

		// Start http server.
		if err := http.ListenAndServe(*httpaddr, handler); err != nil {
			log.Exitf("ListenAndServe %s: %v", *httpaddr, err)
		}
		return
	}

	// Command line mode.
	if *html {
		packageText = packageHTML
	}

	info := pkgHandler.getPageInfo(flag.Arg(0))

	if info.PDoc == nil && info.Dirs == nil {
		// try again, this time assume it's a command
		info = cmdHandler.getPageInfo(flag.Arg(0))
	}

	if info.PDoc != nil && flag.NArg() > 1 {
		args := flag.Args()
		info.PDoc.Filter(args[1:])
	}

	if err := packageText.Execute(info, os.Stdout); err != nil {
		log.Stderrf("packageText.Execute: %s", err)
	}
}
