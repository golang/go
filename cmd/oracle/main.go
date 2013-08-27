// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// oracle: a tool for answering questions about Go source code.
//
// Each query prints its results to the standard output in an
// editor-friendly format.  Currently this is just text in a generic
// compiler diagnostic format, but in future we could provide
// sexpr/json/python formats for the raw data so that editors can
// provide more sophisticated UIs.
//
// Every line of output is of the form "pos: text", where pos = "-" if unknown.
//
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"runtime/pprof"

	"code.google.com/p/go.tools/oracle"
)

// TODO(adonovan): use a format that permits spaces in filenames, and
// doesn't require shell quoting.
var posFlag = flag.String("pos", "",
	"Filename and offset or extent of a syntax element about which to query, "+
		"e.g. 'foo.go 123-456', 'bar.go 123'.")

var modeFlag = flag.String("mode", "",
	"Mode of query to perform: callers, callees, callstack, callgraph, describe.")

var ptalogFlag = flag.String("ptalog", "pta.log",
	"Location of the points-to analysis log file, or empty to disable logging.")

const usage = `Go source code oracle.
Usage: oracle [<flag> ...] [<file.go> ...] [<arg> ...]
Use -help flag to display options.

Examples:
% oracle -pos 'hello.go 123' hello.go
% oracle -pos 'hello.go 123 456' hello.go
`

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

// TODO(adonovan): the caller must---before go/build.init
// runs---specify CGO_ENABLED=0, which entails the "!cgo" go/build
// tag, preferring (dummy) Go to native C implementations of
// cgoLookupHost et al.

func init() {
	// If $GOMAXPROCS isn't set, use the full capacity of the machine.
	// For small machines, use at least 4 threads.
	if os.Getenv("GOMAXPROCS") == "" {
		n := runtime.NumCPU()
		if n < 4 {
			n = 4
		}
		runtime.GOMAXPROCS(n)
	}
}

func main() {
	flag.Parse()
	args := flag.Args()

	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	// Set up points-to analysis log file.
	var ptalog io.Writer
	if *ptalogFlag != "" {
		if f, err := os.Create(*ptalogFlag); err != nil {
			log.Fatalf(err.Error())
		} else {
			buf := bufio.NewWriter(f)
			ptalog = buf
			defer func() {
				buf.Flush()
				f.Close()
			}()
		}
	}

	// Profiling support.
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	if err := oracle.Main(args, *modeFlag, *posFlag, ptalog, os.Stdout, nil); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
}
