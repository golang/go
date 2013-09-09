// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// oracle: a tool for answering questions about Go source code.
//
// With -format=plain, the oracle prints query results to the standard
// output in an editor-friendly format in which every line of output
// is of the form "pos: text", where pos = "-" if unknown.
//
// With -format=json, the oracle prints structured data in JSON syntax.
//
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/build"
	"io"
	"log"
	"os"
	"runtime"
	"runtime/pprof"

	"code.google.com/p/go.tools/oracle"
)

var posFlag = flag.String("pos", "",
	"Filename and byte offset or extent of a syntax element about which to query, "+
		"e.g. foo.go:#123,#456, bar.go:#123.")

var modeFlag = flag.String("mode", "",
	"Mode of query to perform: e.g. callers, describe, etc.")

var ptalogFlag = flag.String("ptalog", "",
	"Location of the points-to analysis log file, or empty to disable logging.")

var formatFlag = flag.String("format", "plain", "Output format: 'plain' or 'json'.")

const usage = `Go source code oracle.
Usage: oracle [<flag> ...] [<arg> ...]
Use -help flag to display options.

The -mode flag is required; the -pos flag is required in most modes.

Examples:

Describe the syntax at offset 532 in this file (an import spec):
% oracle -mode=describe -pos=src/code.google.com/p/go.tools/cmd/oracle/main.go:#532 \
   code.google.com/p/go.tools/cmd/oracle

Print the callgraph of the trivial web-server in JSON format:
% oracle -mode=callgraph -format=json src/pkg/net/http/triv.go

`

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

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

	// -format flag
	if *formatFlag != "json" && *formatFlag != "plain" {
		fmt.Fprintf(os.Stderr, "Error: illegal -format value: %q", *formatFlag)
		os.Exit(1)
	}

	// Ask the oracle.
	res, err := oracle.Query(args, *modeFlag, *posFlag, ptalog, &build.Default)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	// Print the result.
	switch *formatFlag {
	case "json":
		b, err := json.Marshal(res)
		if err != nil {
			fmt.Fprintf(os.Stderr, "JSON error: %s\n", err.Error())
			os.Exit(1)
		}
		var buf bytes.Buffer
		if err := json.Indent(&buf, b, "", "\t"); err != nil {
			fmt.Fprintf(os.Stderr, "json.Indent failed: %s", err)
			os.Exit(1)
		}
		os.Stdout.Write(buf.Bytes())

	case "plain":
		res.WriteTo(os.Stdout)
	}
}
