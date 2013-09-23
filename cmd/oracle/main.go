// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// oracle: a tool for answering questions about Go source code.
// http://golang.org/s/oracle-design
// http://golang.org/s/oracle-user-manual
//
// Run with -help for usage information.
//
// TODO(adonovan): perhaps -mode should be an args[1] verb, e.g. 'oracle callgraph ...'
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

	"code.google.com/p/go.tools/importer"
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

// TODO(adonovan): eliminate or flip this flag after PTA presolver is implemented.
var reflectFlag = flag.Bool("reflect", true, "Analyze reflection soundly (slow).")

const useHelp = "Run 'oracle -help' for more information.\n"

const helpMessage = `Go source code oracle.
Usage: oracle [<flag> ...] <args> ...

The -format flag controls the output format:
	plain	an editor-friendly format in which every line of output
		is of the form "pos: text", where pos is "-" if unknown.
	json	structured data in JSON syntax.

The -pos flag is required in all modes except 'callgraph'.

The -mode flag determines the query to perform:
	callees	  	show possible targets of selected function call
	callers	  	show possible callers of selected function
	callgraph 	show complete callgraph of program
	callstack 	show path from callgraph root to selected function
	describe  	describe selected syntax: definition, methods, etc
	freevars  	show free variables of selection
	implements	show 'implements' relation for selected package
	peers     	show send/receive corresponding to selected channel op
	referrers 	show all refs to entity denoted by selected identifier

The user manual is available here:  http://golang.org/s/oracle-user-manual

Examples:

Describe the syntax at offset 530 in this file (an import spec):
% oracle -mode=describe -pos=src/code.google.com/p/go.tools/cmd/oracle/main.go:#530 \
   code.google.com/p/go.tools/cmd/oracle

Print the callgraph of the trivial web-server in JSON format:
% oracle -mode=callgraph -format=json src/pkg/net/http/triv.go
` + importer.InitialPackagesUsage

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
	// Don't print full help unless -help was requested.
	// Just gently remind users that it's there.
	flag.Usage = func() { fmt.Fprint(os.Stderr, useHelp) }
	flag.CommandLine.Init(os.Args[0], flag.ContinueOnError) // hack
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		// (err has already been printed)
		if err == flag.ErrHelp {
			fmt.Println(helpMessage)
			fmt.Println("Flags:")
			flag.PrintDefaults()
		}
		os.Exit(2)
	}
	args := flag.Args()

	if len(args) == 0 {
		fmt.Fprint(os.Stderr, "Error: no package arguments.\n"+useHelp)
		os.Exit(2)
	}

	// Set up points-to analysis log file.
	var ptalog io.Writer
	if *ptalogFlag != "" {
		if f, err := os.Create(*ptalogFlag); err != nil {
			log.Fatalf("Failed to create PTA log file: %s", err)
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
		fmt.Fprintf(os.Stderr, "Error: illegal -format value: %q\n"+useHelp, *formatFlag)
		os.Exit(2)
	}

	// -mode flag
	if *modeFlag == "" {
		fmt.Fprintf(os.Stderr, "Error: a query -mode is required.\n"+useHelp)
		os.Exit(2)
	}

	// Ask the oracle.
	res, err := oracle.Query(args, *modeFlag, *posFlag, ptalog, &build.Default, *reflectFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n"+useHelp, err)
		os.Exit(1)
	}

	// Print the result.
	switch *formatFlag {
	case "json":
		b, err := json.Marshal(res)
		if err != nil {
			fmt.Fprintf(os.Stderr, "JSON error: %s\n", err)
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
