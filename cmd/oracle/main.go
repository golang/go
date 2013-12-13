// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// oracle: a tool for answering questions about Go source code.
// http://golang.org/s/oracle-design
// http://golang.org/s/oracle-user-manual
//
// Run with -help flag or help subcommand for usage information.
//
package main

import (
	"bufio"
	"encoding/json"
	"encoding/xml"
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

var ptalogFlag = flag.String("ptalog", "",
	"Location of the points-to analysis log file, or empty to disable logging.")

var formatFlag = flag.String("format", "plain", "Output format.  One of {plain,json,xml}.")

// TODO(adonovan): flip this flag after PTA presolver is implemented.
var reflectFlag = flag.Bool("reflect", false, "Analyze reflection soundly (slow).")

const useHelp = "Run 'oracle -help' for more information.\n"

const helpMessage = `Go source code oracle.
Usage: oracle [<flag> ...] <mode> <args> ...

The -format flag controls the output format:
	plain	an editor-friendly format in which every line of output
		is of the form "pos: text", where pos is "-" if unknown.
	json	structured data in JSON syntax.
	xml	structured data in XML syntax.

The -pos flag is required in all modes except 'callgraph'.

The mode argument determines the query to perform:

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
% oracle -pos=src/code.google.com/p/go.tools/cmd/oracle/main.go:#530 describe \
   code.google.com/p/go.tools/cmd/oracle

Print the callgraph of the trivial web-server in JSON format:
% oracle -format=json src/pkg/net/http/triv.go callgraph
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

func printHelp() {
	fmt.Fprintln(os.Stderr, helpMessage)
	fmt.Fprintln(os.Stderr, "Flags:")
	flag.PrintDefaults()
}

func main() {
	// Don't print full help unless -help was requested.
	// Just gently remind users that it's there.
	flag.Usage = func() { fmt.Fprint(os.Stderr, useHelp) }
	flag.CommandLine.Init(os.Args[0], flag.ContinueOnError) // hack
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		// (err has already been printed)
		if err == flag.ErrHelp {
			printHelp()
		}
		os.Exit(2)
	}

	args := flag.Args()
	if len(args) == 0 || args[0] == "" {
		fmt.Fprint(os.Stderr, "Error: a mode argument is required.\n"+useHelp)
		os.Exit(2)
	}

	mode := args[0]
	args = args[1:]
	if mode == "help" {
		printHelp()
		os.Exit(2)
	}

	if len(args) == 0 && mode != "what" {
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
	switch *formatFlag {
	case "json", "plain", "xml":
		// ok
	default:
		fmt.Fprintf(os.Stderr, "Error: illegal -format value: %q.\n"+useHelp, *formatFlag)
		os.Exit(2)
	}

	// Ask the oracle.
	res, err := oracle.Query(args, mode, *posFlag, ptalog, &build.Default, *reflectFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s.\n", err)
		os.Exit(1)
	}

	// Print the result.
	switch *formatFlag {
	case "json":
		b, err := json.MarshalIndent(res.Serial(), "", "\t")
		if err != nil {
			fmt.Fprintf(os.Stderr, "JSON error: %s.\n", err)
			os.Exit(1)
		}
		os.Stdout.Write(b)

	case "xml":
		b, err := xml.MarshalIndent(res.Serial(), "", "\t")
		if err != nil {
			fmt.Fprintf(os.Stderr, "XML error: %s.\n", err)
			os.Exit(1)
		}
		os.Stdout.Write(b)

	case "plain":
		res.WriteTo(os.Stdout)
	}
}
