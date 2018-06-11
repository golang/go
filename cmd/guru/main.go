// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// guru: a tool for answering questions about Go source code.
//
//    http://golang.org/s/using-guru
//
// Run with -help flag or help subcommand for usage information.
//
package main // import "golang.org/x/tools/cmd/guru"

import (
	"bufio"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"

	"golang.org/x/tools/go/buildutil"
)

// flags
var (
	modifiedFlag   = flag.Bool("modified", false, "read archive of modified files from standard input")
	scopeFlag      = flag.String("scope", "", "comma-separated list of `packages` the analysis should be limited to")
	ptalogFlag     = flag.String("ptalog", "", "write points-to analysis log to `file`")
	jsonFlag       = flag.Bool("json", false, "emit output in JSON format")
	reflectFlag    = flag.Bool("reflect", false, "analyze reflection soundly (slow)")
	cpuprofileFlag = flag.String("cpuprofile", "", "write CPU profile to `file`")
)

func init() {
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)

	// gccgo does not provide a GOROOT with standard library sources.
	// If we have one in the environment, force gc mode.
	if build.Default.Compiler == "gccgo" {
		if _, err := os.Stat(filepath.Join(runtime.GOROOT(), "src", "runtime", "runtime.go")); err == nil {
			build.Default.Compiler = "gc"
		}
	}
}

const useHelp = "Run 'guru -help' for more information.\n"

const helpMessage = `Go source code guru.
Usage: guru [flags] <mode> <position>

The mode argument determines the query to perform:

	callees	  	show possible targets of selected function call
	callers	  	show possible callers of selected function
	callstack 	show path from callgraph root to selected function
	definition	show declaration of selected identifier
	describe  	describe selected syntax: definition, methods, etc
	freevars  	show free variables of selection
	implements	show 'implements' relation for selected type or method
	peers     	show send/receive corresponding to selected channel op
	pointsto	show variables the selected pointer may point to
	referrers 	show all refs to entity denoted by selected identifier
	what		show basic information about the selected syntax node
	whicherrs	show possible values of the selected error variable

The position argument specifies the filename and byte offset (or range)
of the syntax element to query.  For example:

	foo.go:#123,#128
	bar.go:#123

The -json flag causes guru to emit output in JSON format;
	golang.org/x/tools/cmd/guru/serial defines its schema.
	Otherwise, the output is in an editor-friendly format in which
	every line has the form "pos: text", where pos is "-" if unknown.

The -modified flag causes guru to read an archive from standard input.
	Files in this archive will be used in preference to those in
	the file system.  In this way, a text editor may supply guru
	with the contents of its unsaved buffers.  Each archive entry
	consists of the file name, a newline, the decimal file size,
	another newline, and the contents of the file.

The -scope flag restricts analysis to the specified packages.
	Its value is a comma-separated list of patterns of these forms:
		golang.org/x/tools/cmd/guru     # a single package
		golang.org/x/tools/...          # all packages beneath dir
		...                             # the entire workspace.
	A pattern preceded by '-' is negative, so the scope
		encoding/...,-encoding/xml
	matches all encoding packages except encoding/xml.

User manual: http://golang.org/s/using-guru

Example: describe syntax at offset 530 in this file (an import spec):

  $ guru describe src/golang.org/x/tools/cmd/guru/main.go:#530
`

func printHelp() {
	fmt.Fprintln(os.Stderr, helpMessage)
	fmt.Fprintln(os.Stderr, "Flags:")
	flag.PrintDefaults()
}

func main() {
	log.SetPrefix("guru: ")
	log.SetFlags(0)

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
	if len(args) != 2 {
		flag.Usage()
		os.Exit(2)
	}
	mode, posn := args[0], args[1]

	if mode == "help" {
		printHelp()
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
				if err := buf.Flush(); err != nil {
					log.Printf("flush: %s", err)
				}
				if err := f.Close(); err != nil {
					log.Printf("close: %s", err)
				}
			}()
		}
	}

	// Profiling support.
	if *cpuprofileFlag != "" {
		f, err := os.Create(*cpuprofileFlag)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	ctxt := &build.Default

	// If there were modified files,
	// read them from the standard input and
	// overlay them on the build context.
	if *modifiedFlag {
		modified, err := buildutil.ParseOverlayArchive(os.Stdin)
		if err != nil {
			log.Fatal(err)
		}

		// All I/O done by guru needs to consult the modified map.
		// The ReadFile done by referrers does,
		// but the loader's cgo preprocessing currently does not.

		if len(modified) > 0 {
			ctxt = buildutil.OverlayContext(ctxt, modified)
		}
	}

	var outputMu sync.Mutex
	output := func(fset *token.FileSet, qr QueryResult) {
		outputMu.Lock()
		defer outputMu.Unlock()
		if *jsonFlag {
			// JSON output
			fmt.Printf("%s\n", qr.JSON(fset))
		} else {
			// plain output
			printf := func(pos interface{}, format string, args ...interface{}) {
				fprintf(os.Stdout, fset, pos, format, args...)
			}
			qr.PrintPlain(printf)
		}
	}

	// Avoid corner case of split("").
	var scope []string
	if *scopeFlag != "" {
		scope = strings.Split(*scopeFlag, ",")
	}

	// Ask the guru.
	query := Query{
		Pos:        posn,
		Build:      ctxt,
		Scope:      scope,
		PTALog:     ptalog,
		Reflection: *reflectFlag,
		Output:     output,
	}

	if err := Run(mode, &query); err != nil {
		log.Fatal(err)
	}
}
