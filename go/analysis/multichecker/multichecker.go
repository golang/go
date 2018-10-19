// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package multichecker defines the main function for an analysis driver
// with several analyzers. This package makes it easy for anyone to build
// an analysis tool containing just the analyzers they need.
package multichecker

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/go/analysis/internal/checker"
)

// TODO(adonovan): document (and verify) the exit codes:
// "Vet's exit code is 2 for erroneous invocation of the tool, 1 if a
// problem was reported, and 0 otherwise. Note that the tool does not
// check every possible problem and depends on unreliable heuristics
// so it should be used as guidance only, not as a firm indicator of
// program correctness."

const usage = `PROGNAME is a tool for static analysis of Go programs.

PROGNAME examines Go source code and reports suspicious constructs, such as Printf
calls whose arguments do not align with the format string. It uses heuristics
that do not guarantee all reports are genuine problems, but it can find errors
not caught by the compilers.

Usage: PROGNAME [-flag] [package]
`

func Main(analyzers ...*analysis.Analyzer) {
	progname := filepath.Base(os.Args[0])
	log.SetFlags(0)
	log.SetPrefix(filepath.Base(os.Args[0]) + ": ") // e.g. "vet: "

	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}

	checker.RegisterFlags()

	analyzers = analysisflags.Parse(analyzers, true)

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, strings.Replace(usage, "PROGNAME", progname, -1))
		fmt.Fprintf(os.Stderr, "Run '%[1]s help' for more detail,\n"+
			" or '%[1]s help name' for details and flags of a specific analyzer.\n",
			progname)
		os.Exit(1)
	}

	if args[0] == "help" {
		help(progname, analyzers, args[1:])
		os.Exit(0)
	}

	if err := checker.Run(args, analyzers); err != nil {
		log.Fatal(err)
	}
}

func help(progname string, analyzers []*analysis.Analyzer, args []string) {
	// No args: show summary of all analyzers.
	if len(args) == 0 {
		fmt.Println(strings.Replace(usage, "PROGNAME", progname, -1))
		fmt.Println("Registered analyzers:")
		fmt.Println()
		sort.Slice(analyzers, func(i, j int) bool {
			return analyzers[i].Name < analyzers[j].Name
		})
		for _, a := range analyzers {
			title := strings.Split(a.Doc, "\n\n")[0]
			fmt.Printf("    %-12s %s\n", a.Name, title)
		}
		fmt.Println("\nBy default all analyzers are run.")
		fmt.Println("To select specific analyzers, use the -NAME.enable flag for each one.")

		// Show only the core command-line flags.
		fmt.Println("\nCore flags:")
		fmt.Println()
		fs := flag.NewFlagSet("", flag.ExitOnError)
		flag.VisitAll(func(f *flag.Flag) {
			if !strings.Contains(f.Name, ".") {
				fs.Var(f.Value, f.Name, f.Usage)
			}
		})
		fs.PrintDefaults()

		fmt.Printf("\nTo see details and flags of a specific analyzer, run '%s help name'.\n", progname)

		return
	}

	// Show help on specific analyzer(s).
outer:
	for _, arg := range args {
		for _, a := range analyzers {
			if a.Name == arg {
				paras := strings.Split(a.Doc, "\n\n")
				title := paras[0]
				fmt.Printf("%s: %s\n", a.Name, title)

				// Show only the flags relating to this analysis,
				// properly prefixed.
				first := true
				fs := flag.NewFlagSet(a.Name, flag.ExitOnError)
				a.Flags.VisitAll(func(f *flag.Flag) {
					if first {
						first = false
						fmt.Println("\nAnalyzer flags:")
						fmt.Println()
					}
					fs.Var(f.Value, a.Name+"."+f.Name, f.Usage)
				})
				fs.PrintDefaults()

				if len(paras) > 1 {
					fmt.Printf("\n%s\n", strings.Join(paras[1:], "\n\n"))
				}

				continue outer
			}
		}
		log.Fatalf("Analyzer %q not registered", arg)
	}
}
