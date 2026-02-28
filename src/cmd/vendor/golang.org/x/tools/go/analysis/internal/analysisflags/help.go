// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysisflags

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
)

const help = `PROGNAME is a tool for static analysis of Go programs.

PROGNAME examines Go source code and reports diagnostics for
suspicious constructs or opportunities for improvement.
Diagnostics may include suggested fixes.

An example of a suspicious construct is a Printf call whose arguments
do not align with the format string. Analyzers may use heuristics that
do not guarantee all reports are genuine problems, but can find
mistakes not caught by the compiler.

An example of an opportunity for improvement is a loop over
strings.Split(doc, "\n"), which may be replaced by a loop over the
strings.SplitSeq iterator, avoiding an array allocation.
Diagnostics in such cases may report non-problems,
but should carry fixes that may be safely applied.

For analyzers of the first kind, use "go vet -vettool=PROGRAM"
to run the tool and report diagnostics.

For analyzers of the second kind, use "go fix -fixtool=PROGRAM"
to run the tool and apply the fixes it suggests.
`

// Help implements the help subcommand for a multichecker or unitchecker
// style command. The optional args specify the analyzers to describe.
// Help calls log.Fatal if no such analyzer exists.
func Help(progname string, analyzers []*analysis.Analyzer, args []string) {
	// No args: show summary of all analyzers.
	if len(args) == 0 {
		fmt.Println(strings.ReplaceAll(help, "PROGNAME", progname))
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
		fmt.Println("To select specific analyzers, use the -NAME flag for each one,")
		fmt.Println(" or -NAME=false to run all analyzers not explicitly disabled.")

		// Show only the core command-line flags.
		fmt.Println("\nCore flags:")
		fmt.Println()
		fs := flag.NewFlagSet("", flag.ExitOnError)
		flag.VisitAll(func(f *flag.Flag) {
			if !strings.Contains(f.Name, ".") {
				fs.Var(f.Value, f.Name, f.Usage)
			}
		})
		fs.SetOutput(os.Stdout)
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
				fs.SetOutput(os.Stdout)
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
