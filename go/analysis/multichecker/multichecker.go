// Package multichecker defines the main function for an analysis driver
// with several analyzers. This package makes it easy for anyone to build
// an analysis tool containing just the analyzers they need.
package multichecker

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/checker"
)

const usage = `Analyze is a tool for static analysis of Go programs.

Usage: analyze [-flag] [package]
`

func Main(analyzers ...*analysis.Analyzer) {
	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}

	checker.RegisterFlags()

	// Connect each analysis flag to the command line as --analysis.flag.
	enabled := make(map[*analysis.Analyzer]*bool)
	for _, a := range analyzers {
		prefix := a.Name + "."

		// Add --foo.enable flag.
		enable := new(bool)
		flag.BoolVar(enable, prefix+"enable", false, "enable only "+a.Name+" analysis")
		enabled[a] = enable

		a.Flags.VisitAll(func(f *flag.Flag) {
			flag.Var(f.Value, prefix+f.Name, f.Usage)
		})
	}

	flag.Parse() // (ExitOnError)

	// If any --foo.enable flag is set,
	// run only those analyzers.
	var keep []*analysis.Analyzer
	for _, a := range analyzers {
		if *enabled[a] {
			keep = append(keep, a)
		}
	}
	if keep != nil {
		analyzers = keep
	}

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, usage)
		fmt.Fprintln(os.Stderr, `Run 'analyze help' for more detail,
 or 'analyze help name' for details and flags of a specific analyzer.`)
		os.Exit(1)
	}

	if args[0] == "help" {
		help(analyzers, args[1:])
		os.Exit(0)
	}

	if err := checker.Run(args, analyzers); err != nil {
		log.Fatal(err)
	}
}

func help(analyzers []*analysis.Analyzer, args []string) {
	// No args: show summary of all analyzers.
	if len(args) == 0 {
		fmt.Println(usage)
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

		fmt.Println("\nTo see details and flags of a specific analyzer, run 'analyze help name'.")

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
