package analysisflags

import (
	"flag"
	"fmt"
	"log"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
)

const help = `PROGNAME is a tool for static analysis of Go programs.

PROGNAME examines Go source code and reports suspicious constructs,
such as Printf calls whose arguments do not align with the format
string. It uses heuristics that do not guarantee all reports are
genuine problems, but it can find errors not caught by the compilers.
`

// Help implements the help subcommand for a multichecker or vet-lite
// style command. The optional args specify the analyzers to describe.
// Help calls log.Fatal if no such analyzer exists.
func Help(progname string, analyzers []*analysis.Analyzer, args []string) {
	// No args: show summary of all analyzers.
	if len(args) == 0 {
		fmt.Println(strings.Replace(help, "PROGNAME", progname, -1))
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
