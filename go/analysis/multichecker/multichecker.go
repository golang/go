// Package multichecker defines the main function for an analysis driver
// with several analyzers. This package makes it easy for anyone to build
// an analysis tool containing just the analyzers they need.
package multichecker

import (
	"flag"
	"log"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/checker"
)

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

	if err := checker.Run(flag.Args(), analyzers); err != nil {
		log.Fatal(err)
	}
}
