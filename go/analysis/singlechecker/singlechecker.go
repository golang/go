// Package singlechecker defines the main function for an analysis
// driver with only a single analysis.
// This package makes it easy for a provider of an analysis package to
// also provide a standalone tool that runs just that analysis.
//
// For example, if example.org/findbadness is an analysis package,
// all that is needed to define a standalone tool is a file,
// example.org/findbadness/cmd/findbadness/main.go, containing:
//
//      // The findbadness command runs an analysis.
// 	package main
//
// 	import (
// 		"example.org/findbadness"
// 		"golang.org/x/tools/go/analysis/singlechecker"
// 	)
//
// 	func main() { singlechecker.Main(findbadness.Analyzer) }
//
package singlechecker

import (
	"flag"
	"log"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/checker"
)

// Main is the main function for a checker command for a single analysis.
func Main(a *analysis.Analyzer) {
	log.SetFlags(0)
	log.SetPrefix(a.Name + ": ")

	checker.RegisterFlags()

	a.Flags.VisitAll(func(f *flag.Flag) {
		if flag.Lookup(f.Name) != nil {
			log.Printf("%s flag -%s would conflict with driver; skipping", a.Name, f.Name)
			return
		}
		flag.Var(f.Value, f.Name, f.Usage)
	})

	flag.Parse()

	if err := checker.Run(flag.Args(), []*analysis.Analyzer{a}); err != nil {
		log.Fatal(err)
	}
}
