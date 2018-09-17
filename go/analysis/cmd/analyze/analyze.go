// The analyze command is a static checker for Go programs, similar to
// vet, but with pluggable analyzers defined using the analysis
// interface, and using the go/packages API to load packages in any
// build system.
//
// Each analysis flag name is preceded by the analysis name: --analysis.flag.
// In addition, the --analysis.enabled flag controls whether the
// diagnostics of that analysis are displayed. (A disabled analysis may yet
// be run if it is required by some other analysis that is enabled.)
package main

import (
	"log"

	"golang.org/x/tools/go/analysis/multichecker"

	// analysis plug-ins
	"golang.org/x/tools/go/analysis/passes/findcall"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("analyze: ")

	multichecker.Main(findcall.Analyzer)
}
