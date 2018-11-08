// The vet-lite command is a driver for static checkers conforming to
// the golang.org/x/tools/go/analysis API. It must be run by go vet:
//
//   $ go vet -vettool=$(which vet-lite)
//
// For a checker also capable of running standalone, use multichecker.
package main

import (
	"flag"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/go/analysis/internal/unitchecker"

	"golang.org/x/tools/go/analysis/passes/asmdecl"
	"golang.org/x/tools/go/analysis/passes/assign"
	"golang.org/x/tools/go/analysis/passes/atomic"
	"golang.org/x/tools/go/analysis/passes/bools"
	"golang.org/x/tools/go/analysis/passes/buildtag"
	"golang.org/x/tools/go/analysis/passes/cgocall"
	"golang.org/x/tools/go/analysis/passes/composite"
	"golang.org/x/tools/go/analysis/passes/copylock"
	"golang.org/x/tools/go/analysis/passes/httpresponse"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/go/analysis/passes/lostcancel"
	"golang.org/x/tools/go/analysis/passes/nilfunc"
	"golang.org/x/tools/go/analysis/passes/pkgfact"
	"golang.org/x/tools/go/analysis/passes/printf"
	"golang.org/x/tools/go/analysis/passes/shift"
	"golang.org/x/tools/go/analysis/passes/stdmethods"
	"golang.org/x/tools/go/analysis/passes/structtag"
	"golang.org/x/tools/go/analysis/passes/tests"
	"golang.org/x/tools/go/analysis/passes/unmarshal"
	"golang.org/x/tools/go/analysis/passes/unreachable"
	"golang.org/x/tools/go/analysis/passes/unsafeptr"
	"golang.org/x/tools/go/analysis/passes/unusedresult"
)

var analyzers = []*analysis.Analyzer{
	// For now, just the traditional vet suite:
	asmdecl.Analyzer,
	assign.Analyzer,
	atomic.Analyzer,
	bools.Analyzer,
	buildtag.Analyzer,
	cgocall.Analyzer,
	composite.Analyzer,
	copylock.Analyzer,
	httpresponse.Analyzer,
	loopclosure.Analyzer,
	lostcancel.Analyzer,
	nilfunc.Analyzer,
	pkgfact.Analyzer,
	printf.Analyzer,
	// shadow.Analyzer, // experimental; not enabled by default
	shift.Analyzer,
	stdmethods.Analyzer,
	structtag.Analyzer,
	tests.Analyzer,
	unmarshal.Analyzer,
	unreachable.Analyzer,
	unsafeptr.Analyzer,
	unusedresult.Analyzer,
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("vet: ")

	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}

	// Flags for legacy vet compatibility.
	//
	// These flags, plus the shims in analysisflags, enable all
	// existing scripts that run vet to continue to work.
	//
	// We still need to deal with legacy vet's "experimental"
	// checkers. In vet there is exactly one such checker, shadow,
	// and it must be enabled explicitly with the -shadow flag, but
	// of course setting it disables all the other tristate flags,
	// requiring the -all flag to reenable them.
	//
	// I don't believe this feature carries its weight. I propose we
	// simply skip shadow for now; the few users that want it can
	// run "go vet -vettool=..." using a vet tool that includes
	// shadow, either as an additional step, with a shadow
	// "singlechecker", or in place of the regular vet step on a
	// multichecker with a hand-picked suite of checkers.
	// Or, we could improve the shadow checker to the point where it
	// need not be experimental.
	for _, name := range []string{"source", "v", "all"} {
		flag.Var(warnBoolFlag(name), name, "no effect (deprecated)")
	}

	flag.Usage = func() {
		analysisflags.Help("vet", analyzers, nil)
		os.Exit(1)
	}

	analyzers = analysisflags.Parse(analyzers, true)

	args := flag.Args()
	if len(args) != 1 || !strings.HasSuffix(args[0], ".cfg") {
		log.Fatalf("invalid command: want .cfg file (this reduced version of vet is intended to be run only by the 'go vet' command)")
	}

	unitchecker.Main(args[0], analyzers)
}

type warnBoolFlag string

func (f warnBoolFlag) Set(s string) error {
	log.Printf("warning: deprecated flag -%s has no effect", string(f))
	return nil
}
func (f warnBoolFlag) IsBoolFlag() bool { return true }
func (f warnBoolFlag) String() string   { return "false" }
