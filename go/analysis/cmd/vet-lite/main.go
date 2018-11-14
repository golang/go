// The vet-lite command is a driver for static checkers conforming to
// the golang.org/x/tools/go/analysis API. It must be run by go vet:
//
//   $ go vet -vettool=$(which vet-lite)
//
// For a checker also capable of running standalone, use multichecker.
package main

import (
	"flag"
	"fmt"
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
	// These flags, plus the shims in analysisflags, enable
	// existing scripts that run vet to continue to work.
	//
	// Legacy vet had the concept of "experimental" checkers. There
	// was exactly one, shadow, and it had to be explicitly enabled
	// by the -shadow flag, which would of course disable all the
	// other tristate flags, requiring the -all flag to reenable them.
	// (By itself, -all did not enable all checkers.)
	// The -all flag is no longer needed, so it is a no-op.
	//
	// The shadow analyzer has been removed from the suite,
	// but can be run using these additional commands:
	//   $ go install golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow
	//   $ go vet -vettool=$(which shadow)
	// Alternatively, one could build a multichecker containing all
	// the desired checks (vet's suite + shadow) and run it in a
	// single "go vet" command.
	for _, name := range []string{"source", "v", "all"} {
		_ = flag.Bool(name, false, "no effect (deprecated)")
	}
	_ = flag.String("tags", "", "no effect (deprecated)")

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, `Usage of vet:
	vet unit.cfg		# execute analysis specified by config file
	vet help		# general help
	vet help name		# help on specific analyzer and its flags`)
		flag.PrintDefaults()
		os.Exit(1)
	}

	analyzers = analysisflags.Parse(analyzers, true)

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
	}
	if args[0] == "help" {
		analysisflags.Help("vet", analyzers, args[1:])
		os.Exit(0)
	}
	if len(args) != 1 || !strings.HasSuffix(args[0], ".cfg") {
		log.Fatalf("invalid command: want .cfg file (this reduced version of vet is intended to be run only by the 'go vet' command)")
	}

	unitchecker.Main(args[0], analyzers)
}
