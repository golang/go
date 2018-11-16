// The vet command is a driver for static checkers conforming to
// the golang.org/x/tools/go/analysis API. Run it using 'go vet'.
//
// For a tool capable of running standalone, use a multichecker-based
// tool such as golang.org/x/tools/go/analysis/cmd/vet.
package main

import (
	"golang.org/x/tools/go/analysis/unitchecker"

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

// Legacy vet had the concept of "experimental" checkers.

// There was exactly one, shadow, and it had to be explicitly
// enabled by the -shadow flag, which would of course disable
// all the other tristate flags, requiring the -all flag (which
// is now a no-op) to reenable them.
//
// The shadow analyzer has been removed from the suite,
// but can be run using these additional commands:
//   $ go install golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow
//   $ go vet -vettool=$(which shadow)
// Alternatively, one could build a multichecker containing all
// the desired checks (vet's suite + shadow) and run it in a
// single "go vet" command.

func main() {
	unitchecker.Main(
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
		printf.Analyzer,
		shift.Analyzer,
		stdmethods.Analyzer,
		structtag.Analyzer,
		tests.Analyzer,
		unmarshal.Analyzer,
		unreachable.Analyzer,
		unsafeptr.Analyzer,
		unusedresult.Analyzer,
	)
}
