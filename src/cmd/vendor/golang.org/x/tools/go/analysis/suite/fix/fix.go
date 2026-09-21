// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The fix package defines the suite of analyzers used by cmd/fix,
// the default analysis tool run by "go fix".
// Its behavior is equivalent to:
//
//	func main() { unitchecker.Main(fix.Suite...) }
//
// If you need a different suite, define your own tool
// and run "go vet -vettool=mytool".
package fix

import (
	"slices"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildtag"
	"golang.org/x/tools/go/analysis/passes/hostport"
	"golang.org/x/tools/go/analysis/passes/inline"
	"golang.org/x/tools/go/analysis/passes/modernize"
)

// Suite is the suite of analyzers run by cmd/fix.
//
// The fix suite analyzers produce fixes are unambiguously safe to apply,
// even if the diagnostics might not describe actual problems.
var Suite = slices.Concat(
	[]*analysis.Analyzer{
		buildtag.Analyzer,
		hostport.Analyzer,
		inline.Analyzer,
	},
	modernize.Suite,
	// TODO(adonovan): add any other vet analyzers whose fixes are always safe.
	// Candidates to audit: sigchanyzer, printf, assign, unreachable.
	// Many of staticcheck's analyzers would make good candidates
	//   (e.g. rewriting WriteString(fmt.Sprintf()) to Fprintf.)
	// Rejected:
	// - composites: some types (e.g. PointXY{1,2}) don't want field names.
	// - timeformat: flipping MM/DD is a behavior change, but the code
	//    could potentially be a workaround for another bug.
	// - stringintconv: offers two fixes, user input required to choose.
	// - fieldalignment: poor signal/noise; fix could be a regression.
)
