// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Fix is a tool executed by "go fix" to update Go programs that use old
features of the language and library and rewrite them to use newer
ones. After you update to a new Go release, fix helps make the
necessary changes to your programs.

See the documentation for "go fix" for how to run this command.
You can provide an alternative tool using "go fix -fixtool=..."

Run "go tool fix help" to see the list of analyzers supported by this
program.

See [golang.org/x/tools/go/analysis] for information on how to write
an analyzer that can suggest fixes.
*/
package main

import (
	"cmd/internal/objabi"
	"cmd/internal/telemetry/counter"
	"slices"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildtag"
	"golang.org/x/tools/go/analysis/passes/hostport"
	"golang.org/x/tools/go/analysis/passes/inline"
	"golang.org/x/tools/go/analysis/passes/modernize"
	"golang.org/x/tools/go/analysis/unitchecker"
)

func main() {
	// Keep consistent with cmd/vet/main.go!
	counter.Open()
	objabi.AddVersionFlag()
	counter.Inc("fix/invocations")

	unitchecker.Main(suite...) // (never returns)
}

// The fix suite analyzers produce fixes are unambiguously safe to apply,
// even if the diagnostics might not describe actual problems.
var suite = slices.Concat(
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
