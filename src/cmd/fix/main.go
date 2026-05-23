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

	"golang.org/x/tools/go/analysis/suite/fix"
	"golang.org/x/tools/go/analysis/unitchecker"
)

func main() {
	// Keep consistent with cmd/vet/main.go!
	counter.Open()
	objabi.AddVersionFlag()
	counter.Inc("fix/invocations")

	unitchecker.Main(fix.Suite...) // (never returns)
}
