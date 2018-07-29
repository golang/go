// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod fix

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/modload"
)

var cmdFix = &base.Command{
	UsageLine: "go mod fix",
	Short:     "make go.mod semantically consistent",
	Long: `
Fix updates go.mod to use canonical version identifiers and
to be semantically consistent. For example, consider this go.mod file:

	module M

	require (
		A v1
		B v1.0.0
		C v1.0.0
		D v1.2.3
		E dev
	)

	exclude D v1.2.3

First, fix rewrites non-canonical version identifiers to semver form, so
A's v1 becomes v1.0.0 and E's dev becomes the pseudo-version for the latest
commit on the dev branch, perhaps v0.0.0-20180523231146-b3f5c0f6e5f1.

Next, fix updates requirements to respect exclusions, so the requirement
on the excluded D v1.2.3 is updated to use the next available version of D,
perhaps D v1.2.4 or D v1.3.0.

Finally, fix removes redundant or misleading requirements.
For example, if A v1.0.0 itself requires B v1.2.0 and C v1.0.0, then go.mod's
requirement of B v1.0.0 is misleading (superseded by A's need for v1.2.0),
and its requirement of C v1.0.0 is redundant (implied by A's need for the
same version), so both will be removed. If module M contains packages
that directly import packages from B or C, then the requirements will be
kept but updated to the actual versions being used.

Although fix runs the fix-up operation in isolation, the fix-up also
runs automatically any time a go command uses the module graph,
to update go.mod to reflect reality. Because the module graph defines
the meaning of import statements, any commands that load packages
also use and therefore fix the module graph. For example,
go build, go get, go install, go list, go test, go mod graph, go mod tidy,
and other commands all effectively imply go mod fix.
	`,
	Run: runFix,
}

func runFix(cmd *base.Command, args []string) {
	if len(args) != 0 {
		base.Fatalf("go mod fix: fix takes no arguments")
	}
	modload.LoadBuildList() // writes go.mod
}
