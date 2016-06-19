// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"path/filepath"
)

func init() {
	addBuildFlagsNX(cmdFmt)
}

var cmdFmt = &Command{
	Run:       runFmt,
	UsageLine: "fmt [-n] [-x] [packages]",
	Short:     "run gofmt on package sources",
	Long: `
Fmt runs the command 'gofmt -l -w' on the packages named
by the import paths.  It prints the names of the files that are modified.

For more about gofmt, see 'go doc cmd/gofmt'.
For more about specifying packages, see 'go help packages'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

To run gofmt with specific options, run gofmt itself.

See also: go fix, go vet.
	`,
}

func runFmt(cmd *Command, args []string) {
	gofmt := gofmtPath()
	for _, pkg := range packages(args) {
		// Use pkg.gofiles instead of pkg.Dir so that
		// the command only applies to this package,
		// not to packages in subdirectories.
		run(stringList(gofmt, "-l", "-w", relPaths(pkg.allgofiles)))
	}
}

func gofmtPath() string {
	gofmt := "gofmt"
	if toolIsWindows {
		gofmt += toolWindowsExtension
	}

	gofmtPath := filepath.Join(gobin, gofmt)
	if _, err := os.Stat(gofmtPath); err == nil {
		return gofmtPath
	}

	gofmtPath = filepath.Join(goroot, "bin", gofmt)
	if _, err := os.Stat(gofmtPath); err == nil {
		return gofmtPath
	}

	// fallback to looking for gofmt in $PATH
	return "gofmt"
}
