// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdFmt = &Command{
	Run:       runFmt,
	UsageLine: "fmt [packages]",
	Short:     "run gofmt on package sources",
	Long: `
Fmt runs the command 'gofmt -l -w' on the packages named
by the import paths.  It prints the names of the files that are modified.

For more about gofmt, see 'godoc gofmt'.
For more about specifying packages, see 'go help packages'.

To run gofmt with specific options, run gofmt itself.

See also: go doc, go fix, go vet.
	`,
}

func runFmt(cmd *Command, args []string) {
	for _, pkg := range packages(args) {
		// Use pkg.gofiles instead of pkg.Dir so that
		// the command only applies to this package,
		// not to packages in subdirectories.
		run(stringList("gofmt", "-l", "-w", relPaths(pkg.gofiles)))
	}
}

var cmdDoc = &Command{
	Run:       runDoc,
	UsageLine: "doc [packages]",
	Short:     "run godoc on package sources",
	Long: `
Doc runs the godoc command on the packages named by the
import paths.

For more about godoc, see 'godoc godoc'.
For more about specifying packages, see 'go help packages'.

To run godoc with specific options, run godoc itself.

See also: go fix, go fmt, go vet.
	`,
}

func runDoc(cmd *Command, args []string) {
	for _, pkg := range packages(args) {
		if pkg.ImportPath == "command-line arguments" {
			errorf("go doc: cannot use package file list")
			continue
		}
		run("godoc", pkg.Dir)
	}
}
