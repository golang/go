// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdFmt = &Command{
	Run:       runFmt,
	UsageLine: "fmt [importpath...]",
	Short:     "run gofmt -w on packages",
	Long: `
Fmt runs the command 'gofmt -w' on the packages named by the import paths.

For more about gofmt, see 'godoc gofmt'.
For more about import paths, see 'go help importpath'.

To run gofmt with specific options, run gofmt itself.

See also: go fix, go vet.
	`,
}

func runFmt(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("fmt not implemented")
}
