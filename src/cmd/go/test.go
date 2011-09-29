// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdTest = &Command{
	Run:       runTest,
	UsageLine: "test [importpath...]",
	Short:     "test packages",
	Long: `
Test runs gotest to test the packages named by the import paths.
It prints a summary of the test results in the format:

	test archive/tar
	FAIL archive/zip
	test compress/gzip
	...

followed by gotest output for each failed package.

For more about import paths, see 'go help importpath'.

See also: go build, go compile, go vet.
	`,
}

func runTest(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("test not implemented")
}
