// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdBuild = &Command{
	Run:       runBuild,
	UsageLine: "build [-n] [-v] [importpath...]",
	Short:     "compile and install packages and dependencies",
	Long: `
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

The -n flag prints the commands but does not run them.
The -v flag prints the commands.

For more about import paths, see 'go help importpath'.

See also: go install, go get, go clean.
	`,
}

var buildN = cmdBuild.Flag.Bool("n", false, "")
var buildV = cmdBuild.Flag.Bool("v", false, "")

func runBuild(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("build not implemented")
}

var cmdInstall = &Command{
	Run:       runInstall,
	UsageLine: "install [-n] [-v] [importpath...]",
	Short:     "install packages and dependencies",
	Long: `
Install compiles and installs the packages named by the import paths,
along with their dependencies.

The -n flag prints the commands but does not run them.
The -v flag prints the commands.

For more about import paths, see 'go help importpath'.

See also: go build, go get, go clean.
	`,
}

var installN = cmdInstall.Flag.Bool("n", false, "")
var installV = cmdInstall.Flag.Bool("v", false, "")

func runInstall(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("install not implemented")
}
