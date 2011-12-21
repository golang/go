// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import ()

// Break init loop.
func init() {
	cmdRun.Run = runRun
}

var cmdRun = &Command{
	UsageLine: "run [-a] [-n] [-x] gofiles...",
	Short:     "compile and run Go program",
	Long: `
Run compiles and runs the main package comprising the named Go source files.

The -a flag forces reinstallation of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -x flag prints the commands.

See also: go build.
	`,
}

var runA = cmdRun.Flag.Bool("a", false, "")
var runN = cmdRun.Flag.Bool("n", false, "")
var runX = cmdRun.Flag.Bool("x", false, "")

func runRun(cmd *Command, args []string) {
	var b builder
	b.init(*runA, *runN, *runX)
	p := goFilesPackage(args, "")
	p.target = "" // must build - not up to date
	a1 := b.action(modeBuild, modeBuild, p)
	a := &action{f: (*builder).runProgram, deps: []*action{a1}}
	b.do(a)
}

// runProgram is the action for running a binary that has already
// been compiled.  We ignore exit status.
func (b *builder) runProgram(a *action) error {
	run(a.deps[0].target)
	return nil
}
