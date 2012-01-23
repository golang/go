// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

var cmdRun = &Command{
	UsageLine: "run [-a] [-n] [-x] gofiles... [arguments...]",
	Short:     "compile and run Go program",
	Long: `
Run compiles and runs the main package comprising the named Go source files.

The -a flag forces reinstallation of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -x flag prints the commands.

See also: go build.
	`,
}

func init() {
	cmdRun.Run = runRun // break init loop

	cmdRun.Flag.BoolVar(&buildA, "a", false, "")
	cmdRun.Flag.BoolVar(&buildN, "n", false, "")
	cmdRun.Flag.BoolVar(&buildX, "x", false, "")
}

func runRun(cmd *Command, args []string) {
	var b builder
	b.init()
	i := 0
	for i < len(args) && strings.HasSuffix(args[i], ".go") {
		i++
	}
	files, cmdArgs := args[:i], args[i:]
	p := goFilesPackage(files, "")
	p.target = "" // must build - not up to date
	a1 := b.action(modeBuild, modeBuild, p)
	a := &action{f: (*builder).runProgram, args: cmdArgs, deps: []*action{a1}}
	b.do(a)
}

// runProgram is the action for running a binary that has already
// been compiled.  We ignore exit status.
func (b *builder) runProgram(a *action) error {
	if buildN || buildX {
		b.showcmd("", "%s %s", a.deps[0].target, strings.Join(a.args, " "))
		if buildN {
			return nil
		}
	}
	run(a.deps[0].target, a.args)
	return nil
}
