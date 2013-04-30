// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

var cmdRun = &Command{
	UsageLine: "run [build flags] gofiles... [arguments...]",
	Short:     "compile and run Go program",
	Long: `
Run compiles and runs the main package comprising the named Go source files.
If no files are named, it compiles and runs all non-test Go source files.

For more about build flags, see 'go help build'.

See also: go build.
	`,
}

func init() {
	cmdRun.Run = runRun // break init loop

	addBuildFlags(cmdRun)
}

func printStderr(args ...interface{}) (int, error) {
	return fmt.Fprint(os.Stderr, args...)
}

func runRun(cmd *Command, args []string) {
	raceInit()
	var b builder
	b.init()
	b.print = printStderr
	i := 0
	for i < len(args) && strings.HasSuffix(args[i], ".go") {
		i++
	}
	files, cmdArgs := args[:i], args[i:]
	if len(files) == 0 {
		fatalf("go run: no go files listed")
	}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			// goFilesPackage is going to assign this to TestGoFiles.
			// Reject since it won't be part of the build.
			fatalf("go run: cannot run *_test.go files (%s)", file)
		}
	}
	p := goFilesPackage(files)
	if p.Error != nil {
		fatalf("%s", p.Error)
	}
	for _, err := range p.DepsErrors {
		errorf("%s", err)
	}
	exitIfErrors()
	if p.Name != "main" {
		fatalf("go run: cannot run non-main package")
	}
	p.target = "" // must build - not up to date
	var src string
	if len(p.GoFiles) > 0 {
		src = p.GoFiles[0]
	} else if len(p.CgoFiles) > 0 {
		src = p.CgoFiles[0]
	} else {
		// this case could only happen if the provided source uses cgo
		// while cgo is disabled.
		hint := ""
		if !buildContext.CgoEnabled {
			hint = " (cgo is disabled)"
		}
		fatalf("go run: no suitable source files%s", hint)
	}
	p.exeName = src[:len(src)-len(".go")] // name temporary executable for first go file
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

	runStdin(a.deps[0].target, a.args)
	return nil
}

// runStdin is like run, but connects Stdin.
func runStdin(cmdargs ...interface{}) {
	cmdline := stringList(cmdargs...)
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	startSigHandlers()
	if err := cmd.Run(); err != nil {
		errorf("%v", err)
	}
}
