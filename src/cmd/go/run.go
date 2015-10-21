// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

var execCmd []string // -exec flag, for run and test

func findExecCmd() []string {
	if execCmd != nil {
		return execCmd
	}
	execCmd = []string{} // avoid work the second time
	if goos == runtime.GOOS && goarch == runtime.GOARCH {
		return execCmd
	}
	path, err := exec.LookPath(fmt.Sprintf("go_%s_%s_exec", goos, goarch))
	if err == nil {
		execCmd = []string{path}
	}
	return execCmd
}

var cmdRun = &Command{
	UsageLine: "run [build flags] [-exec xprog] gofiles... [arguments...]",
	Short:     "compile and run Go program",
	Long: `
Run compiles and runs the main package comprising the named Go source files.
A Go source file is defined to be a file ending in a literal ".go" suffix.

By default, 'go run' runs the compiled binary directly: 'a.out arguments...'.
If the -exec flag is given, 'go run' invokes the binary using xprog:
	'xprog a.out arguments...'.
If the -exec flag is not given, GOOS or GOARCH is different from the system
default, and a program named go_$GOOS_$GOARCH_exec can be found
on the current search path, 'go run' invokes the binary using that program,
for example 'go_nacl_386_exec a.out arguments...'. This allows execution of
cross-compiled programs when a simulator or other execution method is
available.

For more about build flags, see 'go help build'.

See also: go build.
	`,
}

func init() {
	cmdRun.Run = runRun // break init loop

	addBuildFlags(cmdRun)
	cmdRun.Flag.Var((*stringsFlag)(&execCmd), "exec", "")
}

func printStderr(args ...interface{}) (int, error) {
	return fmt.Fprint(os.Stderr, args...)
}

func runRun(cmd *Command, args []string) {
	instrumentInit()
	buildModeInit()
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
	p.omitDWARF = true
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
	cmdline := stringList(findExecCmd(), a.deps[0].target, a.args)
	if buildN || buildX {
		b.showcmd("", "%s", strings.Join(cmdline, " "))
		if buildN {
			return nil
		}
	}

	runStdin(cmdline)
	return nil
}

// runStdin is like run, but connects Stdin.
func runStdin(cmdline []string) {
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = origEnv
	startSigHandlers()
	if err := cmd.Run(); err != nil {
		errorf("%v", err)
	}
}
