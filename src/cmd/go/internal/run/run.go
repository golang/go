// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package run implements the ``go run'' command.
package run

import (
	"context"
	"fmt"
	"os"
	"path"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

var CmdRun = &base.Command{
	UsageLine: "go run [build flags] [-exec xprog] package [arguments...]",
	Short:     "compile and run Go program",
	Long: `
Run compiles and runs the named main Go package.
Typically the package is specified as a list of .go source files from a single directory,
but it may also be an import path, file system path, or pattern
matching a single known package, as in 'go run .' or 'go run my/cmd'.

By default, 'go run' runs the compiled binary directly: 'a.out arguments...'.
If the -exec flag is given, 'go run' invokes the binary using xprog:
	'xprog a.out arguments...'.
If the -exec flag is not given, GOOS or GOARCH is different from the system
default, and a program named go_$GOOS_$GOARCH_exec can be found
on the current search path, 'go run' invokes the binary using that program,
for example 'go_js_wasm_exec a.out arguments...'. This allows execution of
cross-compiled programs when a simulator or other execution method is
available.

The exit status of Run is not the exit status of the compiled binary.

For more about build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build.
	`,
}

func init() {
	CmdRun.Run = runRun // break init loop

	work.AddBuildFlags(CmdRun, work.DefaultBuildFlags)
	CmdRun.Flag.Var((*base.StringsFlag)(&work.ExecCmd), "exec", "")
}

func printStderr(args ...interface{}) (int, error) {
	return fmt.Fprint(os.Stderr, args...)
}

func runRun(ctx context.Context, cmd *base.Command, args []string) {
	work.BuildInit()
	var b work.Builder
	b.Init()
	b.Print = printStderr
	i := 0
	for i < len(args) && strings.HasSuffix(args[i], ".go") {
		i++
	}
	var p *load.Package
	if i > 0 {
		files := args[:i]
		for _, file := range files {
			if strings.HasSuffix(file, "_test.go") {
				// GoFilesPackage is going to assign this to TestGoFiles.
				// Reject since it won't be part of the build.
				base.Fatalf("go run: cannot run *_test.go files (%s)", file)
			}
		}
		p = load.GoFilesPackage(ctx, files)
	} else if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		pkgs := load.PackagesAndErrors(ctx, args[:1])
		if len(pkgs) == 0 {
			base.Fatalf("go run: no packages loaded from %s", args[0])
		}
		if len(pkgs) > 1 {
			var names []string
			for _, p := range pkgs {
				names = append(names, p.ImportPath)
			}
			base.Fatalf("go run: pattern %s matches multiple packages:\n\t%s", args[0], strings.Join(names, "\n\t"))
		}
		p = pkgs[0]
		i++
	} else {
		base.Fatalf("go run: no go files listed")
	}
	cmdArgs := args[i:]
	load.CheckPackageErrors([]*load.Package{p})

	if p.Name != "main" {
		base.Fatalf("go run: cannot run non-main package")
	}
	p.Internal.OmitDebug = true
	p.Target = "" // must build - not up to date
	if p.Internal.CmdlineFiles {
		//set executable name if go file is given as cmd-argument
		var src string
		if len(p.GoFiles) > 0 {
			src = p.GoFiles[0]
		} else if len(p.CgoFiles) > 0 {
			src = p.CgoFiles[0]
		} else {
			// this case could only happen if the provided source uses cgo
			// while cgo is disabled.
			hint := ""
			if !cfg.BuildContext.CgoEnabled {
				hint = " (cgo is disabled)"
			}
			base.Fatalf("go run: no suitable source files%s", hint)
		}
		p.Internal.ExeName = src[:len(src)-len(".go")]
	} else {
		p.Internal.ExeName = path.Base(p.ImportPath)
	}
	a1 := b.LinkAction(work.ModeBuild, work.ModeBuild, p)
	a := &work.Action{Mode: "go run", Func: buildRunProgram, Args: cmdArgs, Deps: []*work.Action{a1}}
	b.Do(ctx, a)
}

// buildRunProgram is the action for running a binary that has already
// been compiled. We ignore exit status.
func buildRunProgram(b *work.Builder, ctx context.Context, a *work.Action) error {
	cmdline := str.StringList(work.FindExecCmd(), a.Deps[0].Target, a.Args)
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "%s", strings.Join(cmdline, " "))
		if cfg.BuildN {
			return nil
		}
	}

	base.RunStdin(cmdline)
	return nil
}
