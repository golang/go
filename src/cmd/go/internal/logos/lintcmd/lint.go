// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lintcmd implements logos lint: it runs the compiler lints over
// the packages of the main module without building binaries.
package lintcmd

import (
	"context"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

// CmdLint implements the logos lint command.
var CmdLint = &base.Command{
	UsageLine: "logos lint [build flags] [--pedantic] [--json-diagnostics] [packages]",
	Short:     "report compiler lints without building binaries",
	Long: `
Lint type-checks the named packages of the main module, runs the compiler
lints through escape analysis, and stops before code generation, so it
writes no binary. It lints loGo/s packages with every rule and Go packages
with the language-neutral rules only. Severities come from the [lints]
table of logos.toml; a lint that is an error fails the command.

The //logos:allow(<rule>) comment before a statement or declaration
silences a deny or warn rule. A forbid rule ignores it in a loGo/s package,
and the directive itself is then an error. Go packages run a forbid rule
at deny, so the directive silences it there.

The //logos:zeroalloc comment on the line above a function declaration
makes the compiler fail with [perf::zeroalloc_violation] when the function,
its function literals, or the code inlined into it allocates on the heap.
No severity lowers that diagnostic and //logos:allow does not silence it.
In a Go package only 'logos lint' applies it.

The build flags are those of 'logos build'. The -pedantic flag enables the
pedantic group, overriding logos.toml. The -json-diagnostics flag prints
one JSON object per diagnostic on stderr. Packages of other modules and the
standard library are never linted.

See also: logos build.
	`,
}

// Register adds the lint subcommand to the logos root command.
func Register(root *base.Command) {
	CmdLint.Run = runLint
	work.AddBuildFlags(CmdLint, work.OmitVFlag|work.OmitBuildOnlyFlags)
	root.Commands = append(root.Commands, CmdLint)
}

// runLint loads the packages and runs one lint action for each package of
// a main module.
func runLint(ctx context.Context, cmd *base.Command, args []string) {
	cfg.LogosLintCmd = true
	moduleLoader := modload.NewLoader()
	moduleLoader.InitWorkfile()
	work.BuildInit(moduleLoader)
	b := work.NewBuilder("", moduleLoader.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()
	pkgs := load.PackagesAndErrors(moduleLoader, ctx, load.PackageOpts{AutoVCS: true}, args)
	load.CheckPackageErrors(pkgs)
	root := &work.Action{Mode: "logos lint"}
	for _, p := range pkgs {
		if work.LintsPackage(p) {
			root.Deps = append(root.Deps, b.LintAction(p))
		}
	}
	b.Do(ctx, root)
}
