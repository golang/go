// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lintcmd

import (
	"context"

	"cmd/go/internal/base"
	"cmd/go/internal/load"
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

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
its function literals, the code inlined into it, or a function it calls
allocates on the heap; //logos:zeroalloc local checks the function alone.
No severity lowers that diagnostic and //logos:allow does not silence it.
In a Go package 'logos lint' applies it, and 'logos build' applies it
where the author wrote the directive.

The build flags are those of 'logos build'. The -pedantic flag enables the
pedantic group, raising it from allow to warn. A stricter logos.toml
severity stays in force, and -pedantic=false turns the group off. The
-json-diagnostics flag prints one JSON object per diagnostic on stderr.
Packages of other modules and the standard library are never linted.
Internal and external test files are linted in their test variants,
the way 'go vet' analyzes them.

See also: logos build.
	`,
}

func Register(root *base.Command) {
	CmdLint.Run = runLint
	work.AddBuildFlags(CmdLint, work.OmitVFlag|work.OmitBuildOnlyFlags)
	root.Commands = append(root.Commands, CmdLint)
}

func runLint(ctx context.Context, cmd *base.Command, args []string) {
	settings.CommandLine.LintCmd = true
	moduleLoader := modload.NewLoader()
	moduleLoader.InitWorkfile()
	work.BuildInit(moduleLoader)
	b := work.NewBuilder("", moduleLoader.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()
	pkgOpts := load.PackageOpts{AutoVCS: true, ModResolveTests: true}
	pkgs := load.PackagesAndErrors(moduleLoader, ctx, pkgOpts, args)
	load.CheckPackageErrors(pkgs)
	root := &work.Action{Mode: "logos lint"}
	for _, p := range pkgs {
		if p.Internal.LogosLint == nil || p.Internal.TestmainGo != nil {
			continue
		}
		if len(p.TestGoFiles) == 0 && len(p.XTestGoFiles) == 0 {
			if work.LintsPackage(p) {
				root.Deps = append(root.Deps, b.LintAction(p))
			}
			continue
		}
		// Lint the test variants like 'go vet' does: ptest carries the
		// package's Go files plus its internal test files, and pxtest is
		// the external test package.
		_, ptest, pxtest, perr := load.TestPackagesFor(moduleLoader, ctx, pkgOpts, p, nil)
		if perr != nil {
			base.Errorf("logos lint: %v", perr.Error)
			if work.LintsPackage(p) {
				root.Deps = append(root.Deps, b.LintAction(p))
			}
			continue
		}
		linted := false
		if ptest != nil && work.LintsPackage(ptest) {
			root.Deps = append(root.Deps, b.LintAction(ptest))
			linted = true
		}
		if !linted && work.LintsPackage(p) {
			root.Deps = append(root.Deps, b.LintAction(p))
		}
		if pxtest != nil && work.LintsPackage(pxtest) {
			root.Deps = append(root.Deps, b.LintAction(pxtest))
		}
	}
	b.Do(ctx, root)
}
