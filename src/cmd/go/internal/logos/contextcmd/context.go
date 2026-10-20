// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package contextcmd implements logos context: it prints the declaration
// of one package symbol with one level of mentioned signatures.
package contextcmd

import (
	"context"
	"os"
	"os/exec"
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

// CmdContext implements the logos context command.
var CmdContext = &base.Command{
	Run:       runContext,
	UsageLine: "logos context [<pkg>#<symbol>]",
	Short:     "print one symbol declaration and its signatures",
	Long: `
Context prints the declaration of the named symbol in the named
package, the signatures of its methods, and the declarations of the
package types those mention one level deep, with no function bodies
and at most 100 lines. It serves .lgo packages and .go packages.

An unknown symbol fails with logos context: <pkg> has no symbol <name>.
	`,
}

// Register adds the context subcommand to the logos root command.
func Register(root *base.Command) {
	root.Commands = append(root.Commands, CmdContext)
}

// runContext resolves the package and drives the compiler context mode.
func runContext(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) != 1 {
		base.Fatalf("logos context: expected <pkg>#<symbol>")
	}
	pkgpath, symbol, _ := strings.Cut(args[0], "#")
	if pkgpath == "" || symbol == "" {
		base.Fatalf("logos context: expected <pkg>#<symbol>")
	}
	moduleLoader := modload.NewLoader()
	moduleLoader.InitWorkfile()
	work.BuildInit(moduleLoader)
	pkgs := load.PackagesAndErrors(moduleLoader, ctx, load.PackageOpts{}, []string{pkgpath})
	load.CheckPackageErrors(pkgs)
	pkg := pkgs[0]
	if len(pkg.GoFiles) == 0 {
		base.Fatalf("logos context: %s has no Go files", pkgpath)
	}
	toolArgs := []string{"-logos.mode=context", "-logos.symbol=" + symbol, "-p", pkg.ImportPath}
	toolArgs = append(toolArgs, pkg.GoFiles...)
	argv := slices.Clone(cfg.BuildToolexec)
	argv = append(argv, base.Tool("compile"))
	argv = append(argv, toolArgs...)
	compile := exec.Command(argv[0], argv[1:]...)
	compile.Dir = pkg.Dir
	compile.Stdout = os.Stdout
	compile.Stderr = os.Stderr
	compile.Env = os.Environ()
	if err := compile.Run(); err != nil {
		base.SetExitStatus(1)
		base.Exit()
	}
}
