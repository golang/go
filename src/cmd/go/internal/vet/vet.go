// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vet implements the “go vet” command.
package vet

import (
	"context"
	"fmt"
	"path/filepath"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/trace"
	"cmd/go/internal/work"
)

// Break init loop.
func init() {
	CmdVet.Run = runVet
}

var CmdVet = &base.Command{
	CustomFlags: true,
	UsageLine:   "go vet [-n] [-x] [-vettool prog] [build flags] [vet flags] [packages]",
	Short:       "report likely mistakes in packages",
	Long: `
Vet runs the Go vet command on the packages named by the import paths.

For more about vet and its flags, see 'go doc cmd/vet'.
For more about specifying packages, see 'go help packages'.
For a list of checkers and their flags, see 'go tool vet help'.
For details of a specific checker such as 'printf', see 'go tool vet help printf'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

The -vettool=prog flag selects a different analysis tool with alternative
or additional checks.
For example, the 'shadow' analyzer can be built and run using these commands:

  go install golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow
  go vet -vettool=$(which shadow)

The build flags supported by go vet are those that control package resolution
and execution, such as -n, -x, -v, -tags, and -toolexec.
For more about these flags, see 'go help build'.

See also: go fmt, go fix.
	`,
}

func runVet(ctx context.Context, cmd *base.Command, args []string) {
	vetFlags, pkgArgs := vetFlags(args)
	modload.InitWorkfile() // The vet command does custom flag processing; initialize workspaces after that.

	if cfg.DebugTrace != "" {
		var close func() error
		var err error
		ctx, close, err = trace.Start(ctx, cfg.DebugTrace)
		if err != nil {
			base.Fatalf("failed to start trace: %v", err)
		}
		defer func() {
			if err := close(); err != nil {
				base.Fatalf("failed to stop trace: %v", err)
			}
		}()
	}

	ctx, span := trace.StartSpan(ctx, fmt.Sprint("Running ", cmd.Name(), " command"))
	defer span.Done()

	work.BuildInit()
	work.VetFlags = vetFlags
	if len(vetFlags) > 0 {
		work.VetExplicit = true
	}
	if vetTool != "" {
		var err error
		work.VetTool, err = filepath.Abs(vetTool)
		if err != nil {
			base.Fatalf("%v", err)
		}
	}

	pkgOpts := load.PackageOpts{ModResolveTests: true}
	pkgs := load.PackagesAndErrors(ctx, pkgOpts, pkgArgs)
	load.CheckPackageErrors(pkgs)
	if len(pkgs) == 0 {
		base.Fatalf("no packages to vet")
	}

	b := work.NewBuilder("")
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatalf("go: %v", err)
		}
	}()

	root := &work.Action{Mode: "go vet"}
	for _, p := range pkgs {
		_, ptest, pxtest, err := load.TestPackagesFor(ctx, pkgOpts, p, nil)
		if err != nil {
			base.Errorf("%v", err)
			continue
		}
		if len(ptest.GoFiles) == 0 && len(ptest.CgoFiles) == 0 && pxtest == nil {
			base.Errorf("go: can't vet %s: no Go files in %s", p.ImportPath, p.Dir)
			continue
		}
		if len(ptest.GoFiles) > 0 || len(ptest.CgoFiles) > 0 {
			root.Deps = append(root.Deps, b.VetAction(work.ModeBuild, work.ModeBuild, ptest))
		}
		if pxtest != nil {
			root.Deps = append(root.Deps, b.VetAction(work.ModeBuild, work.ModeBuild, pxtest))
		}
	}
	b.Do(ctx, root)
}
