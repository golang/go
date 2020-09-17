// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod tidy

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
	"cmd/go/internal/modload"
	"context"
)

var cmdTidy = &base.Command{
	UsageLine: "go mod tidy [-v]",
	Short:     "add missing and remove unused modules",
	Long: `
Tidy makes sure go.mod matches the source code in the module.
It adds any missing modules necessary to build the current module's
packages and dependencies, and it removes unused modules that
don't provide any relevant packages. It also adds any missing entries
to go.sum and removes any unnecessary ones.

The -v flag causes tidy to print information about removed modules
to standard error.
	`,
}

func init() {
	cmdTidy.Run = runTidy // break init cycle
	cmdTidy.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	base.AddModCommonFlags(&cmdTidy.Flag)
}

func runTidy(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("go mod tidy: no arguments allowed")
	}

	// Tidy aims to make 'go test' reproducible for any package in 'all', so we
	// need to include test dependencies. For modules that specify go 1.15 or
	// earlier this is a no-op (because 'all' saturates transitive test
	// dependencies).
	//
	// However, with lazy loading (go 1.16+) 'all' includes only the packages that
	// are transitively imported by the main module, not the test dependencies of
	// those packages. In order to make 'go test' reproducible for the packages
	// that are in 'all' but outside of the main module, we must explicitly
	// request that their test dependencies be included.
	modload.ForceUseModules = true
	modload.RootMode = modload.NeedRoot

	modload.LoadPackages(ctx, modload.PackageOpts{
		Tags:                  imports.AnyTags(),
		ResolveMissingImports: true,
		LoadTests:             true,
		AllowErrors:           false, // TODO(#26603): Make this a flag.
	}, "all")
	modload.TidyBuildList()
	modload.TrimGoSum()
	modload.WriteGoMod()
}
