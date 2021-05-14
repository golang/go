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

	"golang.org/x/mod/modfile"
)

var cmdTidy = &base.Command{
	UsageLine: "go mod tidy [-e] [-v] [-go=version]",
	Short:     "add missing and remove unused modules",
	Long: `
Tidy makes sure go.mod matches the source code in the module.
It adds any missing modules necessary to build the current module's
packages and dependencies, and it removes unused modules that
don't provide any relevant packages. It also adds any missing entries
to go.sum and removes any unnecessary ones.

The -v flag causes tidy to print information about removed modules
to standard error.

The -e flag causes tidy to attempt to proceed despite errors
encountered while loading packages.

The -go flag causes tidy to update the 'go' directive in the go.mod
file to the given version, which may change which module dependencies
are retained as explicit requirements in the go.mod file.
(Go versions 1.17 and higher retain more requirements in order to
support lazy module loading.)

See https://golang.org/ref/mod#go-mod-tidy for more about 'go mod tidy'.
	`,
	Run: runTidy,
}

var (
	tidyE  bool   // if true, report errors but proceed anyway.
	tidyGo string // go version to write to the tidied go.mod file (toggles lazy loading)
)

func init() {
	cmdTidy.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	cmdTidy.Flag.BoolVar(&tidyE, "e", false, "")
	cmdTidy.Flag.StringVar(&tidyGo, "go", "", "")
	base.AddModCommonFlags(&cmdTidy.Flag)
}

func runTidy(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("go mod tidy: no arguments allowed")
	}

	if tidyGo != "" {
		if !modfile.GoVersionRE.MatchString(tidyGo) {
			base.Fatalf(`go mod: invalid -go option %q; expecting something like "-go 1.17"`, tidyGo)
		}
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
		GoVersion:                tidyGo,
		Tags:                     imports.AnyTags(),
		Tidy:                     true,
		VendorModulesInGOROOTSrc: true,
		ResolveMissingImports:    true,
		LoadTests:                true,
		AllowErrors:              tidyE,
		SilenceMissingStdImports: true,
	}, "all")
}
