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
	"fmt"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
)

var cmdTidy = &base.Command{
	UsageLine: "go mod tidy [-e] [-v] [-go=version] [-compat=version]",
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

The -compat flag preserves any additional checksums needed for the
'go' command from the indicated major Go release to successfully load
the module graph, and causes tidy to error out if that version of the
'go' command would load any imported package from a different module
version. By default, tidy acts as if the -compat flag were set to the
version prior to the one indicated by the 'go' directive in the go.mod
file.

See https://golang.org/ref/mod#go-mod-tidy for more about 'go mod tidy'.
	`,
	Run: runTidy,
}

var (
	tidyE      bool          // if true, report errors but proceed anyway.
	tidyGo     goVersionFlag // go version to write to the tidied go.mod file (toggles lazy loading)
	tidyCompat goVersionFlag // go version for which the tidied go.mod and go.sum files should be “compatible”
)

func init() {
	cmdTidy.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	cmdTidy.Flag.BoolVar(&tidyE, "e", false, "")
	cmdTidy.Flag.Var(&tidyGo, "go", "")
	cmdTidy.Flag.Var(&tidyCompat, "compat", "")
	base.AddModCommonFlags(&cmdTidy.Flag)
}

// A goVersionFlag is a flag.Value representing a supported Go version.
//
// (Note that the -go argument to 'go mod edit' is *not* a goVersionFlag.
// It intentionally allows newer-than-supported versions as arguments.)
type goVersionFlag struct {
	v string
}

func (f *goVersionFlag) String() string   { return f.v }
func (f *goVersionFlag) Get() interface{} { return f.v }

func (f *goVersionFlag) Set(s string) error {
	if s != "" {
		latest := modload.LatestGoVersion()
		if !modfile.GoVersionRE.MatchString(s) {
			return fmt.Errorf("expecting a Go version like %q", latest)
		}
		if semver.Compare("v"+s, "v"+latest) > 0 {
			return fmt.Errorf("maximum supported Go version is %s", latest)
		}
	}

	f.v = s
	return nil
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
		GoVersion:                tidyGo.String(),
		Tags:                     imports.AnyTags(),
		Tidy:                     true,
		TidyCompatibleVersion:    tidyCompat.String(),
		VendorModulesInGOROOTSrc: true,
		ResolveMissingImports:    true,
		LoadTests:                true,
		AllowErrors:              tidyE,
		SilenceMissingStdImports: true,
	}, "all")
}
