// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"context"
	"fmt"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/imports"
	"cmd/go/internal/modload"

	"golang.org/x/mod/module"
)

var cmdWhy = &base.Command{
	UsageLine: "go mod why [-m] [-vendor] packages...",
	Short:     "explain why packages or modules are needed",
	Long: `
Why shows a shortest path in the import graph from the main module to
each of the listed packages. If the -m flag is given, why treats the
arguments as a list of modules and finds a path to any package in each
of the modules.

By default, why queries the graph of packages matched by "go list all",
which includes tests for reachable packages. The -vendor flag causes why
to exclude tests of dependencies.

The output is a sequence of stanzas, one for each package or module
name on the command line, separated by blank lines. Each stanza begins
with a comment line "# package" or "# module" giving the target
package or module. Subsequent lines give a path through the import
graph, one package per line. If the package or module is not
referenced from the main module, the stanza will display a single
parenthesized note indicating that fact.

For example:

	$ go mod why golang.org/x/text/language golang.org/x/text/encoding
	# golang.org/x/text/language
	rsc.io/quote
	rsc.io/sampler
	golang.org/x/text/language

	# golang.org/x/text/encoding
	(main module does not need package golang.org/x/text/encoding)
	$

See https://go.dev/ref/mod#go-mod-why for more about 'go mod why'.
	`,
}

var (
	whyM      = cmdWhy.Flag.Bool("m", false, "treat arguments as a list of modules")
	whyVendor = cmdWhy.Flag.Bool("vendor", false, "exclude tests of dependencies")
)

func init() {
	cmdWhy.Run = runWhy // break init cycle
	base.AddChdirFlag(&cmdWhy.Flag)
	base.AddModCommonFlags(&cmdWhy.Flag)
}

func runWhy(ctx context.Context, cmd *base.Command, args []string) {
	moduleLoader := modload.NewLoader()
	moduleLoader.InitWorkfile()
	moduleLoader.ForceUseModules = true
	moduleLoader.RootMode = modload.NeedRoot
	modload.ExplicitWriteGoMod = true // don't write go.mod in ListModules

	loadOpts := modload.PackageOpts{
		Tags:                     imports.AnyTags(),
		VendorModulesInGOROOTSrc: true,
		LoadTests:                !*whyVendor,
		SilencePackageErrors:     true,
		UseVendorAll:             *whyVendor,
	}

	if *whyM {
		for _, arg := range args {
			if strings.Contains(arg, "@") {
				base.Fatalf("go: %s: 'go mod why' requires a module path, not a version query", arg)
			}
			if err := checkModulePathPattern(arg); err != nil {
				base.Errorf("go mod why: %v", err)
			}
		}
		base.ExitIfErrors()

		mods, err := modload.ListModules(moduleLoader, ctx, args, 0, "")
		if err != nil {
			base.Fatal(err)
		}

		byModule := make(map[string][]string)
		_, pkgs := modload.LoadPackages(moduleLoader, ctx, loadOpts, "all")
		for _, path := range pkgs {
			m := moduleLoader.PackageModule(path)
			if m.Path != "" {
				byModule[m.Path] = append(byModule[m.Path], path)
			}
		}
		sep := ""
		for _, m := range mods {
			best := ""
			bestDepth := 1000000000
			for _, path := range byModule[m.Path] {
				d := moduleLoader.WhyDepth(path)
				if d > 0 && d < bestDepth {
					best = path
					bestDepth = d
				}
			}
			why := moduleLoader.Why(best)
			if why == "" {
				vendoring := ""
				if *whyVendor {
					vendoring = " to vendor"
				}
				why = "(main module does not need" + vendoring + " module " + m.Path + ")\n"
			}
			fmt.Printf("%s# %s\n%s", sep, m.Path, why)
			sep = "\n"
		}
	} else {
		// Resolve to packages.
		matches, _ := modload.LoadPackages(moduleLoader, ctx, loadOpts, args...)

		modload.LoadPackages(moduleLoader, ctx, loadOpts, "all") // rebuild graph, from main module (not from named packages)

		sep := ""
		for _, m := range matches {
			for _, path := range m.Pkgs {
				why := moduleLoader.Why(path)
				if why == "" {
					vendoring := ""
					if *whyVendor {
						vendoring = " to vendor"
					}
					why = "(main module does not need" + vendoring + " package " + path + ")\n"
				}
				fmt.Printf("%s# %s\n%s", sep, path, why)
				sep = "\n"
			}
		}
	}
}

func checkModulePathPattern(pattern string) error {
	parts := strings.Split(pattern, "...")
	if len(parts) == 1 {
		return modulePathError(pattern, module.CheckImportPath(pattern))
	}

	// Add placeholders for the wildcards adjoining each literal part so that
	// separators at wildcard boundaries form complete paths during validation.
	if err := module.CheckImportPath(parts[0] + "x"); err != nil {
		return modulePathError(pattern, err)
	}
	for i, part := range parts[1:] {
		if i < len(parts)-2 {
			part += "x"
		}
		if err := module.CheckFilePath("x" + part); err != nil {
			return modulePathError(pattern, err)
		}
	}
	return nil
}

func modulePathError(path string, err error) error {
	if pathErr, ok := err.(*module.InvalidPathError); ok {
		pathErr.Kind = "module"
		pathErr.Path = path
	}
	return err
}
