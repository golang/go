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

See https://golang.org/ref/mod#go-mod-why for more about 'go mod why'.
	`,
}

var (
	whyM      = cmdWhy.Flag.Bool("m", false, "")
	whyVendor = cmdWhy.Flag.Bool("vendor", false, "")
)

func init() {
	cmdWhy.Run = runWhy // break init cycle
	base.AddChdirFlag(&cmdWhy.Flag)
	base.AddModCommonFlags(&cmdWhy.Flag)
}

func runWhy(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()
	modload.LoaderState.ForceUseModules = true
	modload.RootMode = modload.NeedRoot
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
		}

		mods, err := modload.ListModules(ctx, args, 0, "")
		if err != nil {
			base.Fatal(err)
		}

		byModule := make(map[string][]string)
		_, pkgs := modload.LoadPackages(ctx, loadOpts, "all")
		for _, path := range pkgs {
			m := modload.PackageModule(path)
			if m.Path != "" {
				byModule[m.Path] = append(byModule[m.Path], path)
			}
		}
		sep := ""
		for _, m := range mods {
			best := ""
			bestDepth := 1000000000
			for _, path := range byModule[m.Path] {
				d := modload.WhyDepth(path)
				if d > 0 && d < bestDepth {
					best = path
					bestDepth = d
				}
			}
			why := modload.Why(best)
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
		matches, _ := modload.LoadPackages(ctx, loadOpts, args...)

		modload.LoadPackages(ctx, loadOpts, "all") // rebuild graph, from main module (not from named packages)

		sep := ""
		for _, m := range matches {
			for _, path := range m.Pkgs {
				why := modload.Why(path)
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
