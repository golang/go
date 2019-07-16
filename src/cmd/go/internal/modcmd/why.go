// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/modload"
	"cmd/go/internal/module"
	"fmt"
	"strings"
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
	`,
}

var (
	whyM      = cmdWhy.Flag.Bool("m", false, "")
	whyVendor = cmdWhy.Flag.Bool("vendor", false, "")
)

func init() {
	cmdWhy.Run = runWhy // break init cycle
}

func runWhy(cmd *base.Command, args []string) {
	loadALL := modload.LoadALL
	if *whyVendor {
		loadALL = modload.LoadVendor
	}
	if *whyM {
		listU := false
		listVersions := false
		for _, arg := range args {
			if strings.Contains(arg, "@") {
				base.Fatalf("go mod why: module query not allowed")
			}
		}
		mods := modload.ListModules(args, listU, listVersions)
		byModule := make(map[module.Version][]string)
		for _, path := range loadALL() {
			m := modload.PackageModule(path)
			if m.Path != "" {
				byModule[m] = append(byModule[m], path)
			}
		}
		sep := ""
		for _, m := range mods {
			best := ""
			bestDepth := 1000000000
			for _, path := range byModule[module.Version{Path: m.Path, Version: m.Version}] {
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
		matches := modload.ImportPaths(args) // resolve to packages
		loadALL()                            // rebuild graph, from main module (not from named packages)
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
