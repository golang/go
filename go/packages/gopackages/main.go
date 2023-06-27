// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gopackages command is a diagnostic tool that demonstrates
// how to use golang.org/x/tools/go/packages to load, parse,
// type-check, and print one or more Go packages.
// Its precise output is unspecified and may change.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/types"
	"os"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/tool"
)

func main() {
	tool.Main(context.Background(), &application{Mode: "imports"}, os.Args[1:])
}

type application struct {
	// Embed the basic profiling flags supported by the tool package
	tool.Profile

	Deps       bool            `flag:"deps" help:"show dependencies too"`
	Test       bool            `flag:"test" help:"include any tests implied by the patterns"`
	Mode       string          `flag:"mode" help:"mode (one of files, imports, types, syntax, allsyntax)"`
	Private    bool            `flag:"private" help:"show non-exported declarations too (if -mode=syntax)"`
	PrintJSON  bool            `flag:"json" help:"print package in JSON form"`
	BuildFlags stringListValue `flag:"buildflag" help:"pass argument to underlying build system (may be repeated)"`
}

// Name implements tool.Application returning the binary name.
func (app *application) Name() string { return "gopackages" }

// Usage implements tool.Application returning empty extra argument usage.
func (app *application) Usage() string { return "package..." }

// ShortHelp implements tool.Application returning the main binary help.
func (app *application) ShortHelp() string {
	return "gopackages loads, parses, type-checks, and prints one or more Go packages."
}

// DetailedHelp implements tool.Application returning the main binary help.
func (app *application) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Packages are specified using the notation of "go list",
or other underlying build system.

The mode flag determines how much information is computed and printed
for the specified packages. In order of increasing computational cost,
the legal values are:

 -mode=files     shows only the names of the packages' files.
 -mode=imports   also shows the imports. (This is the default.)
 -mode=types     loads the compiler's export data and displays the
                 type of each exported declaration.
 -mode=syntax    parses and type checks syntax trees for the initial
                 packages. (With the -private flag, the types of
                 non-exported declarations are shown too.)
                 Type information for dependencies is obtained from
                 compiler export data.
 -mode=allsyntax is like -mode=syntax but applied to all dependencies.

Flags:
`)
	f.PrintDefaults()
}

// Run takes the args after flag processing and performs the specified query.
func (app *application) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return tool.CommandLineErrorf("not enough arguments")
	}

	// Load, parse, and type-check the packages named on the command line.
	cfg := &packages.Config{
		Mode:       packages.LoadSyntax,
		Tests:      app.Test,
		BuildFlags: app.BuildFlags,
	}

	// -mode flag
	switch strings.ToLower(app.Mode) {
	case "files":
		cfg.Mode = packages.LoadFiles
	case "imports":
		cfg.Mode = packages.LoadImports
	case "types":
		cfg.Mode = packages.LoadTypes
	case "syntax":
		cfg.Mode = packages.LoadSyntax
	case "allsyntax":
		cfg.Mode = packages.LoadAllSyntax
	default:
		return tool.CommandLineErrorf("invalid mode: %s", app.Mode)
	}

	lpkgs, err := packages.Load(cfg, args...)
	if err != nil {
		return err
	}

	// -deps: print dependencies too.
	if app.Deps {
		// We can't use packages.All because
		// we need an ordered traversal.
		var all []*packages.Package // postorder
		seen := make(map[*packages.Package]bool)
		var visit func(*packages.Package)
		visit = func(lpkg *packages.Package) {
			if !seen[lpkg] {
				seen[lpkg] = true

				// visit imports
				var importPaths []string
				for path := range lpkg.Imports {
					importPaths = append(importPaths, path)
				}
				sort.Strings(importPaths) // for determinism
				for _, path := range importPaths {
					visit(lpkg.Imports[path])
				}

				all = append(all, lpkg)
			}
		}
		for _, lpkg := range lpkgs {
			visit(lpkg)
		}
		lpkgs = all
	}

	for _, lpkg := range lpkgs {
		app.print(lpkg)
	}
	return nil
}

func (app *application) print(lpkg *packages.Package) {
	if app.PrintJSON {
		data, _ := json.MarshalIndent(lpkg, "", "\t")
		os.Stdout.Write(data)
		return
	}
	// title
	var kind string
	// TODO(matloob): If IsTest is added back print "test command" or
	// "test package" for packages with IsTest == true.
	if lpkg.Name == "main" {
		kind += "command"
	} else {
		kind += "package"
	}
	fmt.Printf("Go %s %q:\n", kind, lpkg.ID) // unique ID
	fmt.Printf("\tpackage %s\n", lpkg.Name)

	// characterize type info
	if lpkg.Types == nil {
		fmt.Printf("\thas no exported type info\n")
	} else if !lpkg.Types.Complete() {
		fmt.Printf("\thas incomplete exported type info\n")
	} else if len(lpkg.Syntax) == 0 {
		fmt.Printf("\thas complete exported type info\n")
	} else {
		fmt.Printf("\thas complete exported type info and typed ASTs\n")
	}
	if lpkg.Types != nil && lpkg.IllTyped && len(lpkg.Errors) == 0 {
		fmt.Printf("\thas an error among its dependencies\n")
	}

	// source files
	for _, src := range lpkg.GoFiles {
		fmt.Printf("\tfile %s\n", src)
	}

	// imports
	var lines []string
	for importPath, imp := range lpkg.Imports {
		var line string
		if imp.ID == importPath {
			line = fmt.Sprintf("\timport %q", importPath)
		} else {
			line = fmt.Sprintf("\timport %q => %q", importPath, imp.ID)
		}
		lines = append(lines, line)
	}
	sort.Strings(lines)
	for _, line := range lines {
		fmt.Println(line)
	}

	// errors
	for _, err := range lpkg.Errors {
		fmt.Printf("\t%s\n", err)
	}

	// types of package members
	if lpkg.Types != nil {
		qual := types.RelativeTo(lpkg.Types)
		scope := lpkg.Types.Scope()
		for _, name := range scope.Names() {
			obj := scope.Lookup(name)
			if !obj.Exported() && !app.Private {
				continue // skip unexported names
			}

			fmt.Printf("\t%s\n", types.ObjectString(obj, qual))
			if _, ok := obj.(*types.TypeName); ok {
				for _, meth := range typeutil.IntuitiveMethodSet(obj.Type(), nil) {
					if !meth.Obj().Exported() && !app.Private {
						continue // skip unexported names
					}
					fmt.Printf("\t%s\n", types.SelectionString(meth, qual))
				}
			}
		}
	}

	fmt.Println()
}

// stringListValue is a flag.Value that accumulates strings.
// e.g. --flag=one --flag=two would produce []string{"one", "two"}.
type stringListValue []string

func newStringListValue(val []string, p *[]string) *stringListValue {
	*p = val
	return (*stringListValue)(p)
}

func (ss *stringListValue) Get() interface{} { return []string(*ss) }

func (ss *stringListValue) String() string { return fmt.Sprintf("%q", *ss) }

func (ss *stringListValue) Set(s string) error { *ss = append(*ss, s); return nil }
