// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The eg command performs example-based refactoring.
// For documentation, run the command, or see Help in
// golang.org/x/tools/refactor/eg.
package main // import "golang.org/x/tools/cmd/eg"

import (
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"strings"

	exec "golang.org/x/sys/execabs"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/refactor/eg"
)

var (
	beforeeditFlag = flag.String("beforeedit", "", "A command to exec before each file is edited (e.g. chmod, checkout).  Whitespace delimits argument words.  The string '{}' is replaced by the file name.")
	helpFlag       = flag.Bool("help", false, "show detailed help message")
	templateFlag   = flag.String("t", "", "template.go file specifying the refactoring")
	transitiveFlag = flag.Bool("transitive", false, "apply refactoring to all dependencies too")
	writeFlag      = flag.Bool("w", false, "rewrite input files in place (by default, the results are printed to standard output)")
	verboseFlag    = flag.Bool("v", false, "show verbose matcher diagnostics")
)

const usage = `eg: an example-based refactoring tool.

Usage: eg -t template.go [-w] [-transitive] <packages>

-help            show detailed help message
-t template.go	 specifies the template file (use -help to see explanation)
-w          	 causes files to be re-written in place.
-transitive 	 causes all dependencies to be refactored too.
-v               show verbose matcher diagnostics
-beforeedit cmd  a command to exec before each file is modified.
                 "{}" represents the name of the file.
`

func main() {
	if err := doMain(); err != nil {
		fmt.Fprintf(os.Stderr, "eg: %s\n", err)
		os.Exit(1)
	}
}

func doMain() error {
	flag.Parse()
	args := flag.Args()

	if *helpFlag {
		help := eg.Help // hide %s from vet
		fmt.Fprint(os.Stderr, help)
		os.Exit(2)
	}

	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	if *templateFlag == "" {
		return fmt.Errorf("no -t template.go file specified")
	}

	tAbs, err := filepath.Abs(*templateFlag)
	if err != nil {
		return err
	}
	template, err := os.ReadFile(tAbs)
	if err != nil {
		return err
	}

	cfg := &packages.Config{
		Fset:  token.NewFileSet(),
		Mode:  packages.NeedTypesInfo | packages.NeedName | packages.NeedTypes | packages.NeedSyntax | packages.NeedImports | packages.NeedDeps | packages.NeedCompiledGoFiles,
		Tests: true,
	}

	pkgs, err := packages.Load(cfg, args...)
	if err != nil {
		return err
	}

	tFile, err := parser.ParseFile(cfg.Fset, tAbs, template, parser.ParseComments)
	if err != nil {
		return err
	}

	// Type-check the template.
	tInfo := types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Scopes:     make(map[ast.Node]*types.Scope),
	}
	conf := types.Config{
		Importer: pkgsImporter(pkgs),
	}
	tPkg, err := conf.Check("egtemplate", cfg.Fset, []*ast.File{tFile}, &tInfo)
	if err != nil {
		return err
	}

	// Analyze the template.
	xform, err := eg.NewTransformer(cfg.Fset, tPkg, tFile, &tInfo, *verboseFlag)
	if err != nil {
		return err
	}

	// Apply it to the input packages.
	var all []*packages.Package
	if *transitiveFlag {
		packages.Visit(pkgs, nil, func(p *packages.Package) { all = append(all, p) })
	} else {
		all = pkgs
	}
	var hadErrors bool
	for _, pkg := range pkgs {
		for i, filename := range pkg.CompiledGoFiles {
			if filename == tAbs {
				// Don't rewrite the template file.
				continue
			}
			file := pkg.Syntax[i]
			n := xform.Transform(pkg.TypesInfo, pkg.Types, file)
			if n == 0 {
				continue
			}
			fmt.Fprintf(os.Stderr, "=== %s (%d matches)\n", filename, n)
			if *writeFlag {
				// Run the before-edit command (e.g. "chmod +w",  "checkout") if any.
				if *beforeeditFlag != "" {
					args := strings.Fields(*beforeeditFlag)
					// Replace "{}" with the filename, like find(1).
					for i := range args {
						if i > 0 {
							args[i] = strings.Replace(args[i], "{}", filename, -1)
						}
					}
					cmd := exec.Command(args[0], args[1:]...)
					cmd.Stdout = os.Stdout
					cmd.Stderr = os.Stderr
					if err := cmd.Run(); err != nil {
						fmt.Fprintf(os.Stderr, "Warning: edit hook %q failed (%s)\n",
							args, err)
					}
				}
				if err := eg.WriteAST(cfg.Fset, filename, file); err != nil {
					fmt.Fprintf(os.Stderr, "eg: %s\n", err)
					hadErrors = true
				}
			} else {
				format.Node(os.Stdout, cfg.Fset, file)
			}
		}
	}
	if hadErrors {
		os.Exit(1)
	}

	return nil
}

type pkgsImporter []*packages.Package

func (p pkgsImporter) Import(path string) (tpkg *types.Package, err error) {
	packages.Visit([]*packages.Package(p), func(pkg *packages.Package) bool {
		if pkg.PkgPath == path {
			tpkg = pkg.Types
			return false
		}
		return true
	}, nil)
	if tpkg != nil {
		return tpkg, nil
	}
	return nil, fmt.Errorf("package %q not found", path)
}
