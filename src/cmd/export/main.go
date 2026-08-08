// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The export command writes type information of Go programs.
// It should not be relied upon except by go list -export.
//
// It supports the following command-line protocol:
//
//	-V=full			print tool version
//	unit.cfg		description of unit to export
package main

import (
	"cmd/internal/objabi"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/gcexportdata"
)

const message = `export is a tool to write type information of Go programs.

Note that this tool is experimental and its interface is subject to change.
It should not be depended upon directly.

Usage of 'go tool export':
Export type information specified by a config file:
	go tool export unit.cfg
`

func usage() {
	fmt.Fprint(os.Stderr, message)
	fmt.Fprintln(os.Stderr, "\nFlags:")
	flag.PrintDefaults()
	os.Exit(2)
}

// config defines the JSON schema of the file passed by 'go list -export'.
type config struct {
	ImportPath  string            // package path
	Compiler    string            // gc or gccgo, provided to makeTypesImporter
	GoVersion   string            // minimum required Go version, such as "go1.21.0"
	GoFiles     []string          // absolute paths to package source files
	ImportMap   map[string]string // maps import path to package path
	PackageFile map[string]string // maps package path to file of type information
	Output      string            // where to write file of type information
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("export: ")

	objabi.AddVersionFlag()
	objabi.Flagparse(usage)

	args := flag.Args()
	if len(args) != 1 || !strings.HasSuffix(args[0], ".cfg") {
		flag.Usage()
	}
	if err := export(args[0]); err != nil {
		log.Fatal(err)
	}
}

func export(configFile string) error {
	cfg, err := readConfig(configFile)
	if err != nil {
		return err
	}

	// TODO: Determine if it's useful for go list -export to return partial type
	// information, perhaps with the -e flag.
	// Load, parse, type check.
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range cfg.GoFiles {
		f, err := parser.ParseFile(fset, name, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		files = append(files, f)
	}
	tc := &types.Config{
		Importer:  makeTypesImporter(cfg, fset),
		Sizes:     types.SizesFor(cfg.Compiler, build.Default.GOARCH),
		GoVersion: cfg.GoVersion,
	}
	pkg, err := tc.Check(cfg.ImportPath, fset, files, nil)
	if err != nil {
		return err
	}

	// Write export data.
	f, err := os.Create(cfg.Output)
	if err != nil {
		return err
	}
	if err := gcexportdata.Write(f, fset, pkg); err != nil {
		f.Close() // ignore error
		return err
	}
	return f.Close()
}

func readConfig(filename string) (*config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	cfg := new(config)
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("can't decode input %s: %v", filename, err)
	}
	if len(cfg.GoFiles) == 0 {
		return nil, fmt.Errorf("no files in package %s", cfg.ImportPath)
	}
	return cfg, nil
}

func makeTypesImporter(cfg *config, fset *token.FileSet) types.Importer {
	imports := make(map[string]*types.Package)
	imports["unsafe"] = types.Unsafe
	return importerFunc(func(importPath string) (*types.Package, error) {
		pkgPath, ok := cfg.ImportMap[importPath]
		if !ok {
			return nil, fmt.Errorf("can't resolve import %s", importPath)
		}
		// Check for cache hit.
		if pkg, ok := imports[pkgPath]; ok && pkg.Complete() {
			return pkg, nil
		}
		// Miss.
		cfgPath, ok := cfg.PackageFile[pkgPath]
		if !ok {
			return nil, fmt.Errorf("can't resolve export %s", pkgPath)
		}
		f, err := os.Open(cfgPath)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return gcexportdata.Read(f, fset, imports, pkgPath)
	})
}

type importerFunc func(string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
