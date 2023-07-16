// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unitchecker_test

// This file illustrates separate analysis with an example.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/token"
	"go/types"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"

	"golang.org/x/tools/go/analysis/passes/printf"
	"golang.org/x/tools/go/analysis/unitchecker"
	"golang.org/x/tools/go/gcexportdata"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/txtar"
)

// ExampleSeparateAnalysis demonstrates the principle of separate
// analysis, the distribution of units of type-checking and analysis
// work across several processes, using serialized summaries to
// communicate between them.
//
// It uses two different kinds of task, "manager" and "worker":
//
//   - The manager computes the graph of package dependencies, and makes
//     a request to the worker for each package. It does not parse,
//     type-check, or analyze Go code. It is analogous "go vet".
//
//   - The worker, which contains the Analyzers, reads each request,
//     loads, parses, and type-checks the files of one package,
//     applies all necessary analyzers to the package, then writes
//     its results to a file. It is a unitchecker-based driver,
//     analogous to the program specified by go vet -vettool= flag.
//
// In practice these would be separate executables, but for simplicity
// of this example they are provided by one executable in two
// different modes: the Example function is the manager, and the same
// executable invoked with ENTRYPOINT=worker is the worker.
// (See TestIntegration for how this happens.)
func ExampleSeparateAnalysis() {
	// src is an archive containing a module with a printf mistake.
	const src = `
-- go.mod --
module separate
go 1.18

-- main/main.go --
package main

import "separate/lib"

func main() {
	lib.MyPrintf("%s", 123)
}

-- lib/lib.go --
package lib

import "fmt"

func MyPrintf(format string, args ...any) {
	fmt.Printf(format, args...)
}
`

	// Expand archive into tmp tree.
	tmpdir, err := os.MkdirTemp("", "SeparateAnalysis")
	if err != nil {
		log.Fatal(err)
	}
	if err := extractTxtar(txtar.Parse([]byte(src)), tmpdir); err != nil {
		log.Fatal(err)
	}

	// Load metadata for the main package and all its dependencies.
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedModule,
		Dir:  tmpdir,
		Env: append(os.Environ(),
			"GOPROXY=off", // disable network
			"GOWORK=off",  // an ambient GOWORK value would break package loading
		),
	}
	pkgs, err := packages.Load(cfg, "separate/main")
	if err != nil {
		log.Fatal(err)
	}
	// Stop if any package had a metadata error.
	if packages.PrintErrors(pkgs) > 0 {
		os.Exit(1)
	}

	// Now we have loaded the import graph,
	// let's begin the proper work of the manager.

	// Gather root packages. They will get all analyzers,
	// whereas dependencies get only the subset that
	// produce facts or are required by them.
	roots := make(map[*packages.Package]bool)
	for _, pkg := range pkgs {
		roots[pkg] = true
	}

	// nextID generates sequence numbers for each unit of work.
	// We use it to create names of temporary files.
	var nextID atomic.Int32

	// Visit all packages in postorder: dependencies first.
	// TODO(adonovan): opt: use parallel postorder.
	packages.Visit(pkgs, nil, func(pkg *packages.Package) {
		if pkg.PkgPath == "unsafe" {
			return
		}

		// Choose a unique prefix for temporary files
		// (.cfg .types .facts) produced by this package.
		// We stow it in an otherwise unused field of
		// Package so it can be accessed by our importers.
		prefix := fmt.Sprintf("%s/%d", tmpdir, nextID.Add(1))
		pkg.ExportFile = prefix

		// Construct the request to the worker.
		var (
			importMap   = make(map[string]string)
			packageFile = make(map[string]string)
			packageVetx = make(map[string]string)
		)
		for importPath, dep := range pkg.Imports {
			importMap[importPath] = dep.PkgPath
			if depPrefix := dep.ExportFile; depPrefix != "" { // skip "unsafe"
				packageFile[dep.PkgPath] = depPrefix + ".types"
				packageVetx[dep.PkgPath] = depPrefix + ".facts"
			}
		}
		cfg := unitchecker.Config{
			ID:           pkg.ID,
			ImportPath:   pkg.PkgPath,
			GoFiles:      pkg.CompiledGoFiles,
			NonGoFiles:   pkg.OtherFiles,
			IgnoredFiles: pkg.IgnoredFiles,
			ImportMap:    importMap,
			PackageFile:  packageFile,
			PackageVetx:  packageVetx,
			VetxOnly:     !roots[pkg],
			VetxOutput:   prefix + ".facts",
		}
		if pkg.Module != nil {
			if v := pkg.Module.GoVersion; v != "" {
				cfg.GoVersion = "go" + v
			}
		}

		// Write the JSON configuration message to a file.
		cfgData, err := json.Marshal(cfg)
		if err != nil {
			log.Fatal(err)
		}
		cfgFile := prefix + ".cfg"
		if err := os.WriteFile(cfgFile, cfgData, 0666); err != nil {
			log.Fatal(err)
		}

		// Send the request to the worker.
		cmd := exec.Command(os.Args[0], "-json", cfgFile)
		cmd.Stderr = os.Stderr
		cmd.Stdout = new(bytes.Buffer)
		cmd.Env = append(os.Environ(), "ENTRYPOINT=worker")
		if err := cmd.Run(); err != nil {
			log.Fatal(err)
		}

		// Parse JSON output and print plainly.
		dec := json.NewDecoder(cmd.Stdout.(io.Reader))
		for {
			type jsonDiagnostic struct {
				Posn    string `json:"posn"`
				Message string `json:"message"`
			}
			// 'results' maps Package.Path -> Analyzer.Name -> diagnostics
			var results map[string]map[string][]jsonDiagnostic
			if err := dec.Decode(&results); err != nil {
				if err == io.EOF {
					break
				}
				log.Fatal(err)
			}
			for _, result := range results {
				for analyzer, diags := range result {
					for _, diag := range diags {
						rel := strings.ReplaceAll(diag.Posn, tmpdir, "")
						rel = filepath.ToSlash(rel)
						fmt.Printf("%s: [%s] %s\n",
							rel, analyzer, diag.Message)
					}
				}
			}
		}
	})

	// Observe that the example produces a fact-based diagnostic
	// from separate analysis of "main", "lib", and "fmt":

	// Output:
	// /main/main.go:6:2: [printf] separate/lib.MyPrintf format %s has arg 123 of wrong type int
}

// -- worker process --

// worker is the main entry point for a unitchecker-based driver
// with only a single analyzer, for illustration.
func worker() {
	// Currently the unitchecker API doesn't allow clients to
	// control exactly how and where fact and type information
	// is produced and consumed.
	//
	// So, for example, it assumes that type information has
	// already been produced by the compiler, which is true when
	// running under "go vet", but isn't necessary. It may be more
	// convenient and efficient for a distributed analysis system
	// if the worker generates both of them, which is the approach
	// taken in this example; they could even be saved as two
	// sections of a single file.
	//
	// Consequently, this test currently needs special access to
	// private hooks in unitchecker to control how and where facts
	// and types are produced and consumed. In due course this
	// will become a respectable public API. In the meantime, it
	// should at least serve as a demonstration of how one could
	// fork unitchecker to achieve separate analysis without go vet.
	unitchecker.SetTypeImportExport(makeTypesImporter, exportTypes)

	unitchecker.Main(printf.Analyzer)
}

func makeTypesImporter(cfg *unitchecker.Config, fset *token.FileSet) types.Importer {
	imports := make(map[string]*types.Package)
	return importerFunc(func(importPath string) (*types.Package, error) {
		// Resolve import path to package path (vendoring, etc)
		path, ok := cfg.ImportMap[importPath]
		if !ok {
			return nil, fmt.Errorf("can't resolve import %q", path)
		}
		if path == "unsafe" {
			return types.Unsafe, nil
		}

		// Find, read, and decode file containing type information.
		file, ok := cfg.PackageFile[path]
		if !ok {
			return nil, fmt.Errorf("no package file for %q", path)
		}
		f, err := os.Open(file)
		if err != nil {
			return nil, err
		}
		defer f.Close() // ignore error
		return gcexportdata.Read(f, fset, imports, path)
	})
}

func exportTypes(cfg *unitchecker.Config, fset *token.FileSet, pkg *types.Package) error {
	var out bytes.Buffer
	if err := gcexportdata.Write(&out, fset, pkg); err != nil {
		return err
	}
	typesFile := strings.TrimSuffix(cfg.VetxOutput, ".facts") + ".types"
	return os.WriteFile(typesFile, out.Bytes(), 0666)
}

// -- helpers --

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

// extractTxtar writes each archive file to the corresponding location beneath dir.
//
// TODO(adonovan): move this to txtar package, we need it all the time (#61386).
func extractTxtar(ar *txtar.Archive, dir string) error {
	for _, file := range ar.Files {
		name := filepath.Join(dir, file.Name)
		if err := os.MkdirAll(filepath.Dir(name), 0777); err != nil {
			return err
		}
		if err := os.WriteFile(name, file.Data, 0666); err != nil {
			return err
		}
	}
	return nil
}
