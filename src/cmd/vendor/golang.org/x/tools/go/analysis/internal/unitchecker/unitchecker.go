// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The unitchecker package defines the main function for an analysis
// driver that analyzes a single compilation unit during a build.
// It is invoked by a build system such as "go vet":
//
//   $ GOVETTOOL=$(which vet) go vet
//
// It supports the following command-line protocol:
//
//      -V=full         describe executable               (to the build tool)
//      -flags          describe flags                    (to the build tool)
//      foo.cfg         description of compilation unit (from the build tool)
//
// This package does not depend on go/packages.
// If you need a standalone tool, use multichecker,
// which supports this mode but can also load packages
// from source using go/packages.
package unitchecker

// TODO(adonovan):
// - with gccgo, go build does not build standard library,
//   so we will not get to analyze it. Yet we must in order
//   to create base facts for, say, the fmt package for the
//   printf checker.
// - support JSON output, factored with multichecker.

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/facts"
)

// A Config describes a compilation unit to be analyzed.
// It is provided to the tool in a JSON-encoded file
// whose name ends with ".cfg".
type Config struct {
	Compiler                  string
	Dir                       string
	ImportPath                string
	GoFiles                   []string
	OtherFiles                []string // TODO(adonovan): make go vet populate this (github.com/golang/go/issues/27665)
	ImportMap                 map[string]string
	PackageFile               map[string]string
	Standard                  map[string]bool
	PackageVetx               map[string]string
	VetxOnly                  bool
	VetxOutput                string
	SucceedOnTypecheckFailure bool
}

// Main reads the *.cfg file, runs the analysis,
// and calls os.Exit with an appropriate error code.
func Main(configFile string, analyzers []*analysis.Analyzer) {
	cfg, err := readConfig(configFile)
	if err != nil {
		log.Fatal(err)
	}

	fset := token.NewFileSet()
	diags, err := run(fset, cfg, analyzers)
	if err != nil {
		log.Fatal(err)
	}

	if len(diags) > 0 {
		for _, diag := range diags {
			fmt.Fprintf(os.Stderr, "%s: %s\n", fset.Position(diag.Pos), diag.Message)
		}
		os.Exit(1)
	}

	os.Exit(0)
}

func readConfig(filename string) (*Config, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	cfg := new(Config)
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("cannot decode JSON config file %s: %v", filename, err)
	}
	if len(cfg.GoFiles) == 0 {
		// The go command disallows packages with no files.
		// The only exception is unsafe, but the go command
		// doesn't call vet on it.
		return nil, fmt.Errorf("package has no files: %s", cfg.ImportPath)
	}
	return cfg, nil
}

func run(fset *token.FileSet, cfg *Config, analyzers []*analysis.Analyzer) ([]analysis.Diagnostic, error) {
	// Load, parse, typecheck.
	var files []*ast.File
	for _, name := range cfg.GoFiles {
		f, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if err != nil {
			if cfg.SucceedOnTypecheckFailure {
				// Silently succeed; let the compiler
				// report parse errors.
				err = nil
			}
			return nil, err
		}
		files = append(files, f)
	}
	compilerImporter := importer.For(cfg.Compiler, func(path string) (io.ReadCloser, error) {
		// path is a resolved package path, not an import path.
		file, ok := cfg.PackageFile[path]
		if !ok {
			if cfg.Compiler == "gccgo" && cfg.Standard[path] {
				return nil, nil // fall back to default gccgo lookup
			}
			return nil, fmt.Errorf("no package file for %q", path)
		}
		return os.Open(file)
	})
	importer := importerFunc(func(importPath string) (*types.Package, error) {
		path, ok := cfg.ImportMap[importPath] // resolve vendoring, etc
		if !ok {
			return nil, fmt.Errorf("can't resolve import %q", path)
		}
		return compilerImporter.Import(path)
	})
	tc := &types.Config{
		Importer: importer,
		Sizes:    types.SizesFor("gc", build.Default.GOARCH), // assume gccgo ≡ gc?
	}
	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	pkg, err := tc.Check(cfg.ImportPath, fset, files, info)
	if err != nil {
		if cfg.SucceedOnTypecheckFailure {
			// Silently succeed; let the compiler
			// report type errors.
			err = nil
		}
		return nil, err
	}

	// Register fact types with gob.
	// In VetxOnly mode, analyzers are only for their facts,
	// so we can skip any analysis that neither produces facts
	// nor depends on any analysis that produces facts.
	// Also build a map to hold working state and result.
	type action struct {
		once        sync.Once
		result      interface{}
		err         error
		usesFacts   bool // (transitively uses)
		diagnostics []analysis.Diagnostic
	}
	actions := make(map[*analysis.Analyzer]*action)
	var registerFacts func(a *analysis.Analyzer) bool
	registerFacts = func(a *analysis.Analyzer) bool {
		act, ok := actions[a]
		if !ok {
			act = new(action)
			var usesFacts bool
			for _, f := range a.FactTypes {
				usesFacts = true
				gob.Register(f)
			}
			for _, req := range a.Requires {
				if registerFacts(req) {
					usesFacts = true
				}
			}
			act.usesFacts = usesFacts
			actions[a] = act
		}
		return act.usesFacts
	}
	var filtered []*analysis.Analyzer
	for _, a := range analyzers {
		if registerFacts(a) || !cfg.VetxOnly {
			filtered = append(filtered, a)
		}
	}
	analyzers = filtered

	// Read facts from imported packages.
	read := func(path string) ([]byte, error) {
		if vetx, ok := cfg.PackageVetx[path]; ok {
			return ioutil.ReadFile(vetx)
		}
		return nil, nil // no .vetx file, no facts
	}
	facts, err := facts.Decode(pkg, read)
	if err != nil {
		return nil, err
	}

	// In parallel, execute the DAG of analyzers.
	var exec func(a *analysis.Analyzer) *action
	var execAll func(analyzers []*analysis.Analyzer)
	exec = func(a *analysis.Analyzer) *action {
		act := actions[a]
		act.once.Do(func() {
			execAll(a.Requires) // prefetch dependencies in parallel

			// The inputs to this analysis are the
			// results of its prerequisites.
			inputs := make(map[*analysis.Analyzer]interface{})
			var failed []string
			for _, req := range a.Requires {
				reqact := exec(req)
				if reqact.err != nil {
					failed = append(failed, req.String())
					continue
				}
				inputs[req] = reqact.result
			}

			// Report an error if any dependency failed.
			if failed != nil {
				sort.Strings(failed)
				act.err = fmt.Errorf("failed prerequisites: %s", strings.Join(failed, ", "))
				return
			}

			pass := &analysis.Pass{
				Analyzer:          a,
				Fset:              fset,
				Files:             files,
				OtherFiles:        cfg.OtherFiles,
				Pkg:               pkg,
				TypesInfo:         info,
				ResultOf:          inputs,
				Report:            func(d analysis.Diagnostic) { act.diagnostics = append(act.diagnostics, d) },
				ImportObjectFact:  facts.ImportObjectFact,
				ExportObjectFact:  facts.ExportObjectFact,
				ImportPackageFact: facts.ImportPackageFact,
				ExportPackageFact: facts.ExportPackageFact,
			}

			t0 := time.Now()
			act.result, act.err = a.Run(pass)
			if false {
				log.Printf("analysis %s = %s", pass, time.Since(t0))
			}
		})
		return act
	}
	execAll = func(analyzers []*analysis.Analyzer) {
		var wg sync.WaitGroup
		for _, a := range analyzers {
			wg.Add(1)
			go func(a *analysis.Analyzer) {
				_ = exec(a)
				wg.Done()
			}(a)
		}
		wg.Wait()
	}

	execAll(analyzers)

	// Return diagnostics from root analyzers.
	var diags []analysis.Diagnostic
	for _, a := range analyzers {
		act := actions[a]
		if act.err != nil {
			return nil, act.err // some analysis failed
		}
		diags = append(diags, act.diagnostics...)
	}

	data := facts.Encode()
	if err := ioutil.WriteFile(cfg.VetxOutput, data, 0666); err != nil {
		return nil, fmt.Errorf("failed to write analysis facts: %v", err)
	}

	return diags, nil
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
