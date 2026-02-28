// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The unitchecker package defines the main function for an analysis
// driver that analyzes a single compilation unit during a build.
// It is invoked by a build system such as "go vet":
//
//	$ go vet -vettool=$(which vet)
//
// It supports the following command-line protocol:
//
//	-V=full         describe executable               (to the build tool)
//	-flags          describe flags                    (to the build tool)
//	foo.cfg         description of compilation unit (from the build tool)
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

import (
	"archive/zip"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/internal/analysis/driverutil"
	"golang.org/x/tools/internal/facts"
)

// A Config describes a compilation unit to be analyzed.
// It is provided to the tool in a JSON-encoded file
// whose name ends with ".cfg".
type Config struct {
	ID                        string // e.g. "fmt [fmt.test]"
	Compiler                  string // gc or gccgo, provided to MakeImporter
	Dir                       string // (unused)
	ImportPath                string // package path
	GoVersion                 string // minimum required Go version, such as "go1.21.0"
	GoFiles                   []string
	NonGoFiles                []string
	IgnoredFiles              []string
	ModulePath                string            // module path
	ModuleVersion             string            // module version
	ImportMap                 map[string]string // maps import path to package path
	PackageFile               map[string]string // maps package path to file of type information
	Standard                  map[string]bool   // package belongs to standard library
	PackageVetx               map[string]string // maps package path to file of fact information
	VetxOnly                  bool              // run analysis only for facts, not diagnostics
	VetxOutput                string            // where to write file of fact information
	Stdout                    string            // write stdout (e.g. JSON, unified diff) to this file
	FixArchive                string            // write fixed files to this zip archive, if non-empty
	SucceedOnTypecheckFailure bool              // obsolete awful hack; see #18395 and below
}

// Main is the main function of a vet-like analysis tool that must be
// invoked by a build system to analyze a single package.
//
// The protocol required by 'go vet -vettool=...' is that the tool must support:
//
//	-flags          describe flags in JSON
//	-V=full         describe executable for build caching
//	foo.cfg         perform separate modular analyze on the single
//	                unit described by a JSON config file foo.cfg.
//	-fix		don't print each diagnostic, apply its first fix
//	-diff		don't apply a fix, print the diff (requires -fix)
//	-json		print diagnostics and fixes in JSON form
func Main(analyzers ...*analysis.Analyzer) {
	progname := filepath.Base(os.Args[0])
	log.SetFlags(0)
	log.SetPrefix(progname + ": ")

	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `%[1]s is a tool for static analysis of Go programs.

Usage of %[1]s:
	%.16[1]s unit.cfg	# execute analysis specified by config file
	%.16[1]s help    	# general help, including listing analyzers and flags
	%.16[1]s help name	# help on specific analyzer and its flags
`, progname)
		os.Exit(1)
	}

	analyzers = analysisflags.Parse(analyzers, true)

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
	}
	if args[0] == "help" {
		analysisflags.Help(progname, analyzers, args[1:])
		os.Exit(0)
	}
	if len(args) != 1 || !strings.HasSuffix(args[0], ".cfg") {
		log.Fatalf(`invoking "go tool vet" directly is unsupported; use "go vet"`)
	}
	Run(args[0], analyzers)
}

// Run reads the *.cfg file, runs the analysis,
// and calls os.Exit with an appropriate error code.
// It assumes flags have already been set.
func Run(configFile string, analyzers []*analysis.Analyzer) {
	cfg, err := readConfig(configFile)
	if err != nil {
		log.Fatal(err)
	}

	// Redirect stdout to a file as requested.
	if cfg.Stdout != "" {
		f, err := os.Create(cfg.Stdout)
		if err != nil {
			log.Fatal(err)
		}
		os.Stdout = f
	}

	fset := token.NewFileSet()
	results, err := run(fset, cfg, analyzers)
	if err != nil {
		log.Fatal(err)
	}

	code := 0

	// In VetxOnly mode, the analysis is run only for facts.
	if !cfg.VetxOnly {
		code = processResults(fset, cfg.ID, cfg.FixArchive, results)
	}

	os.Exit(code)
}

func readConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
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

func processResults(fset *token.FileSet, id, fixArchive string, results []result) (exit int) {
	if analysisflags.Fix {
		// Don't print the diagnostics,
		// but apply all fixes from the root actions.

		// Convert results to form needed by ApplyFixes.
		fixActions := make([]driverutil.FixAction, len(results))
		for i, res := range results {
			fixActions[i] = driverutil.FixAction{
				Name:         res.a.Name,
				Pkg:          res.pkg,
				Files:        res.files,
				FileSet:      fset,
				ReadFileFunc: os.ReadFile, // TODO(adonovan): respect overlays
				Diagnostics:  res.diagnostics,
			}
		}

		// By default, fixes overwrite the original file.
		// With the -diff flag, print the diffs to stdout.
		// If "go fix" provides a fix archive, we write files
		// into it so that mutations happen after the build.
		write := func(filename string, content []byte) error {
			return os.WriteFile(filename, content, 0644)
		}
		if fixArchive != "" {
			f, err := os.Create(fixArchive)
			if err != nil {
				log.Fatalf("can't create -fix archive: %v", err)
			}
			zw := zip.NewWriter(f)
			zw.SetComment(id) // ignore error
			defer func() {
				if err := zw.Close(); err != nil {
					log.Fatalf("closing -fix archive zip writer: %v", err)
				}
				if err := f.Close(); err != nil {
					log.Fatalf("closing -fix archive file: %v", err)
				}
			}()
			write = func(filename string, content []byte) error {
				f, err := zw.Create(filename)
				if err != nil {
					return err
				}
				_, err = f.Write(content)
				return err
			}
		}

		if err := driverutil.ApplyFixes(fixActions, write, analysisflags.Diff, false); err != nil {
			// Fail when applying fixes failed.
			log.Print(err)
			exit = 1
		}

		// Don't proceed to print text/JSON,
		// and don't report an error
		// just because there were diagnostics.
		return
	}

	// Keep consistent with analogous logic in
	// printDiagnostics in ../internal/checker/checker.go.

	if analysisflags.JSON {
		// JSON output
		tree := make(driverutil.JSONTree)
		for _, res := range results {
			tree.Add(fset, id, res.a.Name, res.diagnostics, res.err)
		}
		tree.Print(os.Stdout) // ignore error

	} else {
		// plain text
		for _, res := range results {
			if res.err != nil {
				log.Println(res.err)
				exit = 1
			}
		}
		for _, res := range results {
			for _, diag := range res.diagnostics {
				driverutil.PrintPlain(os.Stderr, fset, analysisflags.Context, diag)
				exit = 1
			}
		}
	}

	return
}

type factImporter = func(pkgPath string) ([]byte, error)

// These four hook variables are a proof of concept of a future
// parameterization of a unitchecker API that allows the client to
// determine how and where facts and types are produced and consumed.
// (Note that the eventual API will likely be quite different.)
//
// The defaults honor a Config in a manner compatible with 'go vet'.
var (
	makeTypesImporter = func(cfg *Config, fset *token.FileSet) types.Importer {
		compilerImporter := importer.ForCompiler(fset, cfg.Compiler, func(path string) (io.ReadCloser, error) {
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
		return importerFunc(func(importPath string) (*types.Package, error) {
			path, ok := cfg.ImportMap[importPath] // resolve vendoring, etc
			if !ok {
				return nil, fmt.Errorf("can't resolve import %q", path)
			}
			return compilerImporter.Import(path)
		})
	}

	exportTypes = func(*Config, *token.FileSet, *types.Package) error {
		// By default this is a no-op, because "go vet"
		// makes the compiler produce type information.
		return nil
	}

	makeFactImporter = func(cfg *Config) factImporter {
		return func(pkgPath string) ([]byte, error) {
			if vetx, ok := cfg.PackageVetx[pkgPath]; ok {
				return os.ReadFile(vetx)
			}
			return nil, nil // no .vetx file, no facts
		}
	}

	exportFacts = func(cfg *Config, data []byte) error {
		return os.WriteFile(cfg.VetxOutput, data, 0666)
	}
)

func run(fset *token.FileSet, cfg *Config, analyzers []*analysis.Analyzer) ([]result, error) {
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
	tc := &types.Config{
		Importer:  makeTypesImporter(cfg, fset),
		Sizes:     types.SizesFor("gc", build.Default.GOARCH), // TODO(adonovan): use cfg.Compiler
		GoVersion: cfg.GoVersion,
	}
	info := &types.Info{
		Types:        make(map[ast.Expr]types.TypeAndValue),
		Defs:         make(map[*ast.Ident]types.Object),
		Uses:         make(map[*ast.Ident]types.Object),
		Implicits:    make(map[ast.Node]types.Object),
		Instances:    make(map[*ast.Ident]types.Instance),
		Scopes:       make(map[ast.Node]*types.Scope),
		Selections:   make(map[*ast.SelectorExpr]*types.Selection),
		FileVersions: make(map[*ast.File]string),
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
	//
	// TODO(adonovan): fix: the command (and logic!) here are backwards.
	// It should say "...nor is required by any...". (Issue 443099)
	//
	// Also build a map to hold working state and result.
	type action struct {
		once        sync.Once
		result      any
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
	facts, err := facts.NewDecoder(pkg).Decode(makeFactImporter(cfg))
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
			inputs := make(map[*analysis.Analyzer]any)
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

			factFilter := make(map[reflect.Type]bool)
			for _, f := range a.FactTypes {
				factFilter[reflect.TypeOf(f)] = true
			}

			module := &analysis.Module{
				Path:      cfg.ModulePath,
				Version:   cfg.ModuleVersion,
				GoVersion: cfg.GoVersion,
			}

			pass := &analysis.Pass{
				Analyzer:     a,
				Fset:         fset,
				Files:        files,
				OtherFiles:   cfg.NonGoFiles,
				IgnoredFiles: cfg.IgnoredFiles,
				Pkg:          pkg,
				TypesInfo:    info,
				TypesSizes:   tc.Sizes,
				TypeErrors:   nil, // unitchecker doesn't RunDespiteErrors
				ResultOf:     inputs,
				Report: func(d analysis.Diagnostic) {
					// Unitchecker doesn't apply fixes, but it does report them in the JSON output.
					if err := driverutil.ValidateFixes(fset, a, d.SuggestedFixes); err != nil {
						// Since we have diagnostics, the exit code will be nonzero,
						// so logging these errors is sufficient.
						log.Println(err)
						d.SuggestedFixes = nil
					}
					act.diagnostics = append(act.diagnostics, d)
				},
				ImportObjectFact:  facts.ImportObjectFact,
				ExportObjectFact:  facts.ExportObjectFact,
				AllObjectFacts:    func() []analysis.ObjectFact { return facts.AllObjectFacts(factFilter) },
				ImportPackageFact: facts.ImportPackageFact,
				ExportPackageFact: facts.ExportPackageFact,
				AllPackageFacts:   func() []analysis.PackageFact { return facts.AllPackageFacts(factFilter) },
				Module:            module,
			}
			pass.ReadFile = driverutil.CheckedReadFile(pass, os.ReadFile)

			t0 := time.Now()
			act.result, act.err = a.Run(pass)

			if act.err == nil { // resolve URLs on diagnostics.
				for i := range act.diagnostics {
					if url, uerr := driverutil.ResolveURL(a, act.diagnostics[i]); uerr == nil {
						act.diagnostics[i].URL = url
					} else {
						act.err = uerr // keep the last error
					}
				}
			}
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

	// Return diagnostics and errors from root analyzers.
	results := make([]result, len(analyzers))
	for i, a := range analyzers {
		act := actions[a]
		results[i] = result{pkg, files, a, act.diagnostics, act.err}
	}

	data := facts.Encode()
	if err := exportFacts(cfg, data); err != nil {
		return nil, fmt.Errorf("failed to export analysis facts: %v", err)
	}
	if err := exportTypes(cfg, fset, pkg); err != nil {
		return nil, fmt.Errorf("failed to export type information: %v", err)
	}

	return results, nil
}

type result struct {
	pkg         *types.Package
	files       []*ast.File
	a           *analysis.Analyzer
	diagnostics []analysis.Diagnostic
	err         error
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
