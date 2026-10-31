// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The unitchecker package defines the main function for an analysis
// driver that analyzes a single compilation unit during a build.
// It is invoked by a build system such as "go vet":
//
//	$ go build -o my.exe example.com/my
//	$ go vet -vettool=./my.exe [packages...]
//
// It may be executed by other build systems such as Bazel and Blaze.
//
// It supports the following command-line protocol:
//
//	-V=full         describe executable               (to the build tool)
//	-flags          describe flags                    (to the build tool)
//	-json           emit results in JSON form         (to the build tool)
//	foo.cfg         description of compilation unit (from the build tool)
//
// The build tool discovers the remaining flags by running the tool
// with -flags and forwards those that the user set. They include a
// flag for each analyzer, such as -printf, that enables or disables
// it, and the additional flags of unitchecker itself (e.g. -json).
//
// The sole non-flag argument foo.cfg names a file containing a
// JSON-encoded [Config] specifying the work to be done.
//
// The -json flag causes the tool to write its diagnostics, and any
// fixes they suggest, to its standard output as a JSON object keyed
// by the ID of the unit and then by the name of the analyzer, instead
// of printing them as plain text to standard error.
// The build tool may redirect this output to a file using
// [Config.Stdout], so that it may be cached along with the other
// outputs of the action.
//
// The build system must run the tool once for each unit of compilation,
// dependencies first.
// Each run reads the packages's source files, plus a summary file for each
// direct import, then writes the summary file (including diagnostics and
// fixes) for the current package.
// A run is thus (ideally) a deterministic pure function from files to
// files, so the build system may treat it as an ordinary build action,
// running it in parallel with other actions, caching its result,
// or skipping it entirely if none of its inputs have
// changed--just like a compiler.
// See the Drivers section of the [golang.org/x/tools/go/analysis]
// package documentation for the contrast with whole-program drivers.
//
// This package does not use [golang.org/x/tools.go/packages] since it
// runs within a build action and cannot start a metadata query of its own.
// Use this package directly only for a tool that is always invoked
// by a build system.
// For convenience, commands built with singlechecker or
// [golang.org/x/tools/go/analysis/multichecker] support this protocol as
// well, so they can detect when they are being invoked by a build tool
// and run in package-at-a-time mode.
package unitchecker

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
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
	"golang.org/x/tools/internal/gcimporter"
)

// A Config describes a compilation unit to be analyzed.
// It is provided to the tool in a JSON-encoded file
// whose name ends with ".cfg".
//
// Config is the interface between this driver and the build system
// that invokes it. The go command writes one for each unit that it
// vets, and any other build system that runs a unitchecker-based
// tool, such as Bazel's nogo, must do the same.
type Config struct {
	// ID is the build system's name for this unit, such as
	// "fmt" or "fmt [fmt.test]". It appears in diagnostics.
	ID string

	Compiler string // (unused)
	Dir      string // (unused)

	// ImportPath is the path of the package being analyzed.
	ImportPath string

	// GoVersion is the minimum Go version required by this unit,
	// such as "go1.21.0". It determines which language features
	// the type checker permits.
	GoVersion string

	// GoFiles are the names of the Go source files of the unit.
	// They are parsed and type-checked, and are reported to each
	// analyzer as Pass.Files. The list must not be empty.
	GoFiles []string

	// NonGoFiles are the names of the other files of the unit,
	// such as assembly. They are not read by the driver, but
	// their names are reported as Pass.OtherFiles, and analyzers
	// may read them using Pass.ReadFile.
	NonGoFiles []string

	// IgnoredFiles are the names of files, Go or not, that belong
	// to the package in some other build configuration but not
	// this one. They are reported as Pass.IgnoredFiles.
	IgnoredFiles []string

	ModulePath    string // Deprecated: redundant w.r.t. Module.Path in go1.27; remove after go1.28.
	ModuleVersion string // Deprecated: redundant w.r.t. Module.Version in go1.27; remove after go1.28.

	// Module describes the module to which the unit belongs,
	// or nil if it belongs to none. It is reported as Pass.Module.
	Module *analysis.Module

	// ImportMap maps each import path that appears in the source
	// files of the unit to the path of the package it denotes,
	// which may differ from it because of vendoring.
	ImportMap map[string]string

	PackageFile map[string]string // (unused)

	// Standard records, for each package path, whether the
	// package belongs to the standard library.
	Standard map[string]bool

	// PackageVetx maps the path of each package imported by this
	// unit to the name of the file, written by the analysis of
	// that package, that holds its export data and facts.
	PackageVetx map[string]string

	// VetxOnly indicates that the unit is being analyzed only so
	// that its importers may have its facts, and that its
	// diagnostics will be discarded. Analyzers that neither
	// export facts nor are required by one that does may be
	// skipped.
	VetxOnly bool

	// VetxOutput is the name of the file to which the export data
	// and facts of this unit should be written, for the analysis
	// of its importers to read. Nothing is written if it is empty.
	VetxOutput string

	// Stdout redirects the standard output of the tool, such as
	// the JSON tree or the unified diff, to the named file.
	// If empty, the process's standard output is used.
	Stdout string

	// FixArchive, if non-empty, names a zip file to which the -fix
	// flag should write the fixed files, instead of overwriting
	// the originals. This allows a build system to defer the
	// mutation of the source tree until after the build.
	FixArchive string

	SucceedOnTypecheckFailure bool // (unused)
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
		log.Fatalf(`invoking "go tool %[1]s" directly is unsupported; use "go %[1]s"`, progname)
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
		if cfg.VetxOnly {
			os.Exit(1)
		}
		// Print known diagnostic error types without the log prefix.
		// (See go.dev/issue/34142.)
		switch err := err.(type) {
		case types.Error, *scanner.Error, scanner.ErrorList:
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
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

func run(fset *token.FileSet, cfg *Config, analyzers []*analysis.Analyzer) ([]result, error) {
	// Load, parse, typecheck.
	var files []*ast.File
	for _, name := range cfg.GoFiles {
		f, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}

	// Read all direct imports' vetx files (for types and facts).
	vetxEntries, err := readVetxFiles(cfg.PackageVetx)
	if err != nil {
		return nil, err
	}

	// Construct the type importer.
	imports := make(map[string]*types.Package)
	importer := importerFunc(func(importPath string) (*types.Package, error) {
		path, ok := cfg.ImportMap[importPath] // resolve vendoring, etc
		if !ok {
			return nil, fmt.Errorf("can't resolve import %q", importPath)
		}
		if path == "unsafe" {
			return types.Unsafe, nil
		}
		if pkg, ok := imports[path]; ok && pkg.Complete() {
			return pkg, nil
		}
		entry, ok := vetxEntries[path]
		if !ok {
			return nil, fmt.Errorf("no package vetx file for %q", path)
		}
		return gcimporter.IImportData(fset, imports, entry.types, path)
	})

	tc := &types.Config{
		Importer:  importer,
		Sizes:     types.SizesFor("gc", build.Default.GOARCH),
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

	readFacts := func(pkgPath string) ([]byte, error) {
		entry, ok := vetxEntries[pkgPath]
		if !ok {
			// no .vetx file, no facts
			if pkgPath == "unsafe" {
				return nil, nil
			}
			return nil, fmt.Errorf("missing facts for %q", pkgPath)
		}
		return entry.facts, nil
	}

	// Read facts from imported packages.
	// The importer's map contains every package (perhaps indirect)
	// that could own an object about which facts may be queried.
	getPackage := func(path string) *types.Package { return imports[path] }
	facts, err := facts.NewDecoder(pkg, getPackage).Decode(readFacts)
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

			module := cfg.Module
			// cmd/go vet prior to go1.27 did not populate cfg.Module. Do our best.
			if module == nil && cfg.ModulePath != "" {
				module = &analysis.Module{
					Path:      cfg.ModulePath,
					Version:   cfg.ModuleVersion,
					GoVersion: cfg.GoVersion,
				}
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

	// Export types and facts.
	if cfg.VetxOutput != "" {
		if err := writeVetxFile(fset, pkg, cfg.VetxOutput, facts); err != nil {
			return nil, fmt.Errorf("type+fact export failed: %v", err)
		}
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

// -- vetx file --

const vetxMagic = "vetx"

type vetxEntry struct {
	types, facts []byte
}

func readVetxFiles(packageVetx map[string]string) (map[string]*vetxEntry, error) {
	var (
		mu       sync.Mutex // guards entries and firstErr
		entries  = make(map[string]*vetxEntry, len(packageVetx))
		firstErr error
		wg       sync.WaitGroup // (not errgroup, since unitchecker is vendored into std)
	)
	for pkgPath, file := range packageVetx {
		wg.Go(func() {
			entry, err := readVetxFile(file)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("invalid vetx file for %q: %v", pkgPath, err)
				}
				return
			}
			entries[pkgPath] = entry
		})
	}
	wg.Wait()
	return entries, firstErr
}

func readVetxFile(filename string) (*vetxEntry, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	if len(data) < 12 {
		return nil, fmt.Errorf("vetx file %s too short", filename)
	}
	if magic := string(data[:4]); magic != vetxMagic {
		return nil, fmt.Errorf("vetx files %s has bad magic: %04x", filename, magic)
	}
	typesLen := binary.LittleEndian.Uint32(data[4:8])
	factsLen := binary.LittleEndian.Uint32(data[8:12])
	if uint64(12)+uint64(typesLen)+uint64(factsLen) > uint64(len(data)) {
		return nil, fmt.Errorf("invalid vetx file lengths (header claims 12+%d+%d bytes, got %d)", typesLen, factsLen, len(data))
	}
	return &vetxEntry{
		types: data[12 : 12+typesLen],
		facts: data[12+typesLen : 12+typesLen+factsLen],
	}, nil
}

func writeVetxFile(fset *token.FileSet, pkg *types.Package, filename string, facts *facts.Set) error {
	var buf bytes.Buffer
	buf.WriteString(vetxMagic)
	buf.Write(make([]byte, 8)) // placeholder for (types, facts) length fields

	// types
	startTypes := buf.Len()
	if err := gcimporter.IExportData(&buf, fset, pkg); err != nil {
		return err
	}
	typesLen := buf.Len() - startTypes

	// facts
	factsData := facts.Encode()
	buf.Write(factsData)
	factsLen := len(factsData)

	// Patch the length fields and write it out.
	data := buf.Bytes()
	binary.LittleEndian.PutUint32(data[4:8], uint32(typesLen))
	binary.LittleEndian.PutUint32(data[8:12], uint32(factsLen))
	return os.WriteFile(filename, data, 0666)
}
