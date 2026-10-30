// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker

// This file defines the analysis of a single package.

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"go/analysis"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"internal/analysis/gotestchecker/protocol"
	"log"
	"maps"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/go/gcexportdata"
)

// run analyzes the package described by the named request file (see
// [protocol.AnalyzeRequest]).
//
// A failure of an analyzer on this package (an error or a panic) does
// not cause run to fail: it is recorded in the analyzer's pass file,
// and reported, once, by the go command. The passes that cannot run
// because of it (or because of type errors in the package) are marked
// as skipped. Only problems with the tool's inputs cause run to fail.
func run(configFile string, vars []Var) error {
	data, err := os.ReadFile(configFile)
	if err != nil {
		return err
	}
	areq := new(protocol.AnalyzeRequest)
	if err := json.Unmarshal(data, areq); err != nil {
		return fmt.Errorf("cannot decode JSON config file %s: %v", configFile, err)
	}
	prof, err := startProfiler(areq)
	if err != nil {
		return err
	}
	if err := analyze(areq, vars, prof); err != nil {
		prof.finish() // ignore error
		return err
	}
	return prof.finish()
}

// analyze analyzes the package described by areq (see [run]),
// recording its costs in prof, if non-nil.
func analyze(areq *protocol.AnalyzeRequest, vars []Var, prof *profiler) error {
	g := newGraph(vars)
	prof.setAnalyzers(g.analyzers)
	analyzer := func(i int) (*analysis.Analyzer, error) {
		if !(0 <= i && i < len(g.analyzers)) {
			return nil, fmt.Errorf("invalid analyzer index %d", i)
		}
		return g.analyzers[i], nil
	}

	// Identify the analyzers whose results are required.
	var outputs []*analysis.Analyzer
	for _, i := range slices.Sorted(maps.Keys(areq.Outputs)) {
		a, err := analyzer(i)
		if err != nil {
			return err
		}
		outputs = append(outputs, a)
	}
	report := make(map[*analysis.Analyzer]bool)
	for _, i := range areq.Report {
		a, err := analyzer(i)
		if err != nil {
			return err
		}
		report[a] = true
	}

	// If there is nothing to do, avoid type checking.
	if len(outputs) == 0 && areq.TypeExport == "" {
		return nil
	}

	// Parse and type-check. Type errors are written to
	// AnalyzeRequest.TypeErrors, for the go command to report once, and
	// cause every analyzer to be skipped.
	var typeErrs []string
	fset := token.NewFileSet()
	var files []*ast.File
	if err := prof.doErr(phaseParse, -1, func() error {
		for _, name := range areq.GoFiles {
			// Analyzers must use go/types, not the
			// syntactic object resolution of go/parser.
			f, err := parser.ParseFile(fset, name, nil, parser.ParseComments|parser.SkipObjectResolution)
			if err != nil {
				if f == nil {
					return err // can't read file
				}
				typeErrs = append(typeErrs, err.Error())
			}
			files = append(files, f)
		}
		return nil
	}); err != nil {
		return err
	}
	imports := make(map[string]*types.Package)
	importer := importerFunc(func(importPath string) (*types.Package, error) {
		path, ok := areq.ImportMap[importPath] // resolve vendoring, etc
		if !ok {
			return nil, fmt.Errorf("can't resolve import %q", importPath)
		}
		if path == "unsafe" {
			return types.Unsafe, nil
		}
		if pkg, ok := imports[path]; ok && pkg.Complete() {
			return pkg, nil
		}
		file, ok := areq.TypeImports[path]
		if !ok {
			return nil, fmt.Errorf("no types file for %q", path)
		}
		// TODO(adonovan): opt: parallelize reads for all imports before type checking.
		data, err := os.ReadFile(file)
		if err != nil {
			return nil, err
		}
		return gcexportdata.Read(bytes.NewReader(data), fset, imports, path)
	})
	tc := &types.Config{
		Importer:  importer,
		Sizes:     types.SizesFor("gc", areq.GOARCH),
		GoVersion: areq.GoVersion,
		Error: func(err error) {
			typeErrs = append(typeErrs, err.Error())
		},
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
	var pkg *types.Package
	prof.do(phaseTypecheck, -1, func() {
		pkg, _ = tc.Check(areq.ImportPath, fset, files, info) // errors are in typeErrs
	})

	const maxTypeErrors = 10
	if len(typeErrs) > maxTypeErrors {
		typeErrs = append(typeErrs[:maxTypeErrors], "too many errors")
	}

	// Export types (even if there were type errors, so that the
	// analysis of importers can proceed; their analyzers that
	// depend on facts from this package will fail), and the type
	// errors, if any.
	if areq.TypeErrors != "" {
		data, err := json.Marshal(typeErrs)
		if err != nil {
			log.Fatal(err) // can't happen
		}
		if err := os.WriteFile(areq.TypeErrors, data, 0666); err != nil {
			return err
		}
	}
	if areq.TypeExport != "" {
		if err := prof.doErr(phaseExport, -1, func() error {
			var buf bytes.Buffer
			if err := gcexportdata.Write(&buf, fset, pkg); err != nil {
				return fmt.Errorf("type export failed: %v", err)
			}
			return os.WriteFile(areq.TypeExport, buf.Bytes(), 0666)
		}); err != nil {
			return err
		}
	}
	if len(outputs) == 0 {
		return nil
	}

	// Compute the Requires closure of the outputs.
	requires := make(map[*analysis.Analyzer]bool)
	var visit func(a *analysis.Analyzer)
	visit = func(a *analysis.Analyzer) {
		if !requires[a] {
			requires[a] = true
			for _, req := range a.Requires {
				visit(req)
			}
		}
	}
	for _, a := range outputs {
		visit(a)
	}

	// Decode a separate fact set for each analyzer in the Requires closure,
	// from that analyzer's pass files for each direct import.
	// (Analyzers that use no facts get an empty set.)
	// If the analyzer failed or was skipped on a direct import, it
	// is skipped here.
	var (
		getPackage = func(path string) *types.Package { return imports[path] }
		decoder    = newFactDecoder(pkg, getPackage)
		sets       = make(map[*analysis.Analyzer]*factSet)
		depSkipped = make(map[*analysis.Analyzer]bool)
	)
	// (Each analyzer is charged for decoding its facts.)
	for _, a := range g.analyzers {
		if !requires[a] {
			continue
		}
		if err := prof.doErr(phaseFacts, g.index[a], func() error {
			files := areq.Facts[g.index[a]] // (only for analyzers that use facts)
			// Read each pass file once. A pass file with a
			// failure, or of a skipped pass, has no facts.
			//
			// TODO(adonovan): opt: parallelize reads of pass files.
			passes := make(map[string]*protocol.Pass)
			for path, file := range files {
				pf, err := protocol.ReadPass(file)
				if err != nil {
					return err
				}
				passes[path] = pf
				prof.factPassFile(g.index[a], len(pf.Facts) == 0 && pf.Failure == nil && !pf.Skipped)
				if pf.Failure != nil || pf.Skipped {
					depSkipped[a] = true
				}
			}
			if depSkipped[a] {
				return nil
			}
			for _, f := range a.FactTypes {
				registerFactType(f)
			}
			set, decoded, discarded, err := decoder.Decode(func(pkgPath string) ([]byte, error) {
				var data []byte // nil for e.g. unsafe
				if pf := passes[pkgPath]; pf != nil {
					data = pf.Facts
				}
				prof.factsIn(g.index[a], len(data))
				return data, nil
			})
			if err != nil {
				return err
			}
			prof.factsDecoded(g.index[a], decoded, discarded)
			sets[a] = set
			return nil
		}); err != nil {
			return err
		}
	}

	// Execute the DAG of analyzers in the Requires closure, in parallel:
	// each analyzer runs as soon as its prerequisites are done.
	type action struct {
		a           *analysis.Analyzer
		preds       []*action    // actions of the analyzers that require this one
		pending     atomic.Int32 // number of prerequisites not yet done
		result      any
		failure     *protocol.Failure // failure of this pass
		skipped     bool              // pass did not run (see protocol.Pass.Skipped)
		diagnostics []analysis.Diagnostic
	}
	var (
		mu      sync.Mutex // guards action.diagnostics
		actions = make(map[*analysis.Analyzer]*action)
	)
	for _, a := range g.analyzers {
		if requires[a] {
			act := &action{a: a}
			act.pending.Store(int32(len(a.Requires)))
			for _, req := range a.Requires {
				reqact := actions[req] // (prerequisites come first)
				reqact.preds = append(reqact.preds, act)
			}
			actions[a] = act
		}
	}
	exec := func(act *action) {
		a := act.a
		// The analyzer is skipped if a prerequisite failed or was
		// skipped, if it failed or was skipped on a dependency, or
		// if the package has type errors.
		inputs := make(map[*analysis.Analyzer]any)
		for _, req := range a.Requires {
			reqact := actions[req]
			if reqact.failure != nil || reqact.skipped {
				act.skipped = true
			}
			inputs[req] = reqact.result
		}
		if act.skipped || depSkipped[a] || len(typeErrs) > 0 {
			act.skipped = true
			return
		}

		factFilter := make(map[reflect.Type]bool)
		for _, f := range a.FactTypes {
			factFilter[reflect.TypeOf(f)] = true
		}
		set := sets[a]
		pass := &analysis.Pass{
			Analyzer:     a,
			Fset:         fset,
			Files:        files,
			OtherFiles:   areq.NonGoFiles,
			IgnoredFiles: areq.IgnoredFiles,
			Pkg:          pkg,
			TypesInfo:    info,
			TypesSizes:   tc.Sizes,
			ResultOf:     inputs,
			Report: func(d analysis.Diagnostic) {
				mu.Lock()
				act.diagnostics = append(act.diagnostics, d)
				mu.Unlock()
			},
			// TODO(adonovan): make Export*Fact panic (and thus fail
			// the pass) if the fact's type is not among a.FactTypes,
			// or if called after Run returns, as x/tools' checker does
			// for the latter. Today an undeclared fact type is
			// encoded but cannot be decoded, causing the analysis of
			// importers to fail.
			ImportObjectFact:  set.ImportObjectFact,
			ExportObjectFact:  set.ExportObjectFact,
			AllObjectFacts:    func() []analysis.ObjectFact { return set.AllObjectFacts(factFilter) },
			ImportPackageFact: set.ImportPackageFact,
			ExportPackageFact: set.ExportPackageFact,
			AllPackageFacts:   func() []analysis.PackageFact { return set.AllPackageFacts(factFilter) },
			Module:            analysisModule(areq.Module),
		}
		pass.ReadFile = checkedReadFile(pass)
		var (
			result any
			err    error
		)
		prof.do(phaseRun, g.index[a], func() {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("panic: %v\n%s", r, trimStack(debug.Stack(), runPass))
				}
			}()
			result, err = runPass(a, pass)
		})
		if err != nil {
			act.failure = &protocol.Failure{Package: areq.ID, Analyzer: a.Name, Message: err.Error()}
		} else {
			act.result = result
		}
	}
	// Start with the leaves; when an action is done, start each
	// action that requires it whose prerequisites are all done.
	var wg sync.WaitGroup
	var enqueue func(act *action)
	enqueue = func(act *action) {
		wg.Go(func() {
			exec(act)
			for _, pred := range act.preds {
				if pred.pending.Add(-1) == 0 {
					enqueue(pred)
				}
			}
		})
	}
	// (Test the static condition, not pending, which the
	// enqueued actions may already be decrementing.)
	for _, act := range actions {
		if len(act.a.Requires) == 0 {
			enqueue(act)
		}
	}
	wg.Wait()

	// Write a pass file for each required analyzer.
	for _, a := range outputs {
		act := actions[a]
		pf := &protocol.Pass{
			Failure: act.failure,
			Skipped: act.skipped,
		}
		if pf.Failure == nil && !pf.Skipped && len(a.FactTypes) > 0 {
			var (
				data      []byte
				own, deps int
				err       error
			)
			prof.do(phaseFacts, g.index[a], func() {
				// Recover from panics encoding users' Facts.
				defer func() {
					if r := recover(); r != nil {
						err = fmt.Errorf("encoding facts: panic: %v\n%s", r, trimStack(debug.Stack(), encodeFacts))
					}
				}()
				data, own, deps = encodeFacts(sets[a])
			})
			prof.factsOut(g.index[a], own, deps, len(data))
			if err != nil {
				// A fact that cannot be encoded is the analyzer's fault.
				pf.Failure = &protocol.Failure{Package: areq.ID, Analyzer: a.Name, Message: err.Error()}
			} else {
				pf.Facts = data
			}
		}
		if report[a] {
			for _, d := range act.diagnostics {
				diag := &protocol.Diagnostic{
					Posn:     fset.Position(d.Pos).String(),
					Category: d.Category,
					Message:  d.Message,
					URL:      resolveURL(a, d),
				}
				if d.End.IsValid() {
					diag.End = fset.Position(d.End).String()
				}
				pf.Diagnostics = append(pf.Diagnostics, diag)
			}
		}
		if err := prof.doErr(phaseWrite, -1, func() error {
			return protocol.WritePass(areq.Outputs[g.index[a]], pf)
		}); err != nil {
			return err
		}
	}
	return nil
}

// trimStack trims the output of [debug.Stack], called during a panic
// recovered beneath the call of the recoverer function (runPass or
// encodeFacts), to the frames between the call of panic and the call of
// that function, which are those of the analyzer (or of its fact
// types). The recoverer is a function value, not a name, so that it
// cannot fail to exist.
func trimStack(stack []byte, recoverer any) string {
	name := runtime.FuncForPC(reflect.ValueOf(recoverer).Pointer()).Name()
	lines := strings.Split(strings.TrimSpace(string(stack)), "\n")
	start, end := 0, len(lines)
	for i, line := range lines {
		if strings.HasPrefix(line, "panic(") && start == 0 {
			start = i + 2 // skip the panic frame and its file:line
		} else if strings.HasPrefix(line, name+"(") {
			end = i
			break
		}
	}
	if start > end {
		start = 0
	}
	return strings.Join(lines[start:end], "\n")
}

// registerFactType registers the concrete type of a fact with gob,
// under a name that includes the full package path of its type, so
// that fact types of the same name in different packages do not
// conflict. (For a pointer type, which every fact type is,
// gob.Register would use only the package name: see the NOTE in its
// implementation.)
func registerFactType(f analysis.Fact) {
	t := reflect.TypeOf(f)
	name := t.String()
	if t.Kind() == reflect.Pointer && t.Elem().Name() != "" {
		name = "*" + t.Elem().PkgPath() + "." + t.Elem().Name()
	}
	registeredMu.Lock()
	defer registeredMu.Unlock()
	if !registered[t] {
		registered[t] = true
		gob.RegisterName(name, f)
	}
}

var (
	registeredMu sync.Mutex
	registered   = make(map[reflect.Type]bool)
)

func init() {
	// Keep analysisModule in sync with the Module types.
	if reflect.TypeFor[protocol.Module]().NumField() != reflect.TypeFor[analysis.Module]().NumField() {
		panic("protocol.Module and analysis.Module have different fields")
	}
}

// analysisModule converts a protocol.Module to an analysis.Module.
func analysisModule(m *protocol.Module) *analysis.Module {
	if m == nil {
		return nil
	}
	am := &analysis.Module{
		Path:      m.Path,
		Version:   m.Version,
		Replace:   analysisModule(m.Replace),
		Time:      m.Time,
		Main:      m.Main,
		Indirect:  m.Indirect,
		Dir:       m.Dir,
		GoMod:     m.GoMod,
		GoVersion: m.GoVersion,
	}
	// cmd/go supplies the go.mod version ("1.22.0"); the
	// documented form, as supplied by unitchecker, is "go1.22.0".
	if am.GoVersion != "" && !strings.HasPrefix(am.GoVersion, "go") {
		am.GoVersion = "go" + am.GoVersion
	}
	if m.Error != nil {
		am.Error = &analysis.ModuleError{Err: m.Error.Err}
	}
	return am
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

// checkedReadFile returns a Pass.ReadFile function that enforces the
// access policy documented at [analysis.Pass.ReadFile].
func checkedReadFile(pass *analysis.Pass) func(string) ([]byte, error) {
	return func(filename string) ([]byte, error) {
		ok := slices.Contains(pass.OtherFiles, filename) ||
			slices.Contains(pass.IgnoredFiles, filename) ||
			slices.ContainsFunc(pass.Files, func(f *ast.File) bool {
				return pass.Fset.File(f.FileStart).Name() == filename
			})
		if !ok {
			return nil, fmt.Errorf("Pass.ReadFile: %s is not among OtherFiles, IgnoredFiles, or names of Files", filename)
		}
		return os.ReadFile(filename)
	}
}

// resolveURL returns the URL of a diagnostic, resolved against the
// URL of its analyzer, as documented at [analysis.Diagnostic]. An empty
// URL with a category means "#category". (It follows unitchecker; an
// unparsable URL is returned unchanged.)
func resolveURL(a *analysis.Analyzer, d analysis.Diagnostic) string {
	if d.URL == "" && d.Category == "" && a.URL == "" {
		return ""
	}
	raw := d.URL
	if raw == "" && d.Category != "" {
		raw = "#" + d.Category
	}
	u, err := url.Parse(raw)
	if err != nil {
		return d.URL
	}
	base, err := url.Parse(a.URL)
	if err != nil {
		return d.URL
	}
	return base.ResolveReference(u).String()
}

// runPass runs analyzer a on the pass. It is the frame at which
// [trimStack] trims the stacks of panics in analyzers.
//
//go:noinline
func runPass(a *analysis.Analyzer, pass *analysis.Pass) (any, error) {
	return a.Run(pass)
}

// encodeFacts encodes a set of facts. It is the frame at which
// [trimStack] trims the stacks of panics in fact types.
//
//go:noinline
func encodeFacts(set *factSet) (data []byte, own, deps int) {
	return set.Encode()
}
