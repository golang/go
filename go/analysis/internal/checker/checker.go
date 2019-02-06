// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package checker defines the implementation of the checker commands.
// The same code drives the multi-analysis driver, the single-analysis
// driver that is conventionally provided for convenience along with
// each analysis package, and the test driver.
package checker

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"go/token"
	"go/types"
	"log"
	"os"
	"reflect"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/go/packages"
)

var (
	// Debug is a set of single-letter flags:
	//
	//	f	show [f]acts as they are created
	// 	p	disable [p]arallel execution of analyzers
	//	s	do additional [s]anity checks on fact types and serialization
	//	t	show [t]iming info (NB: use 'p' flag to avoid GC/scheduler noise)
	//	v	show [v]erbose logging
	//
	Debug = ""

	// Log files for optional performance tracing.
	CPUProfile, MemProfile, Trace string
)

// RegisterFlags registers command-line flags used by the analysis driver.
func RegisterFlags() {
	// When adding flags here, remember to update
	// the list of suppressed flags in analysisflags.

	flag.StringVar(&Debug, "debug", Debug, `debug flags, any subset of "fpstv"`)

	flag.StringVar(&CPUProfile, "cpuprofile", "", "write CPU profile to this file")
	flag.StringVar(&MemProfile, "memprofile", "", "write memory profile to this file")
	flag.StringVar(&Trace, "trace", "", "write trace log to this file")
}

// Run loads the packages specified by args using go/packages,
// then applies the specified analyzers to them.
// Analysis flags must already have been set.
// It provides most of the logic for the main functions of both the
// singlechecker and the multi-analysis commands.
// It returns the appropriate exit code.
func Run(args []string, analyzers []*analysis.Analyzer) (exitcode int) {
	if CPUProfile != "" {
		f, err := os.Create(CPUProfile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		// NB: profile won't be written in case of error.
		defer pprof.StopCPUProfile()
	}

	if Trace != "" {
		f, err := os.Create(Trace)
		if err != nil {
			log.Fatal(err)
		}
		if err := trace.Start(f); err != nil {
			log.Fatal(err)
		}
		// NB: trace log won't be written in case of error.
		defer func() {
			trace.Stop()
			log.Printf("To view the trace, run:\n$ go tool trace view %s", Trace)
		}()
	}

	if MemProfile != "" {
		f, err := os.Create(MemProfile)
		if err != nil {
			log.Fatal(err)
		}
		// NB: memprofile won't be written in case of error.
		defer func() {
			runtime.GC() // get up-to-date statistics
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Fatalf("Writing memory profile: %v", err)
			}
			f.Close()
		}()
	}

	// Load the packages.
	if dbg('v') {
		log.SetPrefix("")
		log.SetFlags(log.Lmicroseconds) // display timing
		log.Printf("load %s", args)
	}

	// Optimization: if the selected analyzers don't produce/consume
	// facts, we need source only for the initial packages.
	allSyntax := needFacts(analyzers)
	initial, err := load(args, allSyntax)
	if err != nil {
		log.Print(err)
		return 1 // load errors
	}

	// Print the results.
	roots := analyze(initial, analyzers)

	return printDiagnostics(roots)
}

// load loads the initial packages.
func load(patterns []string, allSyntax bool) ([]*packages.Package, error) {
	mode := packages.LoadSyntax
	if allSyntax {
		mode = packages.LoadAllSyntax
	}
	conf := packages.Config{
		Mode:  mode,
		Tests: true,
	}
	initial, err := packages.Load(&conf, patterns...)
	if err == nil {
		if n := packages.PrintErrors(initial); n > 1 {
			err = fmt.Errorf("%d errors during loading", n)
		} else if n == 1 {
			err = fmt.Errorf("error during loading")
		} else if len(initial) == 0 {
			err = fmt.Errorf("%s matched no packages", strings.Join(patterns, " "))
		}
	}

	return initial, err
}

// TestAnalyzer applies an analysis to a set of packages (and their
// dependencies if necessary) and returns the results.
//
// Facts about pkg are returned in a map keyed by object; package facts
// have a nil key.
//
// This entry point is used only by analysistest.
func TestAnalyzer(a *analysis.Analyzer, pkgs []*packages.Package) []*TestAnalyzerResult {
	var results []*TestAnalyzerResult
	for _, act := range analyze(pkgs, []*analysis.Analyzer{a}) {
		facts := make(map[types.Object][]analysis.Fact)
		for key, fact := range act.objectFacts {
			if key.obj.Pkg() == act.pass.Pkg {
				facts[key.obj] = append(facts[key.obj], fact)
			}
		}
		for key, fact := range act.packageFacts {
			if key.pkg == act.pass.Pkg {
				facts[nil] = append(facts[nil], fact)
			}
		}

		results = append(results, &TestAnalyzerResult{act.pass, act.diagnostics, facts, act.result, act.err})
	}
	return results
}

type TestAnalyzerResult struct {
	Pass        *analysis.Pass
	Diagnostics []analysis.Diagnostic
	Facts       map[types.Object][]analysis.Fact
	Result      interface{}
	Err         error
}

func analyze(pkgs []*packages.Package, analyzers []*analysis.Analyzer) []*action {
	// Construct the action graph.
	if dbg('v') {
		log.Printf("building graph of analysis passes")
	}

	// Each graph node (action) is one unit of analysis.
	// Edges express package-to-package (vertical) dependencies,
	// and analysis-to-analysis (horizontal) dependencies.
	type key struct {
		*analysis.Analyzer
		*packages.Package
	}
	actions := make(map[key]*action)

	var mkAction func(a *analysis.Analyzer, pkg *packages.Package) *action
	mkAction = func(a *analysis.Analyzer, pkg *packages.Package) *action {
		k := key{a, pkg}
		act, ok := actions[k]
		if !ok {
			act = &action{a: a, pkg: pkg}

			// Add a dependency on each required analyzers.
			for _, req := range a.Requires {
				act.deps = append(act.deps, mkAction(req, pkg))
			}

			// An analysis that consumes/produces facts
			// must run on the package's dependencies too.
			if len(a.FactTypes) > 0 {
				paths := make([]string, 0, len(pkg.Imports))
				for path := range pkg.Imports {
					paths = append(paths, path)
				}
				sort.Strings(paths) // for determinism
				for _, path := range paths {
					dep := mkAction(a, pkg.Imports[path])
					act.deps = append(act.deps, dep)
				}
			}

			actions[k] = act
		}
		return act
	}

	// Build nodes for initial packages.
	var roots []*action
	for _, a := range analyzers {
		for _, pkg := range pkgs {
			root := mkAction(a, pkg)
			root.isroot = true
			roots = append(roots, root)
		}
	}

	// Execute the graph in parallel.
	execAll(roots)

	return roots
}

// printDiagnostics prints the diagnostics for the root packages in either
// plain text or JSON format. JSON format also includes errors for any
// dependencies.
//
// It returns the exitcode: in plain mode, 0 for success, 1 for analysis
// errors, and 3 for diagnostics. We avoid 2 since the flag package uses
// it. JSON mode always succeeds at printing errors and diagnostics in a
// structured form to stdout.
func printDiagnostics(roots []*action) (exitcode int) {
	// Print the output.
	//
	// Print diagnostics only for root packages,
	// but errors for all packages.
	printed := make(map[*action]bool)
	var print func(*action)
	var visitAll func(actions []*action)
	visitAll = func(actions []*action) {
		for _, act := range actions {
			if !printed[act] {
				printed[act] = true
				visitAll(act.deps)
				print(act)
			}
		}
	}

	if analysisflags.JSON {
		// JSON output
		tree := make(analysisflags.JSONTree)
		print = func(act *action) {
			var diags []analysis.Diagnostic
			if act.isroot {
				diags = act.diagnostics
			}
			tree.Add(act.pkg.Fset, act.pkg.ID, act.a.Name, diags, act.err)
		}
		visitAll(roots)
		tree.Print()
	} else {
		// plain text output

		// De-duplicate diagnostics by position (not token.Pos) to
		// avoid double-reporting in source files that belong to
		// multiple packages, such as foo and foo.test.
		type key struct {
			token.Position
			*analysis.Analyzer
			message string
		}
		seen := make(map[key]bool)

		print = func(act *action) {
			if act.err != nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", act.a.Name, act.err)
				exitcode = 1 // analysis failed, at least partially
				return
			}
			if act.isroot {
				for _, diag := range act.diagnostics {
					// We don't display a.Name/f.Category
					// as most users don't care.

					posn := act.pkg.Fset.Position(diag.Pos)
					k := key{posn, act.a, diag.Message}
					if seen[k] {
						continue // duplicate
					}
					seen[k] = true

					analysisflags.PrintPlain(act.pkg.Fset, diag)
				}
			}
		}
		visitAll(roots)

		if exitcode == 0 && len(seen) > 0 {
			exitcode = 3 // successfuly produced diagnostics
		}
	}

	// Print timing info.
	if dbg('t') {
		if !dbg('p') {
			log.Println("Warning: times are mostly GC/scheduler noise; use -debug=tp to disable parallelism")
		}
		var all []*action
		var total time.Duration
		for act := range printed {
			all = append(all, act)
			total += act.duration
		}
		sort.Slice(all, func(i, j int) bool {
			return all[i].duration > all[j].duration
		})

		// Print actions accounting for 90% of the total.
		var sum time.Duration
		for _, act := range all {
			fmt.Fprintf(os.Stderr, "%s\t%s\n", act.duration, act)
			sum += act.duration
			if sum >= total*9/10 {
				break
			}
		}
	}

	return exitcode
}

// needFacts reports whether any analysis required by the specified set
// needs facts.  If so, we must load the entire program from source.
func needFacts(analyzers []*analysis.Analyzer) bool {
	seen := make(map[*analysis.Analyzer]bool)
	var q []*analysis.Analyzer // for BFS
	q = append(q, analyzers...)
	for len(q) > 0 {
		a := q[0]
		q = q[1:]
		if !seen[a] {
			seen[a] = true
			if len(a.FactTypes) > 0 {
				return true
			}
			q = append(q, a.Requires...)
		}
	}
	return false
}

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type action struct {
	once         sync.Once
	a            *analysis.Analyzer
	pkg          *packages.Package
	pass         *analysis.Pass
	isroot       bool
	deps         []*action
	objectFacts  map[objectFactKey]analysis.Fact
	packageFacts map[packageFactKey]analysis.Fact
	inputs       map[*analysis.Analyzer]interface{}
	result       interface{}
	diagnostics  []analysis.Diagnostic
	err          error
	duration     time.Duration
}

type objectFactKey struct {
	obj types.Object
	typ reflect.Type
}

type packageFactKey struct {
	pkg *types.Package
	typ reflect.Type
}

func (act *action) String() string {
	return fmt.Sprintf("%s@%s", act.a, act.pkg)
}

func execAll(actions []*action) {
	sequential := dbg('p')
	var wg sync.WaitGroup
	for _, act := range actions {
		wg.Add(1)
		work := func(act *action) {
			act.exec()
			wg.Done()
		}
		if sequential {
			work(act)
		} else {
			go work(act)
		}
	}
	wg.Wait()
}

func (act *action) exec() { act.once.Do(act.execOnce) }

func (act *action) execOnce() {
	// Analyze dependencies.
	execAll(act.deps)

	// TODO(adonovan): uncomment this during profiling.
	// It won't build pre-go1.11 but conditional compilation
	// using build tags isn't warranted.
	//
	// ctx, task := trace.NewTask(context.Background(), "exec")
	// trace.Log(ctx, "pass", act.String())
	// defer task.End()

	// Record time spent in this node but not its dependencies.
	// In parallel mode, due to GC/scheduler contention, the
	// time is 5x higher than in sequential mode, even with a
	// semaphore limiting the number of threads here.
	// So use -debug=tp.
	if dbg('t') {
		t0 := time.Now()
		defer func() { act.duration = time.Since(t0) }()
	}

	// Report an error if any dependency failed.
	var failed []string
	for _, dep := range act.deps {
		if dep.err != nil {
			failed = append(failed, dep.String())
		}
	}
	if failed != nil {
		sort.Strings(failed)
		act.err = fmt.Errorf("failed prerequisites: %s", strings.Join(failed, ", "))
		return
	}

	// Plumb the output values of the dependencies
	// into the inputs of this action.  Also facts.
	inputs := make(map[*analysis.Analyzer]interface{})
	act.objectFacts = make(map[objectFactKey]analysis.Fact)
	act.packageFacts = make(map[packageFactKey]analysis.Fact)
	for _, dep := range act.deps {
		if dep.pkg == act.pkg {
			// Same package, different analysis (horizontal edge):
			// in-memory outputs of prerequisite analyzers
			// become inputs to this analysis pass.
			inputs[dep.a] = dep.result

		} else if dep.a == act.a { // (always true)
			// Same analysis, different package (vertical edge):
			// serialized facts produced by prerequisite analysis
			// become available to this analysis pass.
			inheritFacts(act, dep)
		}
	}

	// Run the analysis.
	pass := &analysis.Pass{
		Analyzer:          act.a,
		Fset:              act.pkg.Fset,
		Files:             act.pkg.Syntax,
		OtherFiles:        act.pkg.OtherFiles,
		Pkg:               act.pkg.Types,
		TypesInfo:         act.pkg.TypesInfo,
		TypesSizes:        act.pkg.TypesSizes,
		ResultOf:          inputs,
		Report:            func(d analysis.Diagnostic) { act.diagnostics = append(act.diagnostics, d) },
		ImportObjectFact:  act.importObjectFact,
		ExportObjectFact:  act.exportObjectFact,
		ImportPackageFact: act.importPackageFact,
		ExportPackageFact: act.exportPackageFact,
	}
	act.pass = pass

	var err error
	if act.pkg.IllTyped && !pass.Analyzer.RunDespiteErrors {
		err = fmt.Errorf("analysis skipped due to errors in package")
	} else {
		act.result, err = pass.Analyzer.Run(pass)
		if err == nil {
			if got, want := reflect.TypeOf(act.result), pass.Analyzer.ResultType; got != want {
				err = fmt.Errorf(
					"internal error: on package %s, analyzer %s returned a result of type %v, but declared ResultType %v",
					pass.Pkg.Path(), pass.Analyzer, got, want)
			}
		}
	}
	act.err = err

	// disallow calls after Run
	pass.ExportObjectFact = nil
	pass.ExportPackageFact = nil
}

// inheritFacts populates act.facts with
// those it obtains from its dependency, dep.
func inheritFacts(act, dep *action) {
	serialize := dbg('s')

	for key, fact := range dep.objectFacts {
		// Filter out facts related to objects
		// that are irrelevant downstream
		// (equivalently: not in the compiler export data).
		if !exportedFrom(key.obj, dep.pkg.Types) {
			if false {
				log.Printf("%v: discarding %T fact from %s for %s: %s", act, fact, dep, key.obj, fact)
			}
			continue
		}

		// Optionally serialize/deserialize fact
		// to verify that it works across address spaces.
		if serialize {
			var err error
			fact, err = codeFact(fact)
			if err != nil {
				log.Panicf("internal error: encoding of %T fact failed in %v", fact, act)
			}
		}

		if false {
			log.Printf("%v: inherited %T fact for %s: %s", act, fact, key.obj, fact)
		}
		act.objectFacts[key] = fact
	}

	for key, fact := range dep.packageFacts {
		// TODO: filter out facts that belong to
		// packages not mentioned in the export data
		// to prevent side channels.

		// Optionally serialize/deserialize fact
		// to verify that it works across address spaces
		// and is deterministic.
		if serialize {
			var err error
			fact, err = codeFact(fact)
			if err != nil {
				log.Panicf("internal error: encoding of %T fact failed in %v", fact, act)
			}
		}

		if false {
			log.Printf("%v: inherited %T fact for %s: %s", act, fact, key.pkg.Path(), fact)
		}
		act.packageFacts[key] = fact
	}
}

// codeFact encodes then decodes a fact,
// just to exercise that logic.
func codeFact(fact analysis.Fact) (analysis.Fact, error) {
	// We encode facts one at a time.
	// A real modular driver would emit all facts
	// into one encoder to improve gob efficiency.
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(fact); err != nil {
		return nil, err
	}

	// Encode it twice and assert that we get the same bits.
	// This helps detect nondeterministic Gob encoding (e.g. of maps).
	var buf2 bytes.Buffer
	if err := gob.NewEncoder(&buf2).Encode(fact); err != nil {
		return nil, err
	}
	if !bytes.Equal(buf.Bytes(), buf2.Bytes()) {
		return nil, fmt.Errorf("encoding of %T fact is nondeterministic", fact)
	}

	new := reflect.New(reflect.TypeOf(fact).Elem()).Interface().(analysis.Fact)
	if err := gob.NewDecoder(&buf).Decode(new); err != nil {
		return nil, err
	}
	return new, nil
}

// exportedFrom reports whether obj may be visible to a package that imports pkg.
// This includes not just the exported members of pkg, but also unexported
// constants, types, fields, and methods, perhaps belonging to oether packages,
// that find there way into the API.
// This is an overapproximation of the more accurate approach used by
// gc export data, which walks the type graph, but it's much simpler.
//
// TODO(adonovan): do more accurate filtering by walking the type graph.
func exportedFrom(obj types.Object, pkg *types.Package) bool {
	switch obj := obj.(type) {
	case *types.Func:
		return obj.Exported() && obj.Pkg() == pkg ||
			obj.Type().(*types.Signature).Recv() != nil
	case *types.Var:
		return obj.Exported() && obj.Pkg() == pkg ||
			obj.IsField()
	case *types.TypeName, *types.Const:
		return true
	}
	return false // Nil, Builtin, Label, or PkgName
}

// importObjectFact implements Pass.ImportObjectFact.
// Given a non-nil pointer ptr of type *T, where *T satisfies Fact,
// importObjectFact copies the fact value to *ptr.
func (act *action) importObjectFact(obj types.Object, ptr analysis.Fact) bool {
	if obj == nil {
		panic("nil object")
	}
	key := objectFactKey{obj, factType(ptr)}
	if v, ok := act.objectFacts[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// exportObjectFact implements Pass.ExportObjectFact.
func (act *action) exportObjectFact(obj types.Object, fact analysis.Fact) {
	if act.pass.ExportObjectFact == nil {
		log.Panicf("%s: Pass.ExportObjectFact(%s, %T) called after Run", act, obj, fact)
	}

	if obj.Pkg() != act.pkg.Types {
		log.Panicf("internal error: in analysis %s of package %s: Fact.Set(%s, %T): can't set facts on objects belonging another package",
			act.a, act.pkg, obj, fact)
	}

	key := objectFactKey{obj, factType(fact)}
	act.objectFacts[key] = fact // clobber any existing entry
	if dbg('f') {
		objstr := types.ObjectString(obj, (*types.Package).Name)
		fmt.Fprintf(os.Stderr, "%s: object %s has fact %s\n",
			act.pkg.Fset.Position(obj.Pos()), objstr, fact)
	}
}

// importPackageFact implements Pass.ImportPackageFact.
// Given a non-nil pointer ptr of type *T, where *T satisfies Fact,
// fact copies the fact value to *ptr.
func (act *action) importPackageFact(pkg *types.Package, ptr analysis.Fact) bool {
	if pkg == nil {
		panic("nil package")
	}
	key := packageFactKey{pkg, factType(ptr)}
	if v, ok := act.packageFacts[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// exportPackageFact implements Pass.ExportPackageFact.
func (act *action) exportPackageFact(fact analysis.Fact) {
	if act.pass.ExportPackageFact == nil {
		log.Panicf("%s: Pass.ExportPackageFact(%T) called after Run", act, fact)
	}

	key := packageFactKey{act.pass.Pkg, factType(fact)}
	act.packageFacts[key] = fact // clobber any existing entry
	if dbg('f') {
		fmt.Fprintf(os.Stderr, "%s: package %s has fact %s\n",
			act.pkg.Fset.Position(act.pass.Files[0].Pos()), act.pass.Pkg.Path(), fact)
	}
}

func factType(fact analysis.Fact) reflect.Type {
	t := reflect.TypeOf(fact)
	if t.Kind() != reflect.Ptr {
		log.Fatalf("invalid Fact type: got %T, want pointer", t)
	}
	return t
}

func dbg(b byte) bool { return strings.IndexByte(Debug, b) >= 0 }
