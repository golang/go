// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"
	"reflect"
	"runtime/debug"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/memoize"
)

func (s *snapshot) Analyze(ctx context.Context, id string, analyzers []*source.Analyzer) ([]*source.Diagnostic, error) {
	// TODO(adonovan): merge these two loops. There's no need to
	// construct all the root action handles before beginning
	// analysis. Operations should be concurrent (though that first
	// requires buildPackageHandle not to be inefficient when
	// called in parallel.)
	var roots []*actionHandle
	for _, a := range analyzers {
		if !a.IsEnabled(s.view) {
			continue
		}
		ah, err := s.actionHandle(ctx, PackageID(id), a.Analyzer)
		if err != nil {
			return nil, err
		}
		roots = append(roots, ah)
	}

	// Run and wait for all analyzers, and report diagnostics
	// only from those that succeed. Ignore the others.
	var results []*source.Diagnostic
	for _, ah := range roots {
		v, err := s.awaitPromise(ctx, ah.promise)
		if err != nil {
			return nil, err // wait was cancelled
		}

		res := v.(actionResult)
		if res.err != nil {
			continue // analysis failed; ignore it.
		}

		results = append(results, res.data.diagnostics...)
	}
	return results, nil
}

type actionKey struct {
	pkgid    PackageID
	analyzer *analysis.Analyzer
}

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type actionHandle struct {
	key     actionKey        // just for String()
	promise *memoize.Promise // [actionResult]
}

// actionData is the successful result of analyzing a package.
type actionData struct {
	analyzer     *analysis.Analyzer
	pkgTypes     *types.Package // types only; don't keep syntax live
	diagnostics  []*source.Diagnostic
	result       interface{}
	objectFacts  map[objectFactKey]analysis.Fact
	packageFacts map[packageFactKey]analysis.Fact
}

// actionResult holds the result of a call to actionImpl.
type actionResult struct {
	data *actionData
	err  error
}

type objectFactKey struct {
	obj types.Object
	typ reflect.Type
}

type packageFactKey struct {
	pkg *types.Package
	typ reflect.Type
}

func (s *snapshot) actionHandle(ctx context.Context, id PackageID, a *analysis.Analyzer) (*actionHandle, error) {
	key := actionKey{id, a}

	s.mu.Lock()
	entry, hit := s.actions.Get(key)
	s.mu.Unlock()

	if hit {
		return entry.(*actionHandle), nil
	}

	// TODO(adonovan): opt: this block of code sequentially loads a package
	// (and all its dependencies), then sequentially creates action handles
	// for the direct dependencies (whose packages have by then been loaded
	// as a consequence of ph.check) which does a sequential recursion
	// down the action graph. Only once all that work is complete do we
	// put a handle in the cache. As with buildPackageHandle, this does
	// not exploit the natural parallelism in the problem, and the naive
	// use of concurrency would lead to an exponential amount of duplicated
	// work. We should instead use an atomically updated future cache
	// and a parallel graph traversal.
	ph, err := s.buildPackageHandle(ctx, id, source.ParseFull)
	if err != nil {
		return nil, err
	}
	pkg, err := ph.await(ctx, s)
	if err != nil {
		return nil, err
	}

	// Add a dependency on each required analyzer.
	var deps []*actionHandle // unordered
	for _, req := range a.Requires {
		// TODO(adonovan): opt: there's no need to repeat the package-handle
		// portion of the recursion here, since we have the pkg already.
		reqActionHandle, err := s.actionHandle(ctx, id, req)
		if err != nil {
			return nil, err
		}
		deps = append(deps, reqActionHandle)
	}

	// TODO(golang/go#35089): Re-enable this when we doesn't use ParseExported
	// mode for dependencies. In the meantime, disable analysis for dependencies,
	// since we don't get anything useful out of it.
	if false {
		// An analysis that consumes/produces facts
		// must run on the package's dependencies too.
		if len(a.FactTypes) > 0 {
			for _, importID := range ph.m.Imports {
				depActionHandle, err := s.actionHandle(ctx, importID, a)
				if err != nil {
					return nil, err
				}
				deps = append(deps, depActionHandle)
			}
		}
	}

	// The promises are kept in a store on the package,
	// so the key need only include the analyzer name.
	//
	// (Since type-checking and analysis depend on the identity
	// of packages--distinct packages produced by the same
	// recipe are not fungible--we must in effect use the package
	// itself as part of the key. Rather than actually use a pointer
	// in the key, we get a simpler object graph if we shard the
	// store by packages.)
	promise, release := pkg.analyses.Promise(a.Name, func(ctx context.Context, arg interface{}) interface{} {
		res, err := actionImpl(ctx, arg.(*snapshot), deps, a, pkg)
		return actionResult{res, err}
	})

	ah := &actionHandle{
		key:     key,
		promise: promise,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Check cache again in case another thread got there first.
	if result, ok := s.actions.Get(key); ok {
		release()
		return result.(*actionHandle), nil
	}

	s.actions.Set(key, ah, func(_, _ interface{}) { release() })

	return ah, nil
}

func (key actionKey) String() string {
	return fmt.Sprintf("%s@%s", key.analyzer, key.pkgid)
}

func (act *actionHandle) String() string {
	return act.key.String()
}

// actionImpl runs the analysis for action node (analyzer, pkg),
// whose direct dependencies are deps.
func actionImpl(ctx context.Context, snapshot *snapshot, deps []*actionHandle, analyzer *analysis.Analyzer, pkg *pkg) (*actionData, error) {
	// Run action dependencies first, and plumb the results and
	// facts of each dependency into the inputs of this action.
	var (
		mu           sync.Mutex
		inputs       = make(map[*analysis.Analyzer]interface{})
		objectFacts  = make(map[objectFactKey]analysis.Fact)
		packageFacts = make(map[packageFactKey]analysis.Fact)
	)
	g, ctx := errgroup.WithContext(ctx)
	for _, dep := range deps {
		dep := dep
		g.Go(func() error {
			v, err := snapshot.awaitPromise(ctx, dep.promise)
			if err != nil {
				return err // e.g. cancelled
			}
			res := v.(actionResult)
			if res.err != nil {
				return res.err // analysis of dependency failed
			}
			data := res.data

			mu.Lock()
			defer mu.Unlock()
			if data.pkgTypes == pkg.types {
				// Same package, different analysis (horizontal edge):
				// in-memory outputs of prerequisite analyzers
				// become inputs to this analysis pass.
				inputs[data.analyzer] = data.result

			} else if data.analyzer == analyzer {
				// Same analysis, different package (vertical edge):
				// serialized facts produced by prerequisite analysis
				// become available to this analysis pass.
				for key, fact := range data.objectFacts {
					objectFacts[key] = fact
				}
				for key, fact := range data.packageFacts {
					packageFacts[key] = fact
				}

			} else {
				// Edge is neither vertical nor horizontal.
				// This should never happen, yet an assertion here was
				// observed to fail due to an edge (bools, p) -> (inspector, p')
				// where p and p' are distinct packages with the
				// same ID ("command-line-arguments:file=.../main.go").
				//
				// It is not yet clear whether the command-line-arguments
				// package is significant, but it is clear that package
				// loading (the mapping from ID to *pkg) is inconsistent
				// within a single graph.

				// Use the bug package so that we detect whether our tests
				// discover this problem in regular packages.
				// For command-line-arguments we quietly abort the analysis
				// for now since we already know there is a bug.
				errorf := bug.Errorf // report this discovery
				if source.IsCommandLineArguments(pkg.ID()) {
					errorf = fmt.Errorf // suppress reporting
				}
				err := errorf("internal error: unexpected analysis dependency %s@%s -> %s", analyzer.Name, pkg.ID(), dep)
				// Log the event in any case, as the ultimate
				// consumer of actionResult ignores errors.
				event.Error(ctx, "analysis", err)
				return err
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err // cancelled, or dependency failed
	}

	// Now run the (pkg, analyzer) analysis.
	var syntax []*ast.File
	for _, cgf := range pkg.compiledGoFiles {
		syntax = append(syntax, cgf.File)
	}
	var rawDiagnostics []analysis.Diagnostic
	pass := &analysis.Pass{
		Analyzer:   analyzer,
		Fset:       snapshot.FileSet(),
		Files:      syntax,
		Pkg:        pkg.GetTypes(),
		TypesInfo:  pkg.GetTypesInfo(),
		TypesSizes: pkg.GetTypesSizes(),
		ResultOf:   inputs,
		Report: func(d analysis.Diagnostic) {
			// Prefix the diagnostic category with the analyzer's name.
			if d.Category == "" {
				d.Category = analyzer.Name
			} else {
				d.Category = analyzer.Name + "." + d.Category
			}
			rawDiagnostics = append(rawDiagnostics, d)
		},
		ImportObjectFact: func(obj types.Object, ptr analysis.Fact) bool {
			if obj == nil {
				panic("nil object")
			}
			key := objectFactKey{obj, factType(ptr)}

			if v, ok := objectFacts[key]; ok {
				reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
				return true
			}
			return false
		},
		ExportObjectFact: func(obj types.Object, fact analysis.Fact) {
			if obj.Pkg() != pkg.types {
				panic(fmt.Sprintf("internal error: in analysis %s of package %s: Fact.Set(%s, %T): can't set facts on objects belonging another package",
					analyzer, pkg.ID(), obj, fact))
			}
			key := objectFactKey{obj, factType(fact)}
			objectFacts[key] = fact // clobber any existing entry
		},
		ImportPackageFact: func(pkg *types.Package, ptr analysis.Fact) bool {
			if pkg == nil {
				panic("nil package")
			}
			key := packageFactKey{pkg, factType(ptr)}
			if v, ok := packageFacts[key]; ok {
				reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
				return true
			}
			return false
		},
		ExportPackageFact: func(fact analysis.Fact) {
			key := packageFactKey{pkg.types, factType(fact)}
			packageFacts[key] = fact // clobber any existing entry
		},
		AllObjectFacts: func() []analysis.ObjectFact {
			facts := make([]analysis.ObjectFact, 0, len(objectFacts))
			for k := range objectFacts {
				facts = append(facts, analysis.ObjectFact{Object: k.obj, Fact: objectFacts[k]})
			}
			return facts
		},
		AllPackageFacts: func() []analysis.PackageFact {
			facts := make([]analysis.PackageFact, 0, len(packageFacts))
			for k := range packageFacts {
				facts = append(facts, analysis.PackageFact{Package: k.pkg, Fact: packageFacts[k]})
			}
			return facts
		},
	}
	analysisinternal.SetTypeErrors(pass, pkg.typeErrors)

	if (pkg.HasListOrParseErrors() || pkg.HasTypeErrors()) && !analyzer.RunDespiteErrors {
		return nil, fmt.Errorf("skipping analysis %s because package %s contains errors", analyzer.Name, pkg.ID())
	}

	// Recover from panics (only) within the analyzer logic.
	// (Use an anonymous function to limit the recover scope.)
	var result interface{}
	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				// An Analyzer crashed. This is often merely a symptom
				// of a problem in package loading.
				//
				// We believe that CL 420538 may have fixed these crashes, so enable
				// strict checks in tests.
				const strict = true
				if strict && bug.PanicOnBugs && analyzer.Name != "fact_purity" {
					// During testing, crash. See issues 54762, 56035.
					// But ignore analyzers with known crash bugs:
					// - fact_purity (dominikh/go-tools#1327)
					debug.SetTraceback("all") // show all goroutines
					panic(r)
				} else {
					// In production, suppress the panic and press on.
					err = fmt.Errorf("analysis %s for package %s panicked: %v", analyzer.Name, pkg.PkgPath(), r)
				}
			}
		}()
		result, err = pass.Analyzer.Run(pass)
	}()
	if err != nil {
		return nil, err
	}

	if got, want := reflect.TypeOf(result), pass.Analyzer.ResultType; got != want {
		return nil, fmt.Errorf(
			"internal error: on package %s, analyzer %s returned a result of type %v, but declared ResultType %v",
			pass.Pkg.Path(), pass.Analyzer, got, want)
	}

	// disallow calls after Run
	pass.ExportObjectFact = func(obj types.Object, fact analysis.Fact) {
		panic(fmt.Sprintf("%s:%s: Pass.ExportObjectFact(%s, %T) called after Run", analyzer.Name, pkg.PkgPath(), obj, fact))
	}
	pass.ExportPackageFact = func(fact analysis.Fact) {
		panic(fmt.Sprintf("%s:%s: Pass.ExportPackageFact(%T) called after Run", analyzer.Name, pkg.PkgPath(), fact))
	}

	// Filter out facts related to objects that are irrelevant downstream
	// (equivalently: not in the compiler export data).
	for key := range objectFacts {
		if !exportedFrom(key.obj, pkg.types) {
			delete(objectFacts, key)
		}
	}
	// TODO: filter out facts that belong to packages not
	// mentioned in the export data to prevent side channels.

	var diagnostics []*source.Diagnostic
	for _, diag := range rawDiagnostics {
		srcDiags, err := analysisDiagnosticDiagnostics(snapshot, pkg, analyzer, &diag)
		if err != nil {
			event.Error(ctx, "unable to compute analysis error position", err, tag.Category.Of(diag.Category), tag.Package.Of(pkg.ID()))
			continue
		}
		diagnostics = append(diagnostics, srcDiags...)
	}
	return &actionData{
		analyzer:     analyzer,
		pkgTypes:     pkg.types,
		diagnostics:  diagnostics,
		result:       result,
		objectFacts:  objectFacts,
		packageFacts: packageFacts,
	}, nil
}

// exportedFrom reports whether obj may be visible to a package that imports pkg.
// This includes not just the exported members of pkg, but also unexported
// constants, types, fields, and methods, perhaps belonging to other packages,
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
	case *types.TypeName:
		return true
	case *types.Const:
		return obj.Exported() && obj.Pkg() == pkg
	}
	return false // Nil, Builtin, Label, or PkgName
}

func factType(fact analysis.Fact) reflect.Type {
	t := reflect.TypeOf(fact)
	if t.Kind() != reflect.Ptr {
		panic(fmt.Sprintf("invalid Fact type: got %T, want pointer", fact))
	}
	return t
}

func (s *snapshot) DiagnosePackage(ctx context.Context, spkg source.Package) (map[span.URI][]*source.Diagnostic, error) {
	pkg := spkg.(*pkg)
	var errorAnalyzerDiag []*source.Diagnostic
	if pkg.HasTypeErrors() {
		// Apply type error analyzers.
		// They augment type error diagnostics with their own fixes.
		var analyzers []*source.Analyzer
		for _, a := range s.View().Options().TypeErrorAnalyzers {
			analyzers = append(analyzers, a)
		}
		var err error
		errorAnalyzerDiag, err = s.Analyze(ctx, pkg.ID(), analyzers)
		if err != nil {
			// Keep going: analysis failures should not block diagnostics.
			event.Error(ctx, "type error analysis failed", err, tag.Package.Of(pkg.ID()))
		}
	}
	diags := map[span.URI][]*source.Diagnostic{}
	for _, diag := range pkg.diagnostics {
		for _, eaDiag := range errorAnalyzerDiag {
			if eaDiag.URI == diag.URI && eaDiag.Range == diag.Range && eaDiag.Message == diag.Message {
				// Type error analyzers just add fixes and tags. Make a copy,
				// since we don't own either, and overwrite.
				// The analyzer itself can't do this merge because
				// analysis.Diagnostic doesn't have all the fields, and Analyze
				// can't because it doesn't have the type error, notably its code.
				clone := *diag
				clone.SuggestedFixes = eaDiag.SuggestedFixes
				clone.Tags = eaDiag.Tags
				clone.Analyzer = eaDiag.Analyzer
				diag = &clone
			}
		}
		diags[diag.URI] = append(diags[diag.URI], diag)
	}
	return diags, nil
}
