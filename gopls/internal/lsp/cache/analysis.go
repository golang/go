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
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
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
	pkg      packageKey
	analyzer *analysis.Analyzer
}

type actionHandleKey source.Hash

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type actionHandle struct {
	promise *memoize.Promise // [actionResult]

	analyzer *analysis.Analyzer
	pkg      *pkg
}

// actionData is the successful result of analyzing a package.
type actionData struct {
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
	const mode = source.ParseFull
	key := actionKey{
		pkg:      packageKey{id: id, mode: mode},
		analyzer: a,
	}

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
	ph, err := s.buildPackageHandle(ctx, id, mode)
	if err != nil {
		return nil, err
	}
	pkg, err := ph.await(ctx, s)
	if err != nil {
		return nil, err
	}

	// Add a dependency on each required analyzer.
	var deps []*actionHandle
	for _, req := range a.Requires {
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
			for _, importID := range ph.m.Deps {
				depActionHandle, err := s.actionHandle(ctx, importID, a)
				if err != nil {
					return nil, err
				}
				deps = append(deps, depActionHandle)
			}
		}
	}

	promise, release := s.store.Promise(buildActionKey(a, ph), func(ctx context.Context, arg interface{}) interface{} {
		res, err := actionImpl(ctx, arg.(*snapshot), deps, a, pkg)
		return actionResult{res, err}
	})

	ah := &actionHandle{
		analyzer: a,
		pkg:      pkg,
		promise:  promise,
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

func buildActionKey(a *analysis.Analyzer, ph *packageHandle) actionHandleKey {
	return actionHandleKey(source.Hashf("%p%s", a, ph.key[:]))
}

func (act *actionHandle) String() string {
	return fmt.Sprintf("%s@%s", act.analyzer, act.pkg.PkgPath())
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
			if dep.pkg == pkg {
				// Same package, different analysis (horizontal edge):
				// in-memory outputs of prerequisite analyzers
				// become inputs to this analysis pass.
				inputs[dep.analyzer] = data.result
			} else if dep.analyzer == analyzer { // (always true)
				// Same analysis, different package (vertical edge):
				// serialized facts produced by prerequisite analysis
				// become available to this analysis pass.
				for key, fact := range data.objectFacts {
					// Filter out facts related to objects
					// that are irrelevant downstream
					// (equivalently: not in the compiler export data).
					if !exportedFrom(key.obj, dep.pkg.types) {
						continue
					}
					objectFacts[key] = fact
				}
				for key, fact := range data.packageFacts {
					// TODO: filter out facts that belong to
					// packages not mentioned in the export data
					// to prevent side channels.
					packageFacts[key] = fact
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err // e.g. cancelled
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
				// TODO(adonovan): use bug.Errorf here so that we
				// detect crashes covered by our test suite.
				err = fmt.Errorf("analysis %s for package %s panicked: %v", analyzer.Name, pkg.PkgPath(), r)
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
	case *types.TypeName, *types.Const:
		return true
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
	// Apply type error analyzers. They augment type error diagnostics with their own fixes.
	var analyzers []*source.Analyzer
	for _, a := range s.View().Options().TypeErrorAnalyzers {
		analyzers = append(analyzers, a)
	}
	var errorAnalyzerDiag []*source.Diagnostic
	if pkg.HasTypeErrors() {
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
