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
	"sort"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	errors "golang.org/x/xerrors"
)

func (s *snapshot) Analyze(ctx context.Context, id string, analyzers ...*analysis.Analyzer) ([]*source.Error, error) {
	var roots []*actionHandle

	for _, a := range analyzers {
		ah, err := s.actionHandle(ctx, packageID(id), a)
		if err != nil {
			return nil, err
		}
		roots = append(roots, ah)
	}

	// Check if the context has been canceled before running the analyses.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	var results []*source.Error
	for _, ah := range roots {
		diagnostics, _, err := ah.analyze(ctx, s)
		if err != nil {
			return nil, err
		}
		results = append(results, diagnostics...)
	}
	return results, nil
}

type actionHandleKey string

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type actionHandle struct {
	handle *memoize.Handle

	analyzer *analysis.Analyzer
	pkg      *pkg
}

type actionData struct {
	diagnostics  []*source.Error
	result       interface{}
	objectFacts  map[objectFactKey]analysis.Fact
	packageFacts map[packageFactKey]analysis.Fact
	err          error
}

type objectFactKey struct {
	obj types.Object
	typ reflect.Type
}

type packageFactKey struct {
	pkg *types.Package
	typ reflect.Type
}

func (s *snapshot) actionHandle(ctx context.Context, id packageID, a *analysis.Analyzer) (*actionHandle, error) {
	ph := s.getPackage(id, source.ParseFull)
	if ph == nil {
		return nil, errors.Errorf("no package for %s", id)
	}
	act := s.getActionHandle(id, ph.mode, a)
	if act != nil {
		return act, nil
	}
	if len(ph.key) == 0 {
		return nil, errors.Errorf("no key for package %s", id)
	}
	pkg, err := ph.check(ctx, s)
	if err != nil {
		return nil, err
	}
	act = &actionHandle{
		analyzer: a,
		pkg:      pkg,
	}
	var deps []*actionHandle
	// Add a dependency on each required analyzers.
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
			importIDs := make([]string, 0, len(ph.m.deps))
			for _, importID := range ph.m.deps {
				importIDs = append(importIDs, string(importID))
			}
			sort.Strings(importIDs) // for determinism
			for _, importID := range importIDs {
				depActionHandle, err := s.actionHandle(ctx, packageID(importID), a)
				if err != nil {
					return nil, err
				}
				deps = append(deps, depActionHandle)
			}
		}
	}

	h := s.generation.Bind(buildActionKey(a, ph), func(ctx context.Context, arg memoize.Arg) interface{} {
		snapshot := arg.(*snapshot)
		// Analyze dependencies first.
		results, err := execAll(ctx, snapshot, deps)
		if err != nil {
			return &actionData{
				err: err,
			}
		}
		return runAnalysis(ctx, snapshot, a, pkg, results)
	}, nil)
	act.handle = h

	act = s.addActionHandle(act)
	return act, nil
}

func (act *actionHandle) analyze(ctx context.Context, snapshot *snapshot) ([]*source.Error, interface{}, error) {
	d, err := act.handle.Get(ctx, snapshot.generation, snapshot)
	if err != nil {
		return nil, nil, err
	}
	data, ok := d.(*actionData)
	if !ok {
		return nil, nil, errors.Errorf("unexpected type for %s:%s", act.pkg.ID(), act.analyzer.Name)
	}
	if data == nil {
		return nil, nil, errors.Errorf("unexpected nil analysis for %s:%s", act.pkg.ID(), act.analyzer.Name)
	}
	return data.diagnostics, data.result, data.err
}

func buildActionKey(a *analysis.Analyzer, ph *packageHandle) actionHandleKey {
	return actionHandleKey(hashContents([]byte(fmt.Sprintf("%p %s", a, string(ph.key)))))
}

func (act *actionHandle) String() string {
	return fmt.Sprintf("%s@%s", act.analyzer, act.pkg.PkgPath())
}

func execAll(ctx context.Context, snapshot *snapshot, actions []*actionHandle) (map[*actionHandle]*actionData, error) {
	var mu sync.Mutex
	results := make(map[*actionHandle]*actionData)

	g, ctx := errgroup.WithContext(ctx)
	for _, act := range actions {
		act := act
		g.Go(func() error {
			v, err := act.handle.Get(ctx, snapshot.generation, snapshot)
			if err != nil {
				return err
			}
			data, ok := v.(*actionData)
			if !ok {
				return errors.Errorf("unexpected type for %s: %T", act, v)
			}

			mu.Lock()
			defer mu.Unlock()
			results[act] = data

			return nil
		})
	}
	return results, g.Wait()
}

func runAnalysis(ctx context.Context, snapshot *snapshot, analyzer *analysis.Analyzer, pkg *pkg, deps map[*actionHandle]*actionData) (data *actionData) {
	data = &actionData{
		objectFacts:  make(map[objectFactKey]analysis.Fact),
		packageFacts: make(map[packageFactKey]analysis.Fact),
	}
	defer func() {
		if r := recover(); r != nil {
			event.Log(ctx, fmt.Sprintf("analysis panicked: %s", r), tag.Package.Of(pkg.ID()))
			data.err = errors.Errorf("analysis %s for package %s panicked: %v", analyzer.Name, pkg.PkgPath(), r)
		}
	}()

	// Plumb the output values of the dependencies
	// into the inputs of this action.  Also facts.
	inputs := make(map[*analysis.Analyzer]interface{})

	for depHandle, depData := range deps {
		if depHandle.pkg == pkg {
			// Same package, different analysis (horizontal edge):
			// in-memory outputs of prerequisite analyzers
			// become inputs to this analysis pass.
			inputs[depHandle.analyzer] = depData.result
		} else if depHandle.analyzer == analyzer { // (always true)
			// Same analysis, different package (vertical edge):
			// serialized facts produced by prerequisite analysis
			// become available to this analysis pass.
			for key, fact := range depData.objectFacts {
				// Filter out facts related to objects
				// that are irrelevant downstream
				// (equivalently: not in the compiler export data).
				if !exportedFrom(key.obj, depHandle.pkg.types) {
					continue
				}
				data.objectFacts[key] = fact
			}
			for key, fact := range depData.packageFacts {
				// TODO: filter out facts that belong to
				// packages not mentioned in the export data
				// to prevent side channels.

				data.packageFacts[key] = fact
			}
		}
	}

	var syntax []*ast.File
	for _, cgf := range pkg.compiledGoFiles {
		syntax = append(syntax, cgf.File)
	}

	var diagnostics []*analysis.Diagnostic

	// Run the analysis.
	pass := &analysis.Pass{
		Analyzer:   analyzer,
		Fset:       snapshot.view.session.cache.fset,
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
			diagnostics = append(diagnostics, &d)
		},
		ImportObjectFact: func(obj types.Object, ptr analysis.Fact) bool {
			if obj == nil {
				panic("nil object")
			}
			key := objectFactKey{obj, factType(ptr)}

			if v, ok := data.objectFacts[key]; ok {
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
			data.objectFacts[key] = fact // clobber any existing entry
		},
		ImportPackageFact: func(pkg *types.Package, ptr analysis.Fact) bool {
			if pkg == nil {
				panic("nil package")
			}
			key := packageFactKey{pkg, factType(ptr)}
			if v, ok := data.packageFacts[key]; ok {
				reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
				return true
			}
			return false
		},
		ExportPackageFact: func(fact analysis.Fact) {
			key := packageFactKey{pkg.types, factType(fact)}
			data.packageFacts[key] = fact // clobber any existing entry
		},
		AllObjectFacts: func() []analysis.ObjectFact {
			facts := make([]analysis.ObjectFact, 0, len(data.objectFacts))
			for k := range data.objectFacts {
				facts = append(facts, analysis.ObjectFact{Object: k.obj, Fact: data.objectFacts[k]})
			}
			return facts
		},
		AllPackageFacts: func() []analysis.PackageFact {
			facts := make([]analysis.PackageFact, 0, len(data.packageFacts))
			for k := range data.packageFacts {
				facts = append(facts, analysis.PackageFact{Package: k.pkg, Fact: data.packageFacts[k]})
			}
			return facts
		},
	}
	analysisinternal.SetTypeErrors(pass, pkg.typeErrors)

	if pkg.IsIllTyped() {
		data.err = errors.Errorf("analysis skipped due to errors in package: %v", pkg.GetErrors())
		return data
	}
	data.result, data.err = pass.Analyzer.Run(pass)
	if data.err != nil {
		return data
	}

	if got, want := reflect.TypeOf(data.result), pass.Analyzer.ResultType; got != want {
		data.err = errors.Errorf(
			"internal error: on package %s, analyzer %s returned a result of type %v, but declared ResultType %v",
			pass.Pkg.Path(), pass.Analyzer, got, want)
		return data
	}

	// disallow calls after Run
	pass.ExportObjectFact = func(obj types.Object, fact analysis.Fact) {
		panic(fmt.Sprintf("%s:%s: Pass.ExportObjectFact(%s, %T) called after Run", analyzer.Name, pkg.PkgPath(), obj, fact))
	}
	pass.ExportPackageFact = func(fact analysis.Fact) {
		panic(fmt.Sprintf("%s:%s: Pass.ExportPackageFact(%T) called after Run", analyzer.Name, pkg.PkgPath(), fact))
	}

	for _, diag := range diagnostics {
		srcErr, err := sourceError(ctx, snapshot, pkg, diag)
		if err != nil {
			event.Error(ctx, "unable to compute analysis error position", err, tag.Category.Of(diag.Category), tag.Package.Of(pkg.ID()))
			continue
		}
		if ctx.Err() != nil {
			data.err = ctx.Err()
			return data
		}
		data.diagnostics = append(data.diagnostics, srcErr)
	}
	return data
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

func factType(fact analysis.Fact) reflect.Type {
	t := reflect.TypeOf(fact)
	if t.Kind() != reflect.Ptr {
		panic(fmt.Sprintf("invalid Fact type: got %T, want pointer", t))
	}
	return t
}
