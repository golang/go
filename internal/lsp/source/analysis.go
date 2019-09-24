// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is largely based on go/analysis/internal/checker/checker.go.

package source

import (
	"context"
	"fmt"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func analyze(ctx context.Context, v View, cphs []CheckPackageHandle, analyzers []*analysis.Analyzer) ([]*Action, error) {
	ctx, done := trace.StartSpan(ctx, "source.analyze")
	defer done()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Build nodes for initial packages.
	var roots []*Action
	for _, a := range analyzers {
		for _, cph := range cphs {
			pkg, err := cph.Check(ctx)
			if err != nil {
				return nil, err
			}
			root, err := pkg.GetActionGraph(ctx, a)
			if err != nil {
				return nil, err
			}
			root.isroot = true
			root.view = v
			roots = append(roots, root)
		}
	}

	// Execute the graph in parallel.
	if err := execAll(ctx, v.Session().Cache().FileSet(), roots); err != nil {
		return nil, err
	}
	return roots, nil
}

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type Action struct {
	once        sync.Once
	Analyzer    *analysis.Analyzer
	Pkg         Package
	Deps        []*Action
	diagnostics []analysis.Diagnostic

	pass         *analysis.Pass
	isroot       bool
	objectFacts  map[objectFactKey]analysis.Fact
	packageFacts map[packageFactKey]analysis.Fact
	inputs       map[*analysis.Analyzer]interface{}
	result       interface{}
	err          error
	duration     time.Duration
	view         View
}

type objectFactKey struct {
	obj types.Object
	typ reflect.Type
}

type packageFactKey struct {
	pkg *types.Package
	typ reflect.Type
}

func (act *Action) String() string {
	return fmt.Sprintf("%s@%s", act.Analyzer, act.Pkg.PkgPath())
}

func execAll(ctx context.Context, fset *token.FileSet, actions []*Action) error {
	g, ctx := errgroup.WithContext(ctx)
	for _, act := range actions {
		act := act
		g.Go(func() error {
			return act.exec(ctx, fset)
		})
	}
	return g.Wait()
}

func (act *Action) exec(ctx context.Context, fset *token.FileSet) error {
	var err error
	act.once.Do(func() {
		err = act.execOnce(ctx, fset)
	})
	return err
}

func (act *Action) execOnce(ctx context.Context, fset *token.FileSet) error {
	// Analyze dependencies.
	if err := execAll(ctx, fset, act.Deps); err != nil {
		return err
	}

	// Report an error if any dependency failed.
	var failed []string
	for _, dep := range act.Deps {
		if dep.err != nil {
			failed = append(failed, fmt.Sprintf("%s: %v", dep.String(), dep.err))
		}
	}
	if failed != nil {
		sort.Strings(failed)
		act.err = errors.Errorf("failed prerequisites: %s", strings.Join(failed, ", "))
		return act.err
	}

	// Plumb the output values of the dependencies
	// into the inputs of this action.  Also facts.
	inputs := make(map[*analysis.Analyzer]interface{})
	act.objectFacts = make(map[objectFactKey]analysis.Fact)
	act.packageFacts = make(map[packageFactKey]analysis.Fact)
	for _, dep := range act.Deps {
		if dep.Pkg == act.Pkg {
			// Same package, different analysis (horizontal edge):
			// in-memory outputs of prerequisite analyzers
			// become inputs to this analysis pass.
			inputs[dep.Analyzer] = dep.result

		} else if dep.Analyzer == act.Analyzer { // (always true)
			// Same analysis, different package (vertical edge):
			// serialized facts produced by prerequisite analysis
			// become available to this analysis pass.
			inheritFacts(act, dep)
		}
	}

	// Run the analysis.
	pass := &analysis.Pass{
		Analyzer:          act.Analyzer,
		Fset:              fset,
		Files:             act.Pkg.GetSyntax(ctx),
		Pkg:               act.Pkg.GetTypes(),
		TypesInfo:         act.Pkg.GetTypesInfo(),
		TypesSizes:        act.Pkg.GetTypesSizes(),
		ResultOf:          inputs,
		Report:            func(d analysis.Diagnostic) { act.diagnostics = append(act.diagnostics, d) },
		ImportObjectFact:  act.importObjectFact,
		ExportObjectFact:  act.exportObjectFact,
		ImportPackageFact: act.importPackageFact,
		ExportPackageFact: act.exportPackageFact,
		AllObjectFacts:    act.allObjectFacts,
		AllPackageFacts:   act.allPackageFacts,
	}
	act.pass = pass

	if act.Pkg.IsIllTyped() {
		act.err = errors.Errorf("analysis skipped due to errors in package: %v", act.Pkg.GetErrors())
	} else {
		act.result, act.err = pass.Analyzer.Run(pass)
		if act.err == nil {
			if got, want := reflect.TypeOf(act.result), pass.Analyzer.ResultType; got != want {
				act.err = errors.Errorf(
					"internal error: on package %s, analyzer %s returned a result of type %v, but declared ResultType %v",
					pass.Pkg.Path(), pass.Analyzer, got, want)
			}
		}
	}

	// disallow calls after Run
	pass.ExportObjectFact = nil
	pass.ExportPackageFact = nil

	return act.err
}

// inheritFacts populates act.facts with
// those it obtains from its dependency, dep.
func inheritFacts(act, dep *Action) {
	for key, fact := range dep.objectFacts {
		// Filter out facts related to objects
		// that are irrelevant downstream
		// (equivalently: not in the compiler export data).
		if !exportedFrom(key.obj, dep.Pkg.GetTypes()) {
			continue
		}
		act.objectFacts[key] = fact
	}

	for key, fact := range dep.packageFacts {
		// TODO: filter out facts that belong to
		// packages not mentioned in the export data
		// to prevent side channels.

		act.packageFacts[key] = fact
	}
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
func (act *Action) importObjectFact(obj types.Object, ptr analysis.Fact) bool {
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
func (act *Action) exportObjectFact(obj types.Object, fact analysis.Fact) {
	if act.pass.ExportObjectFact == nil {
		panic(fmt.Sprintf("%s: Pass.ExportObjectFact(%s, %T) called after Run", act, obj, fact))
	}

	if obj.Pkg() != act.Pkg.GetTypes() {
		panic(fmt.Sprintf("internal error: in analysis %s of package %s: Fact.Set(%s, %T): can't set facts on objects belonging another package",
			act.Analyzer, act.Pkg, obj, fact))
	}

	key := objectFactKey{obj, factType(fact)}
	act.objectFacts[key] = fact // clobber any existing entry
}

// allObjectFacts implements Pass.AllObjectFacts.
func (act *Action) allObjectFacts() []analysis.ObjectFact {
	facts := make([]analysis.ObjectFact, 0, len(act.objectFacts))
	for k := range act.objectFacts {
		facts = append(facts, analysis.ObjectFact{Object: k.obj, Fact: act.objectFacts[k]})
	}
	return facts
}

// importPackageFact implements Pass.ImportPackageFact.
// Given a non-nil pointer ptr of type *T, where *T satisfies Fact,
// fact copies the fact value to *ptr.
func (act *Action) importPackageFact(pkg *types.Package, ptr analysis.Fact) bool {
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
func (act *Action) exportPackageFact(fact analysis.Fact) {
	if act.pass.ExportPackageFact == nil {
		panic(fmt.Sprintf("%s: Pass.ExportPackageFact(%T) called after Run", act, fact))
	}

	key := packageFactKey{act.pass.Pkg, factType(fact)}
	act.packageFacts[key] = fact // clobber any existing entry
}

func factType(fact analysis.Fact) reflect.Type {
	t := reflect.TypeOf(fact)
	if t.Kind() != reflect.Ptr {
		panic(fmt.Sprintf("invalid Fact type: got %T, want pointer", t))
	}
	return t
}

// allObjectFacts implements Pass.AllObjectFacts.
func (act *Action) allPackageFacts() []analysis.PackageFact {
	facts := make([]analysis.PackageFact, 0, len(act.packageFacts))
	for k := range act.packageFacts {
		facts = append(facts, analysis.PackageFact{Package: k.pkg, Fact: act.packageFacts[k]})
	}
	return facts
}
