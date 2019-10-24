package cache

import (
	"context"
	"fmt"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

func (s *snapshot) Analyze(ctx context.Context, id string, analyzers []*analysis.Analyzer) ([]*source.Error, error) {
	var roots []*actionHandle

	for _, a := range analyzers {
		ah, err := s.actionHandle(ctx, packageID(id), source.ParseFull, a)
		if err != nil {
			return nil, err
		}
		ah.isroot = true
		roots = append(roots, ah)
	}

	// Check if the context has been canceled before running the analyses.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	var results []*source.Error
	for _, ah := range roots {
		diagnostics, _, err := ah.analyze(ctx)
		if err != nil {
			log.Error(ctx, "no results", err)
			continue
		}
		results = append(results, diagnostics...)
	}
	return results, nil
}

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type actionHandle struct {
	handle *memoize.Handle

	analyzer     *analysis.Analyzer
	deps         []*actionHandle
	pkg          *pkg
	isroot       bool
	objectFacts  map[objectFactKey]analysis.Fact
	packageFacts map[packageFactKey]analysis.Fact
}

type actionData struct {
	diagnostics []*source.Error
	result      interface{}
	err         error
}

type objectFactKey struct {
	obj types.Object
	typ reflect.Type
}

type packageFactKey struct {
	pkg *types.Package
	typ reflect.Type
}

func (s *snapshot) actionHandle(ctx context.Context, id packageID, mode source.ParseMode, a *analysis.Analyzer) (*actionHandle, error) {
	ah := s.getAction(id, mode, a)
	if ah != nil {
		return ah, nil
	}
	cph := s.getPackage(id, mode)
	if cph == nil {
		return nil, errors.Errorf("no CheckPackageHandle for %s:%v", id, mode == source.ParseExported)
	}
	if len(cph.key) == 0 {
		return nil, errors.Errorf("no key for CheckPackageHandle %s", id)
	}
	pkg, err := cph.check(ctx)
	if err != nil {
		return nil, err
	}
	ah = &actionHandle{
		analyzer: a,
		pkg:      pkg,
	}
	// Add a dependency on each required analyzers.
	for _, req := range a.Requires {
		reqActionHandle, err := s.actionHandle(ctx, id, mode, req)
		if err != nil {
			return nil, err
		}
		ah.deps = append(ah.deps, reqActionHandle)
	}
	// An analysis that consumes/produces facts
	// must run on the package's dependencies too.
	if len(a.FactTypes) > 0 {
		importIDs := make([]string, 0, len(cph.m.deps))
		for _, importID := range cph.m.deps {
			importIDs = append(importIDs, string(importID))
		}
		sort.Strings(importIDs) // for determinism
		for _, importID := range importIDs {
			depActionHandle, err := s.actionHandle(ctx, packageID(importID), source.ParseExported, a)
			if err != nil {
				return nil, err
			}
			ah.deps = append(ah.deps, depActionHandle)
		}
	}
	h := s.view.session.cache.store.Bind(buildActionKey(a, cph), func(ctx context.Context) interface{} {
		data := &actionData{}
		data.diagnostics, data.result, data.err = runAnalysis(ctx, s.view.session.cache.fset, ah)
		return data
	})
	ah.handle = h

	s.addAction(ah)
	return ah, nil
}

func (act *actionHandle) analyze(ctx context.Context) ([]*source.Error, interface{}, error) {
	v := act.handle.Get(ctx)
	if v == nil {
		return nil, nil, errors.Errorf("no analyses for %s", act.pkg.ID())
	}
	data := v.(*actionData)
	return data.diagnostics, data.result, data.err
}

func (act *actionHandle) cached() ([]*source.Error, interface{}, error) {
	v := act.handle.Cached()
	if v == nil {
		return nil, nil, errors.Errorf("no analyses for %s", act.pkg.ID())
	}
	data := v.(*actionData)
	return data.diagnostics, data.result, data.err
}

func buildActionKey(a *analysis.Analyzer, cph *checkPackageHandle) string {
	return hashContents([]byte(fmt.Sprintf("%p %s", a, string(cph.key))))
}

func (act *actionHandle) String() string {
	return fmt.Sprintf("%s@%s", act.analyzer, act.pkg.PkgPath())
}

func execAll(ctx context.Context, fset *token.FileSet, actions []*actionHandle) (map[*actionHandle][]*source.Error, map[*actionHandle]interface{}, error) {
	var (
		mu          sync.Mutex
		diagnostics = make(map[*actionHandle][]*source.Error)
		results     = make(map[*actionHandle]interface{})
	)

	g, ctx := errgroup.WithContext(ctx)
	for _, act := range actions {
		act := act
		g.Go(func() error {
			d, r, err := act.analyze(ctx)
			if err != nil {
				return err
			}

			mu.Lock()
			defer mu.Unlock()

			diagnostics[act] = d
			results[act] = r

			return nil
		})
	}
	return diagnostics, results, g.Wait()
}

func runAnalysis(ctx context.Context, fset *token.FileSet, act *actionHandle) ([]*source.Error, interface{}, error) {
	// Analyze dependencies.
	_, depResults, err := execAll(ctx, fset, act.deps)
	if err != nil {
		return nil, nil, err
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
			inputs[dep.analyzer] = depResults[dep]
		} else if dep.analyzer == act.analyzer { // (always true)
			// Same analysis, different package (vertical edge):
			// serialized facts produced by prerequisite analysis
			// become available to this analysis pass.
			inheritFacts(act, dep)
		}
	}

	var diagnostics []*analysis.Diagnostic

	// Run the analysis.
	pass := &analysis.Pass{
		Analyzer:   act.analyzer,
		Fset:       fset,
		Files:      act.pkg.GetSyntax(),
		Pkg:        act.pkg.GetTypes(),
		TypesInfo:  act.pkg.GetTypesInfo(),
		TypesSizes: act.pkg.GetTypesSizes(),
		ResultOf:   inputs,
		Report: func(d analysis.Diagnostic) {
			// Prefix the diagnostic category with the analyzer's name.
			if d.Category == "" {
				d.Category = act.analyzer.Name
			} else {
				d.Category = act.analyzer.Name + "." + d.Category
			}
			diagnostics = append(diagnostics, &d)
		},
		ImportObjectFact:  act.importObjectFact,
		ExportObjectFact:  act.exportObjectFact,
		ImportPackageFact: act.importPackageFact,
		ExportPackageFact: act.exportPackageFact,
		AllObjectFacts:    act.allObjectFacts,
		AllPackageFacts:   act.allPackageFacts,
	}

	if act.pkg.IsIllTyped() {
		return nil, nil, errors.Errorf("analysis skipped due to errors in package: %v", act.pkg.GetErrors())
	}
	result, err := pass.Analyzer.Run(pass)
	if err == nil {
		if got, want := reflect.TypeOf(result), pass.Analyzer.ResultType; got != want {
			err = errors.Errorf(
				"internal error: on package %s, analyzer %s returned a result of type %v, but declared ResultType %v",
				pass.Pkg.Path(), pass.Analyzer, got, want)
		}
	}

	// disallow calls after Run
	pass.ExportObjectFact = func(obj types.Object, fact analysis.Fact) {
		panic(fmt.Sprintf("%s: Pass.ExportObjectFact(%s, %T) called after Run", act, obj, fact))
	}
	pass.ExportPackageFact = func(fact analysis.Fact) {
		panic(fmt.Sprintf("%s: Pass.ExportPackageFact(%T) called after Run", act, fact))
	}

	var errors []*source.Error
	for _, diag := range diagnostics {
		srcErr, err := sourceError(ctx, act.pkg, diag)
		if err != nil {
			return nil, nil, err
		}
		errors = append(errors, srcErr)
	}
	return errors, result, err
}

// inheritFacts populates act.facts with
// those it obtains from its dependency, dep.
func inheritFacts(act, dep *actionHandle) {
	for key, fact := range dep.objectFacts {
		// Filter out facts related to objects
		// that are irrelevant downstream
		// (equivalently: not in the compiler export data).
		if !exportedFrom(key.obj, dep.pkg.types) {
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
func (act *actionHandle) importObjectFact(obj types.Object, ptr analysis.Fact) bool {
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
func (act *actionHandle) exportObjectFact(obj types.Object, fact analysis.Fact) {
	if obj.Pkg() != act.pkg.types {
		panic(fmt.Sprintf("internal error: in analysis %s of package %s: Fact.Set(%s, %T): can't set facts on objects belonging another package",
			act.analyzer, act.pkg.ID(), obj, fact))
	}

	key := objectFactKey{obj, factType(fact)}
	act.objectFacts[key] = fact // clobber any existing entry
}

// allObjectFacts implements Pass.AllObjectFacts.
func (act *actionHandle) allObjectFacts() []analysis.ObjectFact {
	facts := make([]analysis.ObjectFact, 0, len(act.objectFacts))
	for k := range act.objectFacts {
		facts = append(facts, analysis.ObjectFact{Object: k.obj, Fact: act.objectFacts[k]})
	}
	return facts
}

// importPackageFact implements Pass.ImportPackageFact.
// Given a non-nil pointer ptr of type *T, where *T satisfies Fact,
// fact copies the fact value to *ptr.
func (act *actionHandle) importPackageFact(pkg *types.Package, ptr analysis.Fact) bool {
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
func (act *actionHandle) exportPackageFact(fact analysis.Fact) {
	key := packageFactKey{act.pkg.types, factType(fact)}
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
func (act *actionHandle) allPackageFacts() []analysis.PackageFact {
	facts := make([]analysis.PackageFact, 0, len(act.packageFacts))
	for k := range act.packageFacts {
		facts = append(facts, analysis.PackageFact{Package: k.pkg, Fact: act.packageFacts[k]})
	}
	return facts
}
