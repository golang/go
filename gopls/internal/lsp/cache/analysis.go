// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

// This file defines gopls' driver for modular static analysis (go/analysis).

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/gopls/internal/lsp/filecache"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/facts"
	"golang.org/x/tools/internal/gcimporter"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

/*

   DESIGN

   An analysis request is for a set of analyzers and an individual
   package ID, notated (a*, p). The result is the set of diagnostics
   for that package. It could easily be generalized to a set of
   packages, (a*, p*), and perhaps should be, to improve performance
   versus calling it in a loop.

   The snapshot holds a cache (persistent.Map) of entries keyed by
   (a*, p) pairs ("analysisKey") that have been requested so far. Some
   of these entries may be invalidated during snapshot cloning after a
   modification event.  The cache maps each (a*, p) to a promise of
   the analysis result or "analysisSummary". The summary contains the
   results of analysis (e.g. diagnostics) as well as the intermediate
   results required by the recursion, such as serialized types and
   facts.

   The promise represents the result of a call to analyzeImpl, which
   type-checks a package and then applies a graph of analyzers to it
   in parallel postorder. (These graph edges are "horizontal": within
   the same package.) First, analyzeImpl reads the source files of
   package p, and obtains (recursively) the results of the "vertical"
   dependencies (i.e. analyzers applied to the packages imported by
   p). Only the subset of analyzers that use facts need be executed
   recursively, but even if this subset is empty, the step is still
   necessary because it provides type information. It is possible that
   a package may need to be type-checked and analyzed twice, for
   different subsets of analyzers, but the overlap is typically
   insignificant.

   With the file contents and the results of vertical dependencies,
   analyzeImpl is then in a position to produce a key representing the
   unit of work (parsing, type-checking, and analysis) that it has to
   do. The key is a cryptographic hash of the "recipe" for this step,
   including the Metadata, the file contents, the set of analyzers,
   and the type and fact information from the vertical dependencies.

   The key is sought in a machine-global persistent file-system based
   cache. If this gopls process, or another gopls process on the same
   machine, has already performed this analysis step, analyzeImpl will
   make a cache hit and load the serialized summary of the results. If
   not, it will have to proceed to type-checking and analysis, and
   write a new cache entry. The entry contains serialized types
   (export data) and analysis facts.

   For types, we use "shallow" export data. Historically, the Go
   compiler always produced a summary of the types for a given package
   that included types from other packages that it indirectly
   referenced: "deep" export data. This had the advantage that the
   compiler (and analogous tools such as gopls) need only load one
   file per direct import.  However, it meant that the files tended to
   get larger based on the level of the package in the import
   graph. For example, higher-level packages in the kubernetes module
   have over 1MB of "deep" export data, even when they have almost no
   content of their own, merely because they mention a major type that
   references many others. In pathological cases the export data was
   300x larger than the source for a package due to this quadratic
   growth.

   "Shallow" export data means that the serialized types describe only
   a single package. If those types mention types from other packages,
   the type checker may need to request additional packages beyond
   just the direct imports. This means type information for the entire
   transitive closure of imports may need to be available just in
   case. After a cache hit or a cache miss, the summary is
   postprocessed so that it contains the union of export data payloads
   of all its direct dependencies.

   For correct dependency analysis, the digest used as a cache key
   must reflect the "deep" export data, so it is derived recursively
   from the transitive closure. As an optimization, we needn't include
   every package of the transitive closure in the deep hash, only the
   packages that were actually requested by the type checker. This
   allows changes to a package that have no effect on its export data
   to be "pruned". The direct consumer will need to be re-executed,
   but if its export data is unchanged as a result, then indirect
   consumers may not need to be re-executed.  This allows, for example,
   one to insert a print statement in a function and not "rebuild" the
   whole application (though export data does record line numbers of
   types which may be perturbed by otherwise insignificant changes.)

   The summary must record whether a package is transitively
   error-free (whether it would compile) because many analyzers are
   not safe to run on packages with inconsistent types.

   For fact encoding, we use the same fact set as the unitchecker
   (vet) to record and serialize analysis facts. The fact
   serialization mechanism is analogous to "deep" export data.

*/

// TODO(adonovan):
// - Profile + optimize:
//   - on a cold run, mostly type checking + export data, unsurprisingly.
//   - on a hot-disk run, mostly type checking the IWL.
//     Would be nice to have a benchmark that separates this out.
//   - measure and record in the code the typical operation times
//     and file sizes (export data + facts = cache entries).
// - Do "port the old logic" tasks (see TODO in actuallyAnalyze).
// - Add a (white-box) test of pruning when a change doesn't affect export data.
// - Optimise pruning based on subset of packages mentioned in exportdata.
// - Better logging so that it is possible to deduce why an analyzer
//   is not being run--often due to very indirect failures.
//   Even if the ultimate consumer decides to ignore errors,
//   tests and other situations want to be assured of freedom from
//   errors, not just missing results. This should be recorded.
// - Check that the event trace is intelligible.
// - Split this into a subpackage, gopls/internal/lsp/cache/driver,
//   consisting of this file and three helpers from errors.go.
//   The (*snapshot).Analyze method would stay behind and make calls
//   to the driver package.
//   Steps:
//   - define a narrow driver.Snapshot interface with only these methods:
//        Metadata(PackageID) source.Metadata
//        GetFile(Context, URI) (source.FileHandle, error)
//        View() *View // for Options
//   - define a State type that encapsulates the persistent map
//     (with its own mutex), and has methods:
//        New() *State
//        Clone(invalidate map[PackageID]bool) *State
//        Destroy()
//   - share cache.{goVersionRx,parseGoImpl}

var born = time.Now()

// Analyze applies a set of analyzers to the package denoted by id,
// and returns their diagnostics for that package.
//
// The analyzers list must be duplicate free; order does not matter.
//
// Precondition: all analyzers within the process have distinct names.
// (The names are relied on by the serialization logic.)
func (s *snapshot) Analyze(ctx context.Context, id PackageID, analyzers []*source.Analyzer) ([]*source.Diagnostic, error) {
	if false { // debugging
		log.Println("Analyze@", time.Since(born)) // called after the 7s IWL in k8s
	}

	// Filter and sort enabled root analyzers.
	// A disabled analyzer may still be run if required by another.
	toSrc := make(map[*analysis.Analyzer]*source.Analyzer)
	var enabled []*analysis.Analyzer
	for _, a := range analyzers {
		if a.IsEnabled(s.view.Options()) {
			toSrc[a.Analyzer] = a
			enabled = append(enabled, a.Analyzer)
		}
	}
	sort.Slice(enabled, func(i, j int) bool {
		return enabled[i].Name < enabled[j].Name
	})

	// Register fact types of required analyzers.
	for _, a := range requiredAnalyzers(enabled) {
		for _, f := range a.FactTypes {
			gob.Register(f)
		}
	}

	if false { // debugging
		// TODO(adonovan): use proper tracing.
		t0 := time.Now()
		defer func() {
			log.Printf("%v for analyze(%s, %s)", time.Since(t0), id, enabled)
		}()
	}

	// Run the analysis.
	res, err := s.analyze(ctx, id, enabled)
	if err != nil {
		return nil, err
	}

	// Report diagnostics only from enabled actions that succeeded.
	// Errors from creating or analyzing packages are ignored.
	// Diagnostics are reported in the order of the analyzers argument.
	//
	// TODO(adonovan): ignoring action errors gives the caller no way
	// to distinguish "there are no problems in this code" from
	// "the code (or analyzers!) are so broken that we couldn't even
	// begin the analysis you asked for".
	// Even if current callers choose to discard the
	// results, we should propagate the per-action errors.
	var results []*source.Diagnostic
	for _, a := range enabled {
		summary := res.Actions[a.Name]
		if summary.Err != "" {
			continue // action failed
		}
		for _, gobDiag := range summary.Diagnostics {
			results = append(results, toSourceDiagnostic(toSrc[a], &gobDiag))
		}
	}
	return results, nil
}

// analysisKey is the type of keys in the snapshot.analyses map.
type analysisKey struct {
	analyzerNames string
	pkgid         PackageID
}

func (key analysisKey) String() string {
	return fmt.Sprintf("%s@%s", key.analyzerNames, key.pkgid)
}

// analyzeSummary is a gob-serializable summary of successfully
// applying a list of analyzers to a package.
type analyzeSummary struct {
	PkgPath        PackagePath // types.Package.Path() (needed to decode export data)
	Export         []byte
	DeepExportHash source.Hash // hash of reflexive transitive closure of export data
	Compiles       bool        // transitively free of list/parse/type errors
	Actions        actionsMap  // map from analyzer name to analysis results (*actionSummary)

	// Not serialized: populated after the summary is computed or deserialized.
	allExport map[PackagePath][]byte // transitive export data
}

// actionsMap defines a stable Gob encoding for a map.
// TODO(adonovan): generalize and move to a library when we can use generics.
type actionsMap map[string]*actionSummary

var _ gob.GobEncoder = (actionsMap)(nil)
var _ gob.GobDecoder = (*actionsMap)(nil)

type actionsMapEntry struct {
	K string
	V *actionSummary
}

func (m actionsMap) GobEncode() ([]byte, error) {
	entries := make([]actionsMapEntry, 0, len(m))
	for k, v := range m {
		entries = append(entries, actionsMapEntry{k, v})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].K < entries[j].K
	})
	var buf bytes.Buffer
	err := gob.NewEncoder(&buf).Encode(entries)
	return buf.Bytes(), err
}

func (m *actionsMap) GobDecode(data []byte) error {
	var entries []actionsMapEntry
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&entries); err != nil {
		return err
	}
	*m = make(actionsMap, len(entries))
	for _, e := range entries {
		(*m)[e.K] = e.V
	}
	return nil
}

// actionSummary is a gob-serializable summary of one possibly failed analysis action.
// If Err is non-empty, the other fields are undefined.
type actionSummary struct {
	Facts       []byte      // the encoded facts.Set
	FactsHash   source.Hash // hash(Facts)
	Diagnostics []gobDiagnostic
	Err         string // "" => success
}

// analyze is a memoization of analyzeImpl.
func (s *snapshot) analyze(ctx context.Context, id PackageID, analyzers []*analysis.Analyzer) (*analyzeSummary, error) {
	// Use the sorted list of names of analyzers in the key.
	//
	// TODO(adonovan): opt: account for analysis results at a
	// finer grain to avoid duplicate work when a
	// a proper subset of analyzers is requested?
	// In particular, TypeErrorAnalyzers don't use facts
	// but need to request vdeps just for type information.
	names := make([]string, 0, len(analyzers))
	for _, a := range analyzers {
		names = append(names, a.Name)
	}
	// This key describes the result of applying a list of analyzers to a package.
	key := analysisKey{strings.Join(names, ","), id}

	// An analysisPromise represents the result of loading, parsing,
	// type-checking and analyzing a single package.
	type analysisPromise struct {
		promise *memoize.Promise // [analyzeImplResult]
	}

	type analyzeImplResult struct {
		summary *analyzeSummary
		err     error
	}

	// Access the map once, briefly, and atomically.
	s.mu.Lock()
	entry, hit := s.analyses.Get(key)
	if !hit {
		entry = analysisPromise{
			promise: memoize.NewPromise("analysis", func(ctx context.Context, arg interface{}) interface{} {
				summary, err := analyzeImpl(ctx, arg.(*snapshot), analyzers, id)
				return analyzeImplResult{summary, err}
			}),
		}
		s.analyses.Set(key, entry, nil) // nothing needs releasing
	}
	s.mu.Unlock()

	// Await result.
	ap := entry.(analysisPromise)
	v, err := s.awaitPromise(ctx, ap.promise)
	if err != nil {
		return nil, err // e.g. cancelled
	}
	res := v.(analyzeImplResult)
	return res.summary, res.err
}

// analyzeImpl applies a list of analyzers (plus any others
// transitively required by them) to a package.  It succeeds as long
// as it could produce a types.Package, even if there were direct or
// indirect list/parse/type errors, and even if all the analysis
// actions failed. It usually fails only if the package was unknown,
// a file was missing, or the operation was cancelled.
//
// Postcondition: analyzeImpl must not continue to use the snapshot
// (in background goroutines) after it has returned; see memoize.RefCounted.
func analyzeImpl(ctx context.Context, snapshot *snapshot, analyzers []*analysis.Analyzer, id PackageID) (*analyzeSummary, error) {
	m := snapshot.Metadata(id)
	if m == nil {
		return nil, fmt.Errorf("no metadata for %s", id)
	}

	// Also, load the contents of each "compiled" Go file through
	// the snapshot's cache.
	// (These are all cache hits as files are pre-loaded following packages.Load)
	compiledGoFiles := make([]source.FileHandle, len(m.CompiledGoFiles))
	for i, uri := range m.CompiledGoFiles {
		fh, err := snapshot.GetFile(ctx, uri)
		if err != nil {
			return nil, err // e.g. canceled
		}
		compiledGoFiles[i] = fh
	}

	// Recursively analyze each "vertical" dependency
	// for its types.Package and (perhaps) analysis.Facts.
	// If any of them fails to produce a package, we cannot continue.
	// We request only the analyzers that produce facts.
	vdeps := make(map[PackageID]*analyzeSummary)
	{
		var group errgroup.Group

		// Analyze vertical dependencies.
		// We request only the required analyzers that use facts.
		var useFacts []*analysis.Analyzer
		for _, a := range requiredAnalyzers(analyzers) {
			if len(a.FactTypes) > 0 {
				useFacts = append(useFacts, a)
			}
		}
		var vdepsMu sync.Mutex
		for _, id := range m.DepsByPkgPath {
			id := id
			group.Go(func() error {
				res, err := snapshot.analyze(ctx, id, useFacts)
				if err != nil {
					return err // cancelled, or failed to produce a package
				}

				vdepsMu.Lock()
				vdeps[id] = res
				vdepsMu.Unlock()
				return nil
			})
		}

		if err := group.Wait(); err != nil {
			return nil, err
		}
	}

	// Inv: analyze() of all vdeps succeeded (though some actions may have failed).

	// We no longer depend on the snapshot.
	snapshot = nil

	// At this point we have the action results (serialized
	// packages and facts) of our immediate dependencies,
	// and the metadata and content of this package.
	//
	// We now compute a hash for all our inputs, and consult a
	// global cache of promised results. If nothing material
	// has changed, we'll make a hit in the shared cache.
	//
	// The hash of our inputs is based on the serialized export
	// data and facts so that immaterial changes can be pruned
	// without decoding.
	key := analysisCacheKey(analyzers, m, compiledGoFiles, vdeps)

	// Access the cache.
	var summary *analyzeSummary
	const cacheKind = "analysis"
	if data, err := filecache.Get(cacheKind, key); err == nil {
		// cache hit
		mustDecode(data, &summary)

	} else if err != filecache.ErrNotFound {
		return nil, bug.Errorf("internal error reading shared cache: %v", err)

	} else {
		// Cache miss: do the work.
		var err error
		summary, err = actuallyAnalyze(ctx, analyzers, m, vdeps, compiledGoFiles)
		if err != nil {
			return nil, err
		}
		data := mustEncode(summary)
		if false {
			log.Printf("Set key=%d value=%d id=%s\n", len(key), len(data), id)
		}
		if err := filecache.Set(cacheKind, key, data); err != nil {
			return nil, fmt.Errorf("internal error updating shared cache: %v", err)
		}
	}

	// Hit or miss, we need to merge the export data from
	// dependencies so that it includes all the types
	// that might be summoned by the type checker.
	//
	// TODO(adonovan): opt: reduce this set by recording
	// which packages were actually summoned by insert().
	// (Just makes map smaller; probably marginal?)
	allExport := make(map[PackagePath][]byte)
	for _, vdep := range vdeps {
		for k, v := range vdep.allExport {
			allExport[k] = v
		}
	}
	allExport[m.PkgPath] = summary.Export
	summary.allExport = allExport

	return summary, nil
}

// analysisCacheKey returns a cache key that is a cryptographic digest
// of the all the values that might affect type checking and analysis:
// the analyzer names, package metadata, names and contents of
// compiled Go files, and vdeps information (export data and facts).
//
// TODO(adonovan): safety: define our own flavor of Metadata
// containing just the fields we need, and using it in the subsequent
// logic, to keep us honest about hashing all parts that matter?
func analysisCacheKey(analyzers []*analysis.Analyzer, m *source.Metadata, compiledGoFiles []source.FileHandle, vdeps map[PackageID]*analyzeSummary) [sha256.Size]byte {
	hasher := sha256.New()

	// In principle, a key must be the hash of an
	// unambiguous encoding of all the relevant data.
	// If it's ambiguous, we risk collisions.

	// analyzers
	fmt.Fprintf(hasher, "analyzers: %d\n", len(analyzers))
	for _, a := range analyzers {
		fmt.Fprintln(hasher, a.Name)
	}

	// package metadata
	fmt.Fprintf(hasher, "package: %s %s %s\n", m.ID, m.Name, m.PkgPath)
	// We can ignore m.DepsBy{Pkg,Import}Path: although the logic
	// uses those fields, we account for them by hashing vdeps.

	// type sizes
	// This assertion is safe, but if a black-box implementation
	// is ever needed, record Sizeof(*int) and Alignof(int64).
	sz := m.TypesSizes.(*types.StdSizes)
	fmt.Fprintf(hasher, "sizes: %d %d\n", sz.WordSize, sz.MaxAlign)

	// metadata errors: used for 'compiles' field
	fmt.Fprintf(hasher, "errors: %d", len(m.Errors))

	// module Go version
	if m.Module != nil && m.Module.GoVersion != "" {
		fmt.Fprintf(hasher, "go %s\n", m.Module.GoVersion)
	}

	// file names and contents
	fmt.Fprintf(hasher, "files: %d\n", len(compiledGoFiles))
	for _, fh := range compiledGoFiles {
		fmt.Fprintln(hasher, fh.FileIdentity())
	}

	// vdeps, in PackageID order
	depIDs := make([]string, 0, len(vdeps))
	for depID := range vdeps {
		depIDs = append(depIDs, string(depID))
	}
	sort.Strings(depIDs)
	for _, depID := range depIDs {
		vdep := vdeps[PackageID(depID)]
		fmt.Fprintf(hasher, "dep: %s\n", vdep.PkgPath)
		fmt.Fprintf(hasher, "export: %s\n", vdep.DeepExportHash)

		// action results: errors and facts
		names := make([]string, 0, len(vdep.Actions))
		for name := range vdep.Actions {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			summary := vdep.Actions[name]
			fmt.Fprintf(hasher, "action %s\n", name)
			if summary.Err != "" {
				fmt.Fprintf(hasher, "error %s\n", summary.Err)
			} else {
				fmt.Fprintf(hasher, "facts %s\n", summary.FactsHash)
				// We can safely omit summary.diagnostics
				// from the key since they have no downstream effect.
			}
		}
	}

	var hash [sha256.Size]byte
	hasher.Sum(hash[:0])
	return hash
}

// actuallyAnalyze implements the cache-miss case.
// This function does not access the snapshot.
func actuallyAnalyze(ctx context.Context, analyzers []*analysis.Analyzer, m *source.Metadata, vdeps map[PackageID]*analyzeSummary, compiledGoFiles []source.FileHandle) (*analyzeSummary, error) {

	// Create a local FileSet for processing this package only.
	fset := token.NewFileSet()

	// Parse only the "compiled" Go files.
	// Do the computation in parallel.
	parsed := make([]*source.ParsedGoFile, len(compiledGoFiles))
	{
		var group errgroup.Group
		for i, fh := range compiledGoFiles {
			i, fh := i, fh
			group.Go(func() error {
				// Call parseGoImpl directly, not the caching wrapper,
				// as cached ASTs require the global FileSet.
				pgf, err := parseGoImpl(ctx, fset, fh, source.ParseFull)
				parsed[i] = pgf
				return err
			})
		}
		if err := group.Wait(); err != nil {
			return nil, err // cancelled, or catastrophic error (e.g. missing file)
		}
	}

	// Type-check the package.
	pkg := typeCheckForAnalysis(fset, parsed, m, vdeps)

	// Build a map of PkgPath to *Package for all packages mentioned
	// in exportdata for use by facts.
	pkg.factsDecoder = facts.NewDecoder(pkg.types)

	// Poll cancellation state.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// TODO(adonovan): port the old logic to:
	// - gather go/packages diagnostics from m.Errors? (port goPackagesErrorDiagnostics)
	// - record unparseable file URIs so we can suppress type errors for these files.
	// - gather diagnostics from expandErrors + typeErrorDiagnostics + depsErrors.

	// -- analysis --

	// Build action graph for this package.
	// Each graph node (action) is one unit of analysis.
	actions := make(map[*analysis.Analyzer]*action)
	var mkAction func(a *analysis.Analyzer) *action
	mkAction = func(a *analysis.Analyzer) *action {
		act, ok := actions[a]
		if !ok {
			var hdeps []*action
			for _, req := range a.Requires {
				hdeps = append(hdeps, mkAction(req))
			}
			act = &action{a: a, pkg: pkg, vdeps: vdeps, hdeps: hdeps}
			actions[a] = act
		}
		return act
	}

	// Build actions for initial package.
	var roots []*action
	for _, a := range analyzers {
		roots = append(roots, mkAction(a))
	}

	// Execute the graph in parallel.
	execActions(roots)

	// Don't return (or cache) the result in case of cancellation.
	if err := ctx.Err(); err != nil {
		return nil, err // cancelled
	}

	// Return summaries only for the requested actions.
	summaries := make(map[string]*actionSummary)
	for _, act := range roots {
		summaries[act.a.Name] = act.summary
	}

	return &analyzeSummary{
		PkgPath:        PackagePath(pkg.types.Path()),
		Export:         pkg.export,
		DeepExportHash: pkg.deepExportHash,
		Compiles:       pkg.compiles,
		Actions:        summaries,
	}, nil
}

func typeCheckForAnalysis(fset *token.FileSet, parsed []*source.ParsedGoFile, m *source.Metadata, vdeps map[PackageID]*analyzeSummary) *analysisPackage {
	if false { // debugging
		log.Println("typeCheckForAnalysis", m.PkgPath)
	}

	pkg := &analysisPackage{
		m:        m,
		fset:     fset,
		parsed:   parsed,
		files:    make([]*ast.File, len(parsed)),
		compiles: len(m.Errors) == 0, // false => list error
		types:    types.NewPackage(string(m.PkgPath), string(m.Name)),
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		typesSizes: m.TypesSizes,
	}
	typeparams.InitInstanceInfo(pkg.typesInfo)

	for i, p := range parsed {
		pkg.files[i] = p.File
		if p.ParseErr != nil {
			pkg.compiles = false // parse error
		}
	}

	// Unsafe is special.
	if m.PkgPath == "unsafe" {
		pkg.types = types.Unsafe
		return pkg
	}

	// Compute the union of transitive export data.
	// (The actual values are shared, and not serialized.)
	allExport := make(map[PackagePath][]byte)
	for _, vdep := range vdeps {
		for k, v := range vdep.allExport {
			allExport[k] = v
		}

		if !vdep.Compiles {
			pkg.compiles = false // transitive error
		}
	}

	// exportHasher computes a hash of the names and export data of
	// each package that was actually loaded during type checking.
	//
	// Because we use shallow export data, the hash for dependency
	// analysis must incorporate indirect dependencies. As an
	// optimization, we include only those that were actually
	// used, which may be a small subset of those available.
	//
	// TODO(adonovan): opt: even better would be to implement a
	// traversal over the package API like facts.NewDecoder does
	// and only mention that set of packages in the hash.
	// Perhaps there's a way to do that more efficiently.
	//
	// TODO(adonovan): opt: record the shallow hash alongside the
	// shallow export data in the allExport map to avoid repeatedly
	// hashing the export data.
	//
	// The writes to hasher below assume that type checking imports
	// packages in a deterministic order.
	exportHasher := sha256.New()
	hashExport := func(pkgPath PackagePath, export []byte) {
		fmt.Fprintf(exportHasher, "%s %d ", pkgPath, len(export))
		exportHasher.Write(export)
	}

	// importer state
	var (
		insert    func(p *types.Package, name string)
		importMap = make(map[string]*types.Package) // keys are PackagePaths
	)
	loadFromExportData := func(pkgPath PackagePath) (*types.Package, error) {
		export, ok := allExport[pkgPath]
		if !ok {
			return nil, bug.Errorf("missing export data for %q", pkgPath)
		}
		hashExport(pkgPath, export)
		imported, err := gcimporter.IImportShallow(fset, importMap, export, string(pkgPath), insert)
		if err != nil {
			return nil, bug.Errorf("invalid export data for %q: %v", pkgPath, err)
		}
		return imported, nil
	}
	insert = func(p *types.Package, name string) {
		imported, err := loadFromExportData(PackagePath(p.Path()))
		if err != nil {
			log.Fatalf("internal error: %v", err)
		}
		if imported != p {
			log.Fatalf("internal error: inconsistent packages")
		}
	}

	cfg := &types.Config{
		Sizes: m.TypesSizes,
		Error: func(e error) {
			pkg.compiles = false // type error
			pkg.typeErrors = append(pkg.typeErrors, e.(types.Error))
		},
		Importer: importerFunc(func(importPath string) (*types.Package, error) {
			if importPath == "unsafe" {
				return types.Unsafe, nil // unsafe has no export data
			}

			// Beware that returning an error from this function
			// will cause the type checker to synthesize a fake
			// package whose Path is importPath, potentially
			// losing a vendor/ prefix. If type-checking errors
			// are swallowed, these packages may be confusing.

			id, ok := m.DepsByImpPath[ImportPath(importPath)]
			if !ok {
				// The import syntax is inconsistent with the metadata.
				// This could be because the import declaration was
				// incomplete and the metadata only includes complete
				// imports; or because the metadata ignores import
				// edges that would lead to cycles in the graph.
				return nil, fmt.Errorf("missing metadata for import of %q", importPath)
			}

			depResult, ok := vdeps[id] // id may be ""
			if !ok {
				// Analogous to (*snapshot).missingPkgError
				// in the logic for regular type-checking,
				// but without a snapshot we can't provide
				// such detail, and anyway most analysis
				// failures aren't surfaced in the UI.
				return nil, fmt.Errorf("no required module provides package %q (id=%q)", importPath, id)
			}

			// (Duplicates logic from check.go.)
			if !source.IsValidImport(m.PkgPath, depResult.PkgPath) {
				return nil, fmt.Errorf("invalid use of internal package %s", importPath)
			}

			return loadFromExportData(depResult.PkgPath)
		}),
	}

	// Set Go dialect.
	if m.Module != nil && m.Module.GoVersion != "" {
		goVersion := "go" + m.Module.GoVersion
		// types.NewChecker panics if GoVersion is invalid.
		// An unparsable mod file should probably stop us
		// before we get here, but double check just in case.
		if goVersionRx.MatchString(goVersion) {
			typesinternal.SetGoVersion(cfg, goVersion)
		}
	}

	// We want to type check cgo code if go/types supports it.
	// We passed typecheckCgo to go/packages when we Loaded.
	// TODO(adonovan): do we actually need this??
	typesinternal.SetUsesCgo(cfg)

	check := types.NewChecker(cfg, fset, pkg.types, pkg.typesInfo)

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(pkg.files)

	// debugging (type errors are quite normal)
	if false {
		if pkg.typeErrors != nil {
			log.Printf("package %s has type errors: %v", pkg.types.Path(), pkg.typeErrors)
		}
	}

	// Emit the export data and compute the deep hash.
	export, err := gcimporter.IExportShallow(pkg.fset, pkg.types)
	if err != nil {
		// TODO(adonovan): in light of exporter bugs such as #57729,
		// consider using bug.Report here and retrying the IExportShallow
		// call here using an empty types.Package.
		log.Fatalf("internal error writing shallow export data: %v", err)
	}
	pkg.export = export
	hashExport(m.PkgPath, export)
	exportHasher.Sum(pkg.deepExportHash[:0])

	return pkg
}

// analysisPackage contains information about a package, including
// syntax trees, used transiently during its type-checking and analysis.
type analysisPackage struct {
	m              *source.Metadata
	fset           *token.FileSet // local to this package
	parsed         []*source.ParsedGoFile
	files          []*ast.File // same as parsed[i].File
	types          *types.Package
	compiles       bool // package is transitively free of list/parse/type errors
	factsDecoder   *facts.Decoder
	export         []byte      // encoding of types.Package
	deepExportHash source.Hash // reflexive transitive hash of export data
	typesInfo      *types.Info
	typeErrors     []types.Error
	typesSizes     types.Sizes
}

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type action struct {
	once  sync.Once
	a     *analysis.Analyzer
	pkg   *analysisPackage
	hdeps []*action                     // horizontal dependencies
	vdeps map[PackageID]*analyzeSummary // vertical dependencies

	// results of action.exec():
	result  interface{} // result of Run function, of type a.ResultType
	summary *actionSummary
	err     error
}

func (act *action) String() string {
	return fmt.Sprintf("%s@%s", act.a.Name, act.pkg.m.ID)
}

// execActions executes a set of action graph nodes in parallel.
func execActions(actions []*action) {
	var wg sync.WaitGroup
	for _, act := range actions {
		act := act
		wg.Add(1)
		go func() {
			defer wg.Done()
			act.once.Do(func() {
				execActions(act.hdeps) // analyze "horizontal" dependencies
				act.result, act.summary, act.err = act.exec()
				if act.err != nil {
					act.summary = &actionSummary{Err: act.err.Error()}
					// TODO(adonovan): suppress logging. But
					// shouldn't the root error's causal chain
					// include this information?
					if false { // debugging
						log.Printf("act.exec(%v) failed: %v", act, act.err)
					}
				}
			})
		}()
	}
	wg.Wait()
}

// exec defines the execution of a single action.
// It returns the (ephemeral) result of the analyzer's Run function,
// along with its (serializable) facts and diagnostics.
// Or it returns an error if the analyzer did not run to
// completion and deliver a valid result.
func (act *action) exec() (interface{}, *actionSummary, error) {
	analyzer := act.a
	pkg := act.pkg

	hasFacts := len(analyzer.FactTypes) > 0

	// Report an error if any action dependency (vertical or horizontal) failed.
	// To avoid long error messages describing chains of failure,
	// we return the dependencies' error' unadorned.
	if hasFacts {
		// TODO(adonovan): use deterministic order.
		for _, res := range act.vdeps {
			if vdep := res.Actions[analyzer.Name]; vdep.Err != "" {
				return nil, nil, errors.New(vdep.Err)
			}
		}
	}
	for _, dep := range act.hdeps {
		if dep.err != nil {
			return nil, nil, dep.err
		}
	}
	// Inv: all action dependencies succeeded.

	// Were there list/parse/type errors that might prevent analysis?
	if !pkg.compiles && !analyzer.RunDespiteErrors {
		return nil, nil, fmt.Errorf("skipping analysis %q because package %q does not compile", analyzer.Name, pkg.m.ID)
	}
	// Inv: package is well-formed enough to proceed with analysis.

	if false { // debugging
		log.Println("action.exec", act)
	}

	// Gather analysis Result values from horizontal dependencies.
	var inputs = make(map[*analysis.Analyzer]interface{})
	for _, dep := range act.hdeps {
		inputs[dep.a] = dep.result
	}

	// TODO(adonovan): opt: facts.Set works but it may be more
	// efficient to fork and tailor it to our precise needs.
	//
	// We've already sharded the fact encoding by action
	// so that it can be done in parallel (hoisting the
	// ImportMap call so that we build the map once per package).
	// We could eliminate locking.
	// We could also dovetail more closely with the export data
	// decoder to obtain a more compact representation of
	// packages and objects (e.g. its internal IDs, instead
	// of PkgPaths and objectpaths.)

	// Read and decode analysis facts for each imported package.
	factset, err := pkg.factsDecoder.Decode(func(imp *types.Package) ([]byte, error) {
		if !hasFacts {
			return nil, nil // analyzer doesn't use facts, so no vdeps
		}

		// Package.Imports() may contain a fake "C" package. Ignore it.
		if imp.Path() == "C" {
			return nil, nil
		}

		id, ok := pkg.m.DepsByPkgPath[PackagePath(imp.Path())]
		if !ok {
			// This may mean imp was synthesized by the type
			// checker because it failed to import it for any reason
			// (e.g. bug processing export data; metadata ignoring
			// a cycle-forming import).
			// In that case, the fake package's imp.Path
			// is set to the failed importPath (and thus
			// it may lack a "vendor/" prefix).
			//
			// For now, silently ignore it on the assumption
			// that the error is already reported elsewhere.
			// return nil, fmt.Errorf("missing metadata")
			return nil, nil
		}

		vdep, ok := act.vdeps[id]
		if !ok {
			return nil, bug.Errorf("internal error in %s: missing vdep for id=%s", pkg.types.Path(), id)
		}
		return vdep.Actions[analyzer.Name].Facts, nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("internal error decoding analysis facts: %w", err)
	}

	// TODO(adonovan): make Export*Fact panic rather than discarding
	// undeclared fact types, so that we discover bugs in analyzers.
	factFilter := make(map[reflect.Type]bool)
	for _, f := range analyzer.FactTypes {
		factFilter[reflect.TypeOf(f)] = true
	}

	// posToLocation converts from token.Pos to protocol form.
	// TODO(adonovan): improve error messages.
	posToLocation := func(start, end token.Pos) (protocol.Location, error) {
		tokFile := pkg.fset.File(start)
		for _, p := range pkg.parsed {
			if p.Tok == tokFile {
				if end == token.NoPos {
					end = start
				}
				return p.PosLocation(start, end)
			}
		}
		return protocol.Location{},
			bug.Errorf("internal error: token.Pos not within package")
	}

	// Now run the (pkg, analyzer) action.
	var diagnostics []gobDiagnostic
	pass := &analysis.Pass{
		Analyzer:   analyzer,
		Fset:       pkg.fset,
		Files:      pkg.files,
		Pkg:        pkg.types,
		TypesInfo:  pkg.typesInfo,
		TypesSizes: pkg.typesSizes,
		TypeErrors: pkg.typeErrors,
		ResultOf:   inputs,
		Report: func(d analysis.Diagnostic) {
			// Prefix the diagnostic category with the analyzer's name.
			if d.Category == "" {
				d.Category = analyzer.Name
			} else {
				d.Category = analyzer.Name + "." + d.Category
			}

			diagnostic, err := toGobDiagnostic(posToLocation, d)
			if err != nil {
				bug.Reportf("internal error converting diagnostic from analyzer %q: %v", analyzer.Name, err)
				return
			}
			diagnostics = append(diagnostics, diagnostic)
		},
		ImportObjectFact:  factset.ImportObjectFact,
		ExportObjectFact:  factset.ExportObjectFact,
		ImportPackageFact: factset.ImportPackageFact,
		ExportPackageFact: factset.ExportPackageFact,
		AllObjectFacts:    func() []analysis.ObjectFact { return factset.AllObjectFacts(factFilter) },
		AllPackageFacts:   func() []analysis.PackageFact { return factset.AllPackageFacts(factFilter) },
	}

	// Recover from panics (only) within the analyzer logic.
	// (Use an anonymous function to limit the recover scope.)
	var result interface{}
	func() {
		defer func() {
			if r := recover(); r != nil {
				// An Analyzer panicked, likely due to a bug.
				//
				// In general we want to discover and fix such panics quickly,
				// so we don't suppress them, but some bugs in third-party
				// analyzers cannot be quickly fixed, so we use an allowlist
				// to suppress panics.
				const strict = true
				if strict && bug.PanicOnBugs &&
					analyzer.Name != "buildir" { // see https://github.com/dominikh/go-tools/issues/1343
					// Uncomment this when debugging suspected failures
					// in the driver, not the analyzer.
					if false {
						debug.SetTraceback("all") // show all goroutines
					}
					panic(r)
				} else {
					// In production, suppress the panic and press on.
					err = fmt.Errorf("analysis %s for package %s panicked: %v", analyzer.Name, pass.Pkg.Path(), r)
				}
			}
		}()
		result, err = pass.Analyzer.Run(pass)
	}()
	if err != nil {
		return nil, nil, err
	}

	if got, want := reflect.TypeOf(result), pass.Analyzer.ResultType; got != want {
		return nil, nil, bug.Errorf(
			"internal error: on package %s, analyzer %s returned a result of type %v, but declared ResultType %v",
			pass.Pkg.Path(), pass.Analyzer, got, want)
	}

	// Disallow Export*Fact calls after Run.
	// (A panic means the Analyzer is abusing concurrency.)
	pass.ExportObjectFact = func(obj types.Object, fact analysis.Fact) {
		panic(fmt.Sprintf("%v: Pass.ExportObjectFact(%s, %T) called after Run", act, obj, fact))
	}
	pass.ExportPackageFact = func(fact analysis.Fact) {
		panic(fmt.Sprintf("%v: Pass.ExportPackageFact(%T) called after Run", act, fact))
	}

	factsdata := factset.Encode()
	return result, &actionSummary{
		Diagnostics: diagnostics,
		Facts:       factsdata,
		FactsHash:   source.HashOf(factsdata),
	}, nil
}

// requiredAnalyzers returns the transitive closure of required analyzers in preorder.
func requiredAnalyzers(analyzers []*analysis.Analyzer) []*analysis.Analyzer {
	var result []*analysis.Analyzer
	seen := make(map[*analysis.Analyzer]bool)
	var visitAll func([]*analysis.Analyzer)
	visitAll = func(analyzers []*analysis.Analyzer) {
		for _, a := range analyzers {
			if !seen[a] {
				seen[a] = true
				result = append(result, a)
				visitAll(a.Requires)
			}
		}
	}
	visitAll(analyzers)
	return result
}

func mustEncode(x interface{}) []byte {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(x); err != nil {
		log.Fatalf("internal error encoding %T: %v", x, err)
	}
	return buf.Bytes()
}

func mustDecode(data []byte, ptr interface{}) {
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(ptr); err != nil {
		log.Fatalf("internal error decoding %T: %v", ptr, err)
	}
}

// -- data types for serialization of analysis.Diagnostic and source.Diagnostic --

type gobDiagnostic struct {
	Location       protocol.Location
	Severity       protocol.DiagnosticSeverity
	Code           string
	CodeHref       string
	Source         string
	Message        string
	SuggestedFixes []gobSuggestedFix
	Related        []gobRelatedInformation
	Tags           []protocol.DiagnosticTag
}

type gobRelatedInformation struct {
	Location protocol.Location
	Message  string
}

type gobSuggestedFix struct {
	Message    string
	TextEdits  []gobTextEdit
	Command    *gobCommand
	ActionKind protocol.CodeActionKind
}

type gobCommand struct {
	Title     string
	Command   string
	Arguments []json.RawMessage
}

type gobTextEdit struct {
	Location protocol.Location
	NewText  []byte
}

// toGobDiagnostic converts an analysis.Diagnosic to a serializable gobDiagnostic,
// which requires expanding token.Pos positions into protocol.Location form.
func toGobDiagnostic(posToLocation func(start, end token.Pos) (protocol.Location, error), diag analysis.Diagnostic) (gobDiagnostic, error) {
	var fixes []gobSuggestedFix
	for _, fix := range diag.SuggestedFixes {
		var gobEdits []gobTextEdit
		for _, textEdit := range fix.TextEdits {
			loc, err := posToLocation(textEdit.Pos, textEdit.End)
			if err != nil {
				return gobDiagnostic{}, fmt.Errorf("in SuggestedFixes: %w", err)
			}
			gobEdits = append(gobEdits, gobTextEdit{
				Location: loc,
				NewText:  textEdit.NewText,
			})
		}
		fixes = append(fixes, gobSuggestedFix{
			Message:   fix.Message,
			TextEdits: gobEdits,
		})
	}

	var related []gobRelatedInformation
	for _, r := range diag.Related {
		loc, err := posToLocation(r.Pos, r.End)
		if err != nil {
			return gobDiagnostic{}, fmt.Errorf("in Related: %w", err)
		}
		related = append(related, gobRelatedInformation{
			Location: loc,
			Message:  r.Message,
		})
	}

	loc, err := posToLocation(diag.Pos, diag.End)
	if err != nil {
		return gobDiagnostic{}, err
	}

	return gobDiagnostic{
		Location: loc,
		// Severity for analysis diagnostics is dynamic, based on user
		// configuration per analyzer.
		// Code and CodeHref are unset for Analysis diagnostics,
		// TODO(rfindley): set Code fields if/when golang/go#57906 is accepted.
		Source:         diag.Category,
		Message:        diag.Message,
		SuggestedFixes: fixes,
		Related:        related,
		// Analysis diagnostics do not contain tags.
	}, nil
}
