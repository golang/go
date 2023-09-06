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
	urlpkg "net/url"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/filecache"
	"golang.org/x/tools/gopls/internal/lsp/frob"
	"golang.org/x/tools/gopls/internal/lsp/progress"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/facts"
	"golang.org/x/tools/internal/gcimporter"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

/*

   DESIGN

   An analysis request (Snapshot.Analyze) is for a set of Analyzers and
   PackageIDs. The result is the set of diagnostics for those
   packages. Each request constructs a transitively closed DAG of
   nodes, each representing a package, then works bottom up in
   parallel postorder calling runCached to ensure that each node's
   analysis summary is up to date. The summary contains the analysis
   diagnostics as well as the intermediate results required by the
   recursion, such as serialized types and facts.

   The entire DAG is ephemeral. Each node in the DAG records the set
   of analyzers to run: the complete set for the root packages, and
   the "facty" subset for dependencies. Each package is thus analyzed
   at most once. The entire DAG shares a single FileSet for parsing
   and importing.

   Each node is processed by runCached. It gets the source file
   content hashes for package p, and the summaries of its "vertical"
   dependencies (direct imports), and from them it computes a key
   representing the unit of work (parsing, type-checking, and
   analysis) that it has to do. The key is a cryptographic hash of the
   "recipe" for this step, including the Metadata, the file contents,
   the set of analyzers, and the type and fact information from the
   vertical dependencies.

   The key is sought in a machine-global persistent file-system based
   cache. If this gopls process, or another gopls process on the same
   machine, has already performed this analysis step, runCached will
   make a cache hit and load the serialized summary of the results. If
   not, it will have to proceed to run() to parse and type-check the
   package and then apply a set of analyzers to it. (The set of
   analyzers applied to a single package itself forms a graph of
   "actions", and it too is evaluated in parallel postorder; these
   dependency edges within the same package are called "horizontal".)
   Finally it writes a new cache entry. The entry contains serialized
   types (export data) and analysis facts.

   Each node in the DAG acts like a go/types importer mapping,
   providing a consistent view of packages and their objects: the
   mapping for a node is a superset of its dependencies' mappings.
   Every node has an associated *types.Package, initially nil. A
   package is populated during run (cache miss) by type-checking its
   syntax; but for a cache hit, the package is populated lazily, i.e.
   not until it later becomes necessary because it is imported
   directly or referenced by export data higher up in the DAG.

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
   just the direct imports. Type information for the entire transitive
   closure of imports is provided (lazily) by the DAG.

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
   whole application (though export data does record line numbers and
   offsets of types which may be perturbed by otherwise insignificant
   changes.)

   The summary must record whether a package is transitively
   error-free (whether it would compile) because many analyzers are
   not safe to run on packages with inconsistent types.

   For fact encoding, we use the same fact set as the unitchecker
   (vet) to record and serialize analysis facts. The fact
   serialization mechanism is analogous to "deep" export data.

*/

// TODO(adonovan):
// - Add a (white-box) test of pruning when a change doesn't affect export data.
// - Optimise pruning based on subset of packages mentioned in exportdata.
// - Better logging so that it is possible to deduce why an analyzer
//   is not being run--often due to very indirect failures.
//   Even if the ultimate consumer decides to ignore errors,
//   tests and other situations want to be assured of freedom from
//   errors, not just missing results. This should be recorded.
// - Split this into a subpackage, gopls/internal/lsp/cache/driver,
//   consisting of this file and three helpers from errors.go.
//   The (*snapshot).Analyze method would stay behind and make calls
//   to the driver package.
//   Steps:
//   - define a narrow driver.Snapshot interface with only these methods:
//        Metadata(PackageID) source.Metadata
//        ReadFile(Context, URI) (source.FileHandle, error)
//        View() *View // for Options
//   - share cache.{goVersionRx,parseGoImpl}

// AnalysisProgressTitle is the title of the progress report for ongoing
// analysis. It is sought by regression tests for the progress reporting
// feature.
const AnalysisProgressTitle = "Analyzing Dependencies"

// Analyze applies a set of analyzers to the package denoted by id,
// and returns their diagnostics for that package.
//
// The analyzers list must be duplicate free; order does not matter.
//
// Notifications of progress may be sent to the optional reporter.
//
// Precondition: all analyzers within the process have distinct names.
// (The names are relied on by the serialization logic.)
func (snapshot *snapshot) Analyze(ctx context.Context, pkgs map[PackageID]unit, analyzers []*source.Analyzer, reporter *progress.Tracker) ([]*source.Diagnostic, error) {
	start := time.Now() // for progress reporting

	var tagStr string // sorted comma-separated list of PackageIDs
	{
		// TODO(adonovan): replace with a generic map[S]any -> string
		// function in the tag package, and use maps.Keys + slices.Sort.
		keys := make([]string, 0, len(pkgs))
		for id := range pkgs {
			keys = append(keys, string(id))
		}
		sort.Strings(keys)
		tagStr = strings.Join(keys, ",")
	}
	ctx, done := event.Start(ctx, "snapshot.Analyze", tag.Package.Of(tagStr))
	defer done()

	// Filter and sort enabled root analyzers.
	// A disabled analyzer may still be run if required by another.
	toSrc := make(map[*analysis.Analyzer]*source.Analyzer)
	var enabled []*analysis.Analyzer // enabled subset + transitive requirements
	for _, a := range analyzers {
		if a.IsEnabled(snapshot.options) {
			toSrc[a.Analyzer] = a
			enabled = append(enabled, a.Analyzer)
		}
	}
	sort.Slice(enabled, func(i, j int) bool {
		return enabled[i].Name < enabled[j].Name
	})
	analyzers = nil // prevent accidental use

	// Register fact types of required analyzers.
	enabled = requiredAnalyzers(enabled)
	var facty []*analysis.Analyzer // facty subset of enabled + transitive requirements
	for _, a := range enabled {
		if len(a.FactTypes) > 0 {
			facty = append(facty, a)
			for _, f := range a.FactTypes {
				gob.Register(f) // <2us
			}
		}
	}
	facty = requiredAnalyzers(facty)

	// File set for this batch (entire graph) of analysis.
	fset := token.NewFileSet()

	// Starting from the root packages and following DepsByPkgPath,
	// build the DAG of packages we're going to analyze.
	//
	// Root nodes will run the enabled set of analyzers,
	// whereas dependencies will run only the facty set.
	// Because (by construction) enabled is a superset of facty,
	// we can analyze each node with exactly one set of analyzers.
	nodes := make(map[PackageID]*analysisNode)
	var leaves []*analysisNode // nodes with no unfinished successors
	var makeNode func(from *analysisNode, id PackageID) (*analysisNode, error)
	makeNode = func(from *analysisNode, id PackageID) (*analysisNode, error) {
		an, ok := nodes[id]
		if !ok {
			m := snapshot.Metadata(id)
			if m == nil {
				return nil, bug.Errorf("no metadata for %s", id)
			}

			// -- preorder --

			an = &analysisNode{
				fset:       fset,
				m:          m,
				analyzers:  facty, // all nodes run at least the facty analyzers
				allDeps:    make(map[PackagePath]*analysisNode),
				exportDeps: make(map[PackagePath]*analysisNode),
			}
			nodes[id] = an

			// -- recursion --

			// Build subgraphs for dependencies.
			an.succs = make(map[PackageID]*analysisNode, len(m.DepsByPkgPath))
			for _, depID := range m.DepsByPkgPath {
				dep, err := makeNode(an, depID)
				if err != nil {
					return nil, err
				}
				an.succs[depID] = dep

				// Compute the union of all dependencies.
				// (This step has quadratic complexity.)
				for pkgPath, node := range dep.allDeps {
					an.allDeps[pkgPath] = node
				}
			}

			// -- postorder --

			an.allDeps[m.PkgPath] = an // add self entry (reflexive transitive closure)

			// Add leaf nodes (no successors) directly to queue.
			if len(an.succs) == 0 {
				leaves = append(leaves, an)
			}

			// Load the contents of each compiled Go file through
			// the snapshot's cache. (These are all cache hits as
			// files are pre-loaded following packages.Load)
			an.files = make([]source.FileHandle, len(m.CompiledGoFiles))
			for i, uri := range m.CompiledGoFiles {
				fh, err := snapshot.ReadFile(ctx, uri)
				if err != nil {
					return nil, err
				}
				an.files[i] = fh
			}
		}
		// Add edge from predecessor.
		if from != nil {
			atomic.AddInt32(&from.unfinishedSuccs, 1) // TODO(adonovan): use generics
			an.preds = append(an.preds, from)
		}
		atomic.AddInt32(&an.unfinishedPreds, 1)
		return an, nil
	}

	// For root packages, we run the enabled set of analyzers.
	var roots []*analysisNode
	for id := range pkgs {
		root, err := makeNode(nil, id)
		if err != nil {
			return nil, err
		}
		root.analyzers = enabled
		roots = append(roots, root)
	}

	// Now that we have read all files,
	// we no longer need the snapshot.
	// (but options are needed for progress reporting)
	options := snapshot.options
	snapshot = nil

	// Progress reporting. If supported, gopls reports progress on analysis
	// passes that are taking a long time.
	maybeReport := func(completed int64) {}

	// Enable progress reporting if enabled by the user
	// and we have a capable reporter.
	if reporter != nil && reporter.SupportsWorkDoneProgress() && options.AnalysisProgressReporting {
		var reportAfter = options.ReportAnalysisProgressAfter // tests may set this to 0
		const reportEvery = 1 * time.Second

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		var (
			reportMu   sync.Mutex
			lastReport time.Time
			wd         *progress.WorkDone
		)
		defer func() {
			reportMu.Lock()
			defer reportMu.Unlock()

			if wd != nil {
				wd.End(ctx, "Done.") // ensure that the progress report exits
			}
		}()
		maybeReport = func(completed int64) {
			now := time.Now()
			if now.Sub(start) < reportAfter {
				return
			}

			reportMu.Lock()
			defer reportMu.Unlock()

			if wd == nil {
				wd = reporter.Start(ctx, AnalysisProgressTitle, "", nil, cancel)
			}

			if now.Sub(lastReport) > reportEvery {
				lastReport = now
				// Trailing space is intentional: some LSP clients strip newlines.
				msg := fmt.Sprintf(`Indexed %d/%d packages. (Set "analysisProgressReporting" to false to disable notifications.)`,
					completed, len(nodes))
				pct := 100 * float64(completed) / float64(len(nodes))
				wd.Report(ctx, msg, pct)
			}
		}
	}

	// Execute phase: run leaves first, adding
	// new nodes to the queue as they become leaves.
	var g errgroup.Group

	// Analysis is CPU-bound.
	//
	// Note: avoid g.SetLimit here: it makes g.Go stop accepting work, which
	// prevents workers from enqeuing, and thus finishing, and thus allowing the
	// group to make progress: deadlock.
	limiter := make(chan unit, runtime.GOMAXPROCS(0))
	var completed int64

	var enqueue func(*analysisNode)
	enqueue = func(an *analysisNode) {
		g.Go(func() error {
			limiter <- unit{}
			defer func() { <-limiter }()

			summary, err := an.runCached(ctx)
			if err != nil {
				return err // cancelled, or failed to produce a package
			}
			maybeReport(atomic.AddInt64(&completed, 1))
			an.summary = summary

			// Notify each waiting predecessor,
			// and enqueue it when it becomes a leaf.
			for _, pred := range an.preds {
				if atomic.AddInt32(&pred.unfinishedSuccs, -1) == 0 {
					enqueue(pred)
				}
			}

			// Notify each successor that we no longer need
			// its action summaries, which hold Result values.
			// After the last one, delete it, so that we
			// free up large results such as SSA.
			for _, succ := range an.succs {
				succ.decrefPreds()
			}
			return nil
		})
	}
	for _, leaf := range leaves {
		enqueue(leaf)
	}
	if err := g.Wait(); err != nil {
		return nil, err // cancelled, or failed to produce a package
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
	for _, root := range roots {
		for _, a := range enabled {
			// Skip analyzers that were added only to
			// fulfil requirements of the original set.
			srcAnalyzer, ok := toSrc[a]
			if !ok {
				// Although this 'skip' operation is logically sound,
				// it is nonetheless surprising that its absence should
				// cause #60909 since none of the analyzers added for
				// requirements (e.g. ctrlflow, inspect, buildssa)
				// is capable of reporting diagnostics.
				if summary := root.summary.Actions[a.Name]; summary != nil {
					if n := len(summary.Diagnostics); n > 0 {
						bug.Reportf("Internal error: got %d unexpected diagnostics from analyzer %s. This analyzer was added only to fulfil the requirements of the requested set of analyzers, and it is not expected that such analyzers report diagnostics. Please report this in issue #60909.", n, a)
					}
				}
				continue
			}

			// Inv: root.summary is the successful result of run (via runCached).
			summary, ok := root.summary.Actions[a.Name]
			if summary == nil {
				panic(fmt.Sprintf("analyzeSummary.Actions[%q] = (nil, %t); got %v (#60551)",
					a.Name, ok, root.summary.Actions))
			}
			if summary.Err != "" {
				continue // action failed
			}
			for _, gobDiag := range summary.Diagnostics {
				results = append(results, toSourceDiagnostic(srcAnalyzer, &gobDiag))
			}
		}
	}
	return results, nil
}

func (an *analysisNode) decrefPreds() {
	if atomic.AddInt32(&an.unfinishedPreds, -1) == 0 {
		an.summary.Actions = nil
	}
}

// An analysisNode is a node in a doubly-linked DAG isomorphic to the
// import graph. Each node represents a single package, and the DAG
// represents a batch of analysis work done at once using a single
// realm of token.Pos or types.Object values.
//
// A complete DAG is created anew for each batch of analysis;
// subgraphs are not reused over time. Each node's *types.Package
// field is initially nil and is populated on demand, either from
// type-checking syntax trees (typeCheck) or from importing export
// data (_import). When this occurs, the typesOnce event becomes
// "done".
//
// Each node's allDeps map is a "view" of all its dependencies keyed by
// package path, which defines the types.Importer mapping used when
// populating the node's types.Package. Different nodes have different
// views (e.g. due to variants), but two nodes that are related by
// graph ordering have views that are consistent in their overlap.
// exportDeps is the subset actually referenced by export data;
// this is the set for which we attempt to decode facts.
//
// Each node's run method is called in parallel postorder. On success,
// its summary field is populated, either from the cache (hit), or by
// type-checking and analyzing syntax (miss).
type analysisNode struct {
	fset            *token.FileSet              // file set shared by entire batch (DAG)
	m               *source.Metadata            // metadata for this package
	files           []source.FileHandle         // contents of CompiledGoFiles
	analyzers       []*analysis.Analyzer        // set of analyzers to run
	preds           []*analysisNode             // graph edges:
	succs           map[PackageID]*analysisNode //   (preds -> self -> succs)
	unfinishedSuccs int32
	unfinishedPreds int32                         // effectively a summary.Actions refcount
	allDeps         map[PackagePath]*analysisNode // all dependencies including self
	exportDeps      map[PackagePath]*analysisNode // subset of allDeps ref'd by export data (+self)
	summary         *analyzeSummary               // serializable result of analyzing this package

	typesOnce sync.Once      // guards lazy population of types and typesErr fields
	types     *types.Package // type information lazily imported from summary
	typesErr  error          // an error producing type information
}

func (an *analysisNode) String() string { return string(an.m.ID) }

// _import imports this node's types.Package from export data, if not already done.
// Precondition: analysis was a success.
// Postcondition: an.types and an.exportDeps are populated.
func (an *analysisNode) _import() (*types.Package, error) {
	an.typesOnce.Do(func() {
		if an.m.PkgPath == "unsafe" {
			an.types = types.Unsafe
			return
		}

		an.types = types.NewPackage(string(an.m.PkgPath), string(an.m.Name))

		// getPackages recursively imports each dependency
		// referenced by the export data, in parallel.
		getPackages := func(items []gcimporter.GetPackagesItem) error {
			var g errgroup.Group
			for i, item := range items {
				path := PackagePath(item.Path)
				dep, ok := an.allDeps[path]
				if !ok {
					// This early return bypasses Wait; that's ok.
					return fmt.Errorf("%s: unknown dependency %q", an.m, path)
				}
				an.exportDeps[path] = dep // record, for later fact decoding
				if dep == an {
					if an.typesErr != nil {
						return an.typesErr
					} else {
						items[i].Pkg = an.types
					}
				} else {
					i := i
					g.Go(func() error {
						depPkg, err := dep._import()
						if err == nil {
							items[i].Pkg = depPkg
						}
						return err
					})
				}
			}
			return g.Wait()
		}
		pkg, err := gcimporter.IImportShallow(an.fset, getPackages, an.summary.Export, string(an.m.PkgPath), bug.Reportf)
		if err != nil {
			an.typesErr = bug.Errorf("%s: invalid export data: %v", an.m, err)
			an.types = nil
		} else if pkg != an.types {
			log.Fatalf("%s: inconsistent packages", an.m)
		}
	})
	return an.types, an.typesErr
}

// analyzeSummary is a gob-serializable summary of successfully
// applying a list of analyzers to a package.
type analyzeSummary struct {
	Export         []byte      // encoded types of package
	DeepExportHash source.Hash // hash of reflexive transitive closure of export data
	Compiles       bool        // transitively free of list/parse/type errors
	Actions        actionsMap  // map from analyzer name to analysis results (*actionSummary)
}

// actionsMap defines a stable Gob encoding for a map.
// TODO(adonovan): generalize and move to a library when we can use generics.
type actionsMap map[string]*actionSummary

var (
	_ gob.GobEncoder = (actionsMap)(nil)
	_ gob.GobDecoder = (*actionsMap)(nil)
)

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

// runCached applies a list of analyzers (plus any others
// transitively required by them) to a package.  It succeeds as long
// as it could produce a types.Package, even if there were direct or
// indirect list/parse/type errors, and even if all the analysis
// actions failed. It usually fails only if the package was unknown,
// a file was missing, or the operation was cancelled.
//
// Postcondition: runCached must not continue to use the snapshot
// (in background goroutines) after it has returned; see memoize.RefCounted.
func (an *analysisNode) runCached(ctx context.Context) (*analyzeSummary, error) {
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
	key := an.cacheKey()

	// Access the cache.
	var summary *analyzeSummary
	const cacheKind = "analysis"
	if data, err := filecache.Get(cacheKind, key); err == nil {
		// cache hit
		analyzeSummaryCodec.Decode(data, &summary)
	} else if err != filecache.ErrNotFound {
		return nil, bug.Errorf("internal error reading shared cache: %v", err)
	} else {
		// Cache miss: do the work.
		var err error
		summary, err = an.run(ctx)
		if err != nil {
			return nil, err
		}

		atomic.AddInt32(&an.unfinishedPreds, +1) // incref
		go func() {
			defer an.decrefPreds() //decref

			cacheLimit <- unit{}            // acquire token
			defer func() { <-cacheLimit }() // release token

			data := analyzeSummaryCodec.Encode(summary)
			if false {
				log.Printf("Set key=%d value=%d id=%s\n", len(key), len(data), an.m.ID)
			}
			if err := filecache.Set(cacheKind, key, data); err != nil {
				event.Error(ctx, "internal error updating analysis shared cache", err)
			}
		}()
	}

	return summary, nil
}

// cacheLimit reduces parallelism of cache updates.
// We allow more than typical GOMAXPROCS as it's a mix of CPU and I/O.
var cacheLimit = make(chan unit, 32)

// analysisCacheKey returns a cache key that is a cryptographic digest
// of the all the values that might affect type checking and analysis:
// the analyzer names, package metadata, names and contents of
// compiled Go files, and vdeps (successor) information
// (export data and facts).
func (an *analysisNode) cacheKey() [sha256.Size]byte {
	hasher := sha256.New()

	// In principle, a key must be the hash of an
	// unambiguous encoding of all the relevant data.
	// If it's ambiguous, we risk collisions.

	// analyzers
	fmt.Fprintf(hasher, "analyzers: %d\n", len(an.analyzers))
	for _, a := range an.analyzers {
		fmt.Fprintln(hasher, a.Name)
	}

	// package metadata
	m := an.m
	fmt.Fprintf(hasher, "package: %s %s %s\n", m.ID, m.Name, m.PkgPath)
	// We can ignore m.DepsBy{Pkg,Import}Path: although the logic
	// uses those fields, we account for them by hashing vdeps.

	// type sizes
	wordSize := an.m.TypesSizes.Sizeof(types.Typ[types.Int])
	maxAlign := an.m.TypesSizes.Alignof(types.NewPointer(types.Typ[types.Int64]))
	fmt.Fprintf(hasher, "sizes: %d %d\n", wordSize, maxAlign)

	// metadata errors: used for 'compiles' field
	fmt.Fprintf(hasher, "errors: %d", len(m.Errors))

	// module Go version
	if m.Module != nil && m.Module.GoVersion != "" {
		fmt.Fprintf(hasher, "go %s\n", m.Module.GoVersion)
	}

	// file names and contents
	fmt.Fprintf(hasher, "files: %d\n", len(an.files))
	for _, fh := range an.files {
		fmt.Fprintln(hasher, fh.FileIdentity())
	}

	// vdeps, in PackageID order
	depIDs := make([]string, 0, len(an.succs))
	for depID := range an.succs {
		depIDs = append(depIDs, string(depID))
	}
	sort.Strings(depIDs) // TODO(adonovan): avoid conversions by using slices.Sort[PackageID]
	for _, depID := range depIDs {
		vdep := an.succs[PackageID(depID)]
		fmt.Fprintf(hasher, "dep: %s\n", vdep.m.PkgPath)
		fmt.Fprintf(hasher, "export: %s\n", vdep.summary.DeepExportHash)

		// action results: errors and facts
		actions := vdep.summary.Actions
		names := make([]string, 0, len(actions))
		for name := range actions {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			summary := actions[name]
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

// run implements the cache-miss case.
// This function does not access the snapshot.
//
// Postcondition: on success, the analyzeSummary.Actions
// key set is {a.Name for a in analyzers}.
func (an *analysisNode) run(ctx context.Context) (*analyzeSummary, error) {
	// Parse only the "compiled" Go files.
	// Do the computation in parallel.
	parsed := make([]*source.ParsedGoFile, len(an.files))
	{
		var group errgroup.Group
		group.SetLimit(4) // not too much: run itself is already called in parallel
		for i, fh := range an.files {
			i, fh := i, fh
			group.Go(func() error {
				// Call parseGoImpl directly, not the caching wrapper,
				// as cached ASTs require the global FileSet.
				// ast.Object resolution is unfortunately an implied part of the
				// go/analysis contract.
				pgf, err := parseGoImpl(ctx, an.fset, fh, source.ParseFull&^source.SkipObjectResolution, false)
				parsed[i] = pgf
				return err
			})
		}
		if err := group.Wait(); err != nil {
			return nil, err // cancelled, or catastrophic error (e.g. missing file)
		}
	}

	// Type-check the package syntax.
	pkg := an.typeCheck(parsed)

	// Publish the completed package.
	an.typesOnce.Do(func() { an.types = pkg.types })
	if an.types != pkg.types {
		log.Fatalf("typesOnce prematurely done")
	}

	// Compute the union of exportDeps across our direct imports.
	// This is the set that will be needed by the fact decoder.
	allExportDeps := make(map[PackagePath]*analysisNode)
	for _, succ := range an.succs {
		for k, v := range succ.exportDeps {
			allExportDeps[k] = v
		}
	}

	// The fact decoder needs a means to look up a Package by path.
	pkg.factsDecoder = facts.NewDecoderFunc(pkg.types, func(path string) *types.Package {
		// Note: Decode is called concurrently, and thus so is this function.

		// Does the fact relate to a package referenced by export data?
		if dep, ok := allExportDeps[PackagePath(path)]; ok {
			dep.typesOnce.Do(func() { log.Fatal("dep.types not populated") })
			if dep.typesErr == nil {
				return dep.types
			}
			return nil
		}

		// If the fact relates to a dependency not referenced
		// by export data, it is safe to ignore it.
		// (In that case dep.types exists but may be unpopulated
		// or in the process of being populated from export data.)
		if an.allDeps[PackagePath(path)] == nil {
			log.Fatalf("fact package %q is not a dependency", path)
		}
		return nil
	})

	// Poll cancellation state.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

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
			act = &action{a: a, pkg: pkg, vdeps: an.succs, hdeps: hdeps}
			actions[a] = act
		}
		return act
	}

	// Build actions for initial package.
	var roots []*action
	for _, a := range an.analyzers {
		roots = append(roots, mkAction(a))
	}

	// Execute the graph in parallel.
	execActions(roots)
	// Inv: each root's summary is set (whether success or error).

	// Don't return (or cache) the result in case of cancellation.
	if err := ctx.Err(); err != nil {
		return nil, err // cancelled
	}

	// Return summaries only for the requested actions.
	summaries := make(map[string]*actionSummary)
	for _, root := range roots {
		if root.summary == nil {
			panic("root has nil action.summary (#60551)")
		}
		summaries[root.a.Name] = root.summary
	}

	return &analyzeSummary{
		Export:         pkg.export,
		DeepExportHash: pkg.deepExportHash,
		Compiles:       pkg.compiles,
		Actions:        summaries,
	}, nil
}

// Postcondition: analysisPackage.types and an.exportDeps are populated.
func (an *analysisNode) typeCheck(parsed []*source.ParsedGoFile) *analysisPackage {
	m := an.m

	if false { // debugging
		log.Println("typeCheck", m.ID)
	}

	pkg := &analysisPackage{
		m:        m,
		fset:     an.fset,
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

	// Unsafe has no syntax.
	if m.PkgPath == "unsafe" {
		pkg.types = types.Unsafe
		return pkg
	}

	for i, p := range parsed {
		pkg.files[i] = p.File
		if p.ParseErr != nil {
			pkg.compiles = false // parse error
		}
	}

	for _, vdep := range an.succs {
		if !vdep.summary.Compiles {
			pkg.compiles = false // transitive error
		}
	}

	cfg := &types.Config{
		Sizes: m.TypesSizes,
		Error: func(e error) {
			pkg.compiles = false // type error

			// Suppress type errors in files with parse errors
			// as parser recovery can be quite lossy (#59888).
			typeError := e.(types.Error)
			for _, p := range parsed {
				if p.ParseErr != nil && source.NodeContains(p.File, typeError.Pos) {
					return
				}
			}
			pkg.typeErrors = append(pkg.typeErrors, typeError)
		},
		Importer: importerFunc(func(importPath string) (*types.Package, error) {
			// Beware that returning an error from this function
			// will cause the type checker to synthesize a fake
			// package whose Path is importPath, potentially
			// losing a vendor/ prefix. If type-checking errors
			// are swallowed, these packages may be confusing.

			// Map ImportPath to ID.
			id, ok := m.DepsByImpPath[ImportPath(importPath)]
			if !ok {
				// The import syntax is inconsistent with the metadata.
				// This could be because the import declaration was
				// incomplete and the metadata only includes complete
				// imports; or because the metadata ignores import
				// edges that would lead to cycles in the graph.
				return nil, fmt.Errorf("missing metadata for import of %q", importPath)
			}

			// Map ID to node. (id may be "")
			dep := an.succs[id]
			if dep == nil {
				// Analogous to (*snapshot).missingPkgError
				// in the logic for regular type-checking,
				// but without a snapshot we can't provide
				// such detail, and anyway most analysis
				// failures aren't surfaced in the UI.
				return nil, fmt.Errorf("no required module provides analysis package %q (id=%q)", importPath, id)
			}

			// (Duplicates logic from check.go.)
			if !source.IsValidImport(an.m.PkgPath, dep.m.PkgPath) {
				return nil, fmt.Errorf("invalid use of internal package %s", importPath)
			}

			return dep._import()
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

	check := types.NewChecker(cfg, pkg.fset, pkg.types, pkg.typesInfo)

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(pkg.files)

	// debugging (type errors are quite normal)
	if false {
		if pkg.typeErrors != nil {
			log.Printf("package %s has type errors: %v", pkg.types.Path(), pkg.typeErrors)
		}
	}

	// Emit the export data and compute the recursive hash.
	export, err := gcimporter.IExportShallow(pkg.fset, pkg.types, bug.Reportf)
	if err != nil {
		// TODO(adonovan): in light of exporter bugs such as #57729,
		// consider using bug.Report here and retrying the IExportShallow
		// call here using an empty types.Package.
		log.Fatalf("internal error writing shallow export data: %v", err)
	}
	pkg.export = export

	// Compute a recursive hash to account for the export data of
	// this package and each dependency referenced by it.
	// Also, populate exportDeps.
	hash := sha256.New()
	fmt.Fprintf(hash, "%s %d\n", m.PkgPath, len(export))
	hash.Write(export)
	paths, err := readShallowManifest(export)
	if err != nil {
		log.Fatalf("internal error: bad export data: %v", err)
	}
	for _, path := range paths {
		dep, ok := an.allDeps[PackagePath(path)]
		if !ok {
			log.Fatalf("%s: missing dependency: %q", an, path)
		}
		fmt.Fprintf(hash, "%s %s\n", dep.m.PkgPath, dep.summary.DeepExportHash)
		an.exportDeps[PackagePath(path)] = dep
	}
	an.exportDeps[m.PkgPath] = an // self
	hash.Sum(pkg.deepExportHash[:0])

	return pkg
}

// readShallowManifest returns the manifest of packages referenced by
// a shallow export data file for a package (excluding the package itself).
// TODO(adonovan): add a test.
func readShallowManifest(export []byte) ([]PackagePath, error) {
	const selfPath = "<self>" // dummy path
	var paths []PackagePath
	getPackages := func(items []gcimporter.GetPackagesItem) error {
		paths = []PackagePath{} // non-nil
		for _, item := range items {
			if item.Path != selfPath {
				paths = append(paths, PackagePath(item.Path))
			}
		}
		return errors.New("stop") // terminate importer
	}
	_, err := gcimporter.IImportShallow(token.NewFileSet(), getPackages, export, selfPath, bug.Reportf)
	if paths == nil {
		if err != nil {
			return nil, err // failed before getPackages callback
		}
		return nil, bug.Errorf("internal error: IImportShallow did not call getPackages")
	}
	return paths, nil // success
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
	hdeps []*action                   // horizontal dependencies
	vdeps map[PackageID]*analysisNode // vertical dependencies

	// results of action.exec():
	result  interface{} // result of Run function, of type a.ResultType
	summary *actionSummary
	err     error
}

func (act *action) String() string {
	return fmt.Sprintf("%s@%s", act.a.Name, act.pkg.m.ID)
}

// execActions executes a set of action graph nodes in parallel.
// Postcondition: each action.summary is set, even in case of error.
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
			if act.summary == nil {
				panic("nil action.summary (#60551)")
			}
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
		for _, vdep := range act.vdeps {
			if summ := vdep.summary.Actions[analyzer.Name]; summ.Err != "" {
				return nil, nil, errors.New(summ.Err)
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
	inputs := make(map[*analysis.Analyzer]interface{})
	for _, dep := range act.hdeps {
		inputs[dep.a] = dep.result
	}

	// TODO(adonovan): opt: facts.Set works but it may be more
	// efficient to fork and tailor it to our precise needs.
	//
	// We've already sharded the fact encoding by action
	// so that it can be done in parallel.
	// We could eliminate locking.
	// We could also dovetail more closely with the export data
	// decoder to obtain a more compact representation of
	// packages and objects (e.g. its internal IDs, instead
	// of PkgPaths and objectpaths.)
	// More importantly, we should avoid re-export of
	// facts that related to objects that are discarded
	// by "deep" export data. Better still, use a "shallow" approach.

	// Read and decode analysis facts for each direct import.
	factset, err := pkg.factsDecoder.Decode(true, func(pkgPath string) ([]byte, error) {
		if !hasFacts {
			return nil, nil // analyzer doesn't use facts, so no vdeps
		}

		// Package.Imports() may contain a fake "C" package. Ignore it.
		if pkgPath == "C" {
			return nil, nil
		}

		id, ok := pkg.m.DepsByPkgPath[PackagePath(pkgPath)]
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

		vdep := act.vdeps[id]
		if vdep == nil {
			return nil, bug.Errorf("internal error in %s: missing vdep for id=%s", pkg.types.Path(), id)
		}

		return vdep.summary.Actions[analyzer.Name].Facts, nil
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
			diagnostic, err := toGobDiagnostic(posToLocation, analyzer, d)
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
		start := time.Now()
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

			// Accumulate running time for each checker.
			analyzerRunTimesMu.Lock()
			analyzerRunTimes[analyzer] += time.Since(start)
			analyzerRunTimesMu.Unlock()
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

	factsdata := factset.Encode(true)
	return result, &actionSummary{
		Diagnostics: diagnostics,
		Facts:       factsdata,
		FactsHash:   source.HashOf(factsdata),
	}, nil
}

var (
	analyzerRunTimesMu sync.Mutex
	analyzerRunTimes   = make(map[*analysis.Analyzer]time.Duration)
)

type LabelDuration struct {
	Label    string
	Duration time.Duration
}

// AnalyzerTimes returns the accumulated time spent in each Analyzer's
// Run function since process start, in descending order.
func AnalyzerRunTimes() []LabelDuration {
	analyzerRunTimesMu.Lock()
	defer analyzerRunTimesMu.Unlock()

	slice := make([]LabelDuration, 0, len(analyzerRunTimes))
	for a, t := range analyzerRunTimes {
		slice = append(slice, LabelDuration{Label: a.Name, Duration: t})
	}
	sort.Slice(slice, func(i, j int) bool {
		return slice[i].Duration > slice[j].Duration
	})
	return slice
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

var analyzeSummaryCodec = frob.CodecFor[*analyzeSummary]()

// -- data types for serialization of analysis.Diagnostic and source.Diagnostic --

// (The name says gob but we use frob.)
var diagnosticsCodec = frob.CodecFor[[]gobDiagnostic]()

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
func toGobDiagnostic(posToLocation func(start, end token.Pos) (protocol.Location, error), a *analysis.Analyzer, diag analysis.Diagnostic) (gobDiagnostic, error) {
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

	// The Code column of VSCode's Problems table renders this
	// information as "Source(Code)" where code is a link to CodeHref.
	// (The code field must be nonempty for anything to appear.)
	diagURL := effectiveURL(a, diag)
	code := "default"
	if diag.Category != "" {
		code = diag.Category
	}

	return gobDiagnostic{
		Location: loc,
		// Severity for analysis diagnostics is dynamic,
		// based on user configuration per analyzer.
		Code:           code,
		CodeHref:       diagURL,
		Source:         a.Name,
		Message:        diag.Message,
		SuggestedFixes: fixes,
		Related:        related,
		// Analysis diagnostics do not contain tags.
	}, nil
}

// effectiveURL computes the effective URL of diag,
// using the algorithm specified at Diagnostic.URL.
func effectiveURL(a *analysis.Analyzer, diag analysis.Diagnostic) string {
	u := diag.URL
	if u == "" && diag.Category != "" {
		u = "#" + diag.Category
	}
	if base, err := urlpkg.Parse(a.URL); err == nil {
		if rel, err := urlpkg.Parse(u); err == nil {
			u = base.ResolveReference(rel).String()
		}
	}
	return u
}
