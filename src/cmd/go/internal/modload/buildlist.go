// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/mvs"
	"cmd/go/internal/par"
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// capVersionSlice returns s with its cap reduced to its length.
func capVersionSlice(s []module.Version) []module.Version {
	return s[:len(s):len(s)]
}

// A Requirements represents a logically-immutable set of root module requirements.
type Requirements struct {
	// depth is the depth at which the requirement graph is computed.
	//
	// If eager, the graph includes all transitive requirements regardless of depth.
	//
	// If lazy, the graph includes only the root modules, the explicit
	// requirements of those root modules, and the transitive requirements of only
	// the *non-lazy* root modules.
	depth modDepth

	// rootModules is the set of module versions explicitly required by the main
	// module, sorted and capped to length. It may contain duplicates, and may
	// contain multiple versions for a given module path.
	rootModules    []module.Version
	maxRootVersion map[string]string

	// direct is the set of module paths for which we believe the module provides
	// a package directly imported by a package or test in the main module.
	//
	// The "direct" map controls which modules are annotated with "// indirect"
	// comments in the go.mod file, and may impact which modules are listed as
	// explicit roots (vs. indirect-only dependencies). However, it should not
	// have a semantic effect on the build list overall.
	//
	// The initial direct map is populated from the existing "// indirect"
	// comments (or lack thereof) in the go.mod file. It is updated by the
	// package loader: dependencies may be promoted to direct if new
	// direct imports are observed, and may be demoted to indirect during
	// 'go mod tidy' or 'go mod vendor'.
	//
	// The direct map is keyed by module paths, not module versions. When a
	// module's selected version changes, we assume that it remains direct if the
	// previous version was a direct dependency. That assumption might not hold in
	// rare cases (such as if a dependency splits out a nested module, or merges a
	// nested module back into a parent module).
	direct map[string]bool

	graphOnce sync.Once    // guards writes to (but not reads from) graph
	graph     atomic.Value // cachedGraph
}

// A cachedGraph is a non-nil *ModuleGraph, together with any error discovered
// while loading that graph.
type cachedGraph struct {
	mg  *ModuleGraph
	err error // If err is non-nil, mg may be incomplete (but must still be non-nil).
}

// requirements is the requirement graph for the main module.
//
// It is always non-nil if the main module's go.mod file has been loaded.
//
// This variable should only be read from the loadModFile function, and should
// only be written in the loadModFile and commitRequirements functions.
// All other functions that need or produce a *Requirements should
// accept and/or return an explicit parameter.
var requirements *Requirements

// newRequirements returns a new requirement set with the given root modules.
// The dependencies of the roots will be loaded lazily at the first call to the
// Graph method.
//
// The rootModules slice must be sorted according to module.Sort.
// The caller must not modify the rootModules slice or direct map after passing
// them to newRequirements.
//
// If vendoring is in effect, the caller must invoke initVendor on the returned
// *Requirements before any other method.
func newRequirements(depth modDepth, rootModules []module.Version, direct map[string]bool) *Requirements {
	for i, m := range rootModules {
		if m == Target {
			panic(fmt.Sprintf("newRequirements called with untrimmed build list: rootModules[%v] is Target", i))
		}
		if m.Path == "" || m.Version == "" {
			panic(fmt.Sprintf("bad requirement: rootModules[%v] = %v", i, m))
		}
		if i > 0 {
			prev := rootModules[i-1]
			if prev.Path > m.Path || (prev.Path == m.Path && semver.Compare(prev.Version, m.Version) > 0) {
				panic(fmt.Sprintf("newRequirements called with unsorted roots: %v", rootModules))
			}
		}
	}

	rs := &Requirements{
		depth:          depth,
		rootModules:    capVersionSlice(rootModules),
		maxRootVersion: make(map[string]string, len(rootModules)),
		direct:         direct,
	}

	for _, m := range rootModules {
		if v, ok := rs.maxRootVersion[m.Path]; ok && cmpVersion(v, m.Version) >= 0 {
			continue
		}
		rs.maxRootVersion[m.Path] = m.Version
	}
	return rs
}

// initVendor initializes rs.graph from the given list of vendored module
// dependencies, overriding the graph that would normally be loaded from module
// requirements.
func (rs *Requirements) initVendor(vendorList []module.Version) {
	rs.graphOnce.Do(func() {
		mg := &ModuleGraph{
			g: mvs.NewGraph(cmpVersion, []module.Version{Target}),
		}

		if rs.depth == lazy {
			// The roots of a lazy module should already include every module in the
			// vendor list, because the vendored modules are the same as those
			// maintained as roots by the lazy loading “import invariant”.
			if go117LazyTODO {
				// Double-check here that that invariant holds.
			}

			// So we can just treat the rest of the module graph as effectively
			// “pruned out”, like a more aggressive version of lazy loading:
			// the root requirements *are* the complete module graph.
			mg.g.Require(Target, rs.rootModules)
		} else {
			// The transitive requirements of the main module are not in general available
			// from the vendor directory, and we don't actually know how we got from
			// the roots to the final build list.
			//
			// Instead, we'll inject a fake "vendor/modules.txt" module that provides
			// those transitive dependencies, and mark it as a dependency of the main
			// module. That allows us to elide the actual structure of the module
			// graph, but still distinguishes between direct and indirect
			// dependencies.
			vendorMod := module.Version{Path: "vendor/modules.txt", Version: ""}
			mg.g.Require(Target, append(rs.rootModules, vendorMod))
			mg.g.Require(vendorMod, vendorList)
		}

		rs.graph.Store(cachedGraph{mg, nil})
	})
}

// rootSelected returns the version of the root dependency with the given module
// path, or the zero module.Version and ok=false if the module is not a root
// dependency.
func (rs *Requirements) rootSelected(path string) (version string, ok bool) {
	if path == Target.Path {
		return Target.Version, true
	}
	if v, ok := rs.maxRootVersion[path]; ok {
		return v, true
	}
	return "", false
}

// Graph returns the graph of module requirements loaded from the current
// root modules (as reported by RootModules).
//
// Graph always makes a best effort to load the requirement graph despite any
// errors, and always returns a non-nil *ModuleGraph.
//
// If the requirements of any relevant module fail to load, Graph also
// returns a non-nil error of type *mvs.BuildListError.
func (rs *Requirements) Graph(ctx context.Context) (*ModuleGraph, error) {
	rs.graphOnce.Do(func() {
		mg, mgErr := readModGraph(ctx, rs.depth, rs.rootModules)
		rs.graph.Store(cachedGraph{mg, mgErr})
	})
	cached := rs.graph.Load().(cachedGraph)
	return cached.mg, cached.err
}

// IsDirect returns whether the given module provides a package directly
// imported by a package or test in the main module.
func (rs *Requirements) IsDirect(path string) bool {
	return rs.direct[path]
}

// A ModuleGraph represents the complete graph of module dependencies
// of a main module.
//
// If the main module is lazily loaded, the graph does not include
// transitive dependencies of non-root (implicit) dependencies.
type ModuleGraph struct {
	g         *mvs.Graph
	loadCache par.Cache // module.Version → summaryError

	buildListOnce sync.Once
	buildList     []module.Version
}

// A summaryError is either a non-nil modFileSummary or a non-nil error
// encountered while reading or parsing that summary.
type summaryError struct {
	summary *modFileSummary
	err     error
}

// readModGraph reads and returns the module dependency graph starting at the
// given roots.
//
// Unlike LoadModGraph, readModGraph does not attempt to diagnose or update
// inconsistent roots.
func readModGraph(ctx context.Context, depth modDepth, roots []module.Version) (*ModuleGraph, error) {
	var (
		mu       sync.Mutex // guards mg.g and hasError during loading
		hasError bool
		mg       = &ModuleGraph{
			g: mvs.NewGraph(cmpVersion, []module.Version{Target}),
		}
	)
	mg.g.Require(Target, roots)

	var (
		loadQueue    = par.NewQueue(runtime.GOMAXPROCS(0))
		loadingEager sync.Map // module.Version → nil; the set of modules that have been or are being loaded via eager roots
	)

	// loadOne synchronously loads the explicit requirements for module m.
	// It does not load the transitive requirements of m even if the go version in
	// m's go.mod file indicates eager loading.
	loadOne := func(m module.Version) (*modFileSummary, error) {
		cached := mg.loadCache.Do(m, func() interface{} {
			summary, err := goModSummary(m)

			mu.Lock()
			if err == nil {
				mg.g.Require(m, summary.require)
			} else {
				hasError = true
			}
			mu.Unlock()

			return summaryError{summary, err}
		}).(summaryError)

		return cached.summary, cached.err
	}

	var enqueue func(m module.Version, depth modDepth)
	enqueue = func(m module.Version, depth modDepth) {
		if m.Version == "none" {
			return
		}

		if depth == eager {
			if _, dup := loadingEager.LoadOrStore(m, nil); dup {
				// m has already been enqueued for loading. Since eager loading may
				// follow cycles in the the requirement graph, we need to return early
				// to avoid making the load queue infinitely long.
				return
			}
		}

		loadQueue.Add(func() {
			summary, err := loadOne(m)
			if err != nil {
				return // findError will report the error later.
			}

			// If the version in m's go.mod file implies eager loading, then we cannot
			// assume that the explicit requirements of m (added by loadOne) are
			// sufficient to build the packages it contains. We must load its full
			// transitive dependency graph to be sure that we see all relevant
			// dependencies.
			if depth == eager || summary.depth() == eager {
				for _, r := range summary.require {
					enqueue(r, eager)
				}
			}
		})
	}

	for _, m := range roots {
		enqueue(m, depth)
	}
	<-loadQueue.Idle()

	if hasError {
		return mg, mg.findError()
	}
	return mg, nil
}

// RequiredBy returns the dependencies required by module m in the graph,
// or ok=false if module m's dependencies are not relevant (such as if they
// are pruned out by lazy loading).
//
// The caller must not modify the returned slice, but may safely append to it
// and may rely on it not to be modified.
func (mg *ModuleGraph) RequiredBy(m module.Version) (reqs []module.Version, ok bool) {
	return mg.g.RequiredBy(m)
}

// Selected returns the selected version of the module with the given path.
//
// If no version is selected, Selected returns version "none".
func (mg *ModuleGraph) Selected(path string) (version string) {
	return mg.g.Selected(path)
}

// WalkBreadthFirst invokes f once, in breadth-first order, for each module
// version other than "none" that appears in the graph, regardless of whether
// that version is selected.
func (mg *ModuleGraph) WalkBreadthFirst(f func(m module.Version)) {
	mg.g.WalkBreadthFirst(f)
}

// BuildList returns the selected versions of all modules present in the graph,
// beginning with Target.
//
// The order of the remaining elements in the list is deterministic
// but arbitrary.
//
// The caller must not modify the returned list, but may safely append to it
// and may rely on it not to be modified.
func (mg *ModuleGraph) BuildList() []module.Version {
	mg.buildListOnce.Do(func() {
		mg.buildList = capVersionSlice(mg.g.BuildList())
	})
	return mg.buildList
}

func (mg *ModuleGraph) findError() error {
	errStack := mg.g.FindPath(func(m module.Version) bool {
		cached := mg.loadCache.Get(m)
		return cached != nil && cached.(summaryError).err != nil
	})
	if len(errStack) > 0 {
		err := mg.loadCache.Get(errStack[len(errStack)-1]).(summaryError).err
		var noUpgrade func(from, to module.Version) bool
		return mvs.NewBuildListError(err, errStack, noUpgrade)
	}

	return nil
}

func (mg *ModuleGraph) allRootsSelected() bool {
	roots, _ := mg.g.RequiredBy(Target)
	for _, m := range roots {
		if mg.Selected(m.Path) != m.Version {
			return false
		}
	}
	return true
}

// LoadModGraph loads and returns the graph of module dependencies of the main module,
// without loading any packages.
//
// Modules are loaded automatically (and lazily) in LoadPackages:
// LoadModGraph need only be called if LoadPackages is not,
// typically in commands that care about modules but no particular package.
func LoadModGraph(ctx context.Context) *ModuleGraph {
	rs, mg, err := expandGraph(ctx, LoadModFile(ctx))
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	commitRequirements(ctx, rs)
	return mg
}

// expandGraph loads the complete module graph from rs.
//
// If the complete graph reveals that some root of rs is not actually the
// selected version of its path, expandGraph computes a new set of roots that
// are consistent. (When lazy loading is implemented, this may result in
// upgrades to other modules due to requirements that were previously pruned
// out.)
//
// expandGraph returns the updated roots, along with the module graph loaded
// from those roots and any error encountered while loading that graph.
// expandGraph returns non-nil requirements and a non-nil graph regardless of
// errors. On error, the roots might not be updated to be consistent.
func expandGraph(ctx context.Context, rs *Requirements) (*Requirements, *ModuleGraph, error) {
	mg, mgErr := rs.Graph(ctx)
	if mgErr != nil {
		// Without the graph, we can't update the roots: we don't know which
		// versions of transitive dependencies would be selected.
		return rs, mg, mgErr
	}

	if !mg.allRootsSelected() {
		// The roots of rs are not consistent with the rest of the graph. Update
		// them. In an eager module this is a no-op for the build list as a whole —
		// it just promotes what were previously transitive requirements to be
		// roots — but in a lazy module it may pull in previously-irrelevant
		// transitive dependencies.

		newRS, rsErr := updateRoots(ctx, rs.direct, rs, nil)
		if rsErr != nil {
			// Failed to update roots, perhaps because of an error in a transitive
			// dependency needed for the update. Return the original Requirements
			// instead.
			return rs, mg, rsErr
		}
		rs = newRS
		mg, mgErr = rs.Graph(ctx)
	}

	return rs, mg, mgErr
}

// EditBuildList edits the global build list by first adding every module in add
// to the existing build list, then adjusting versions (and adding or removing
// requirements as needed) until every module in mustSelect is selected at the
// given version.
//
// (Note that the newly-added modules might not be selected in the resulting
// build list: they could be lower than existing requirements or conflict with
// versions in mustSelect.)
//
// If the versions listed in mustSelect are mutually incompatible (due to one of
// the listed modules requiring a higher version of another), EditBuildList
// returns a *ConstraintError and leaves the build list in its previous state.
//
// On success, EditBuildList reports whether the selected version of any module
// in the build list may have been changed (possibly to or from "none") as a
// result.
func EditBuildList(ctx context.Context, add, mustSelect []module.Version) (changed bool, err error) {
	rs, changed, err := editRequirements(ctx, LoadModFile(ctx), add, mustSelect)
	if err != nil {
		return false, err
	}
	commitRequirements(ctx, rs)
	return changed, err
}

// A ConstraintError describes inconsistent constraints in EditBuildList
type ConstraintError struct {
	// Conflict lists the source of the conflict for each version in mustSelect
	// that could not be selected due to the requirements of some other version in
	// mustSelect.
	Conflicts []Conflict
}

func (e *ConstraintError) Error() string {
	b := new(strings.Builder)
	b.WriteString("version constraints conflict:")
	for _, c := range e.Conflicts {
		fmt.Fprintf(b, "\n\t%v requires %v, but %v is requested", c.Source, c.Dep, c.Constraint)
	}
	return b.String()
}

// A Conflict documents that Source requires Dep, which conflicts with Constraint.
// (That is, Dep has the same module path as Constraint but a higher version.)
type Conflict struct {
	Source     module.Version
	Dep        module.Version
	Constraint module.Version
}

// tidyBuildList trims the build list to the minimal requirements needed to
// retain the same versions of all packages loaded by ld.
func tidyBuildList(ctx context.Context, ld *loader, initialRS *Requirements) *Requirements {
	if go117LazyTODO {
		// Tidy needs to maintain the lazy-loading invariants for lazy modules.
		// The implementation for eager modules should be factored out into a function.
	}

	depth := index.depth()
	if go117LazyTODO {
		// TODO(#45094): add a -go flag to 'go mod tidy' to allow the depth to be
		// changed after loading packages.
	}

	var (
		tidy *Requirements
		err  error
	)
	if depth == lazy {
		panic("internal error: 'go mod tidy' for lazy modules is not yet implemented")
	} else {
		tidy, err = tidyEagerRoots(ctx, ld.requirements, ld.pkgs)
	}
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	if cfg.BuildV {
		mg, _ := tidy.Graph(ctx)

		for _, m := range initialRS.rootModules {
			if mg.Selected(m.Path) == "none" {
				fmt.Fprintf(os.Stderr, "unused %s\n", m.Path)
			} else if go117LazyTODO {
				// If the main module is lazy and we demote a root to a non-root
				// (because it is not actually relevant), should we log that too?
			}
		}
	}

	return tidy
}

// tidyEagerRoots returns a minimal set of root requirements that maintains the
// selected version of every module that provided a package in pkgs, and
// includes the selected version of every such module in rs.direct as a root.
func tidyEagerRoots(ctx context.Context, rs *Requirements, pkgs []*loadPkg) (*Requirements, error) {
	var (
		keep     []module.Version
		keptPath = map[string]bool{}
	)
	var (
		rootPaths   []string // module paths that should be included as roots
		inRootPaths = map[string]bool{}
	)
	for _, pkg := range pkgs {
		if !pkg.fromExternalModule() {
			continue
		}
		if m := pkg.mod; !keptPath[m.Path] {
			keep = append(keep, m)
			keptPath[m.Path] = true
			if rs.direct[m.Path] && !inRootPaths[m.Path] {
				rootPaths = append(rootPaths, m.Path)
				inRootPaths[m.Path] = true
			}
		}
	}

	min, err := mvs.Req(Target, rootPaths, &mvsReqs{roots: keep})
	if err != nil {
		return nil, err
	}
	return newRequirements(eager, min, rs.direct), nil
}

// updateRoots returns a set of root requirements that includes the selected
// version of every module path in direct as a root, and maintains the selected
// version of every module selected in the graph of rs.
//
// The roots are updated such that:
//
// 	1. The selected version of every module path in direct is included as a root
// 	   (if it is not "none").
// 	2. Each root is the selected version of its path. (We say that such a root
// 	   set is “consistent”.)
// 	3. Every version selected in the graph of rs remains selected unless upgraded
// 	   by a dependency in add.
// 	4. Every version in add is selected at its given version unless upgraded by
// 	   an existing root or another module in add.
func updateRoots(ctx context.Context, direct map[string]bool, rs *Requirements, add []module.Version) (*Requirements, error) {
	mg, err := rs.Graph(ctx)
	if err != nil {
		// We can't ignore errors in the module graph even if the user passed the -e
		// flag to try to push past them. If we can't load the complete module
		// dependencies, then we can't reliably compute a minimal subset of them.
		return rs, err
	}

	if cfg.BuildMod != "mod" {
		// Instead of actually updating the requirements, just check that no updates
		// are needed.
		if rs == nil {
			// We're being asked to reconstruct the requirements from scratch,
			// but we aren't even allowed to modify them.
			return rs, errGoModDirty
		}
		for _, m := range rs.rootModules {
			if m.Version != mg.Selected(m.Path) {
				// The root version v is misleading: the actual selected version is higher.
				return rs, errGoModDirty
			}
		}
		for _, m := range add {
			if m.Version != mg.Selected(m.Path) {
				return rs, errGoModDirty
			}
		}
		for mPath := range direct {
			if _, ok := rs.rootSelected(mPath); !ok {
				// Module m is supposed to be listed explicitly, but isn't.
				//
				// Note that this condition is also detected (and logged with more
				// detail) earlier during package loading, so it shouldn't actually be
				// possible at this point — this is just a defense in depth.
				return rs, errGoModDirty
			}
		}

		// No explicit roots are missing and all roots are already at the versions
		// we want to keep. Any other changes we would make are purely cosmetic,
		// such as pruning redundant indirect dependencies. Per issue #34822, we
		// ignore cosmetic changes when we cannot update the go.mod file.
		return rs, nil
	}

	var (
		rootPaths   []string // module paths that should be included as roots
		inRootPaths = map[string]bool{}
	)
	for _, root := range rs.rootModules {
		// If the selected version of the root is the same as what was already
		// listed in the go.mod file, retain it as a root (even if redundant) to
		// avoid unnecessary churn. (See https://golang.org/issue/34822.)
		//
		// We do this even for indirect requirements, since we don't know why they
		// were added and they could become direct at any time.
		if !inRootPaths[root.Path] && mg.Selected(root.Path) == root.Version {
			rootPaths = append(rootPaths, root.Path)
			inRootPaths[root.Path] = true
		}
	}

	// “The selected version of every module path in direct is included as a root.”
	//
	// This is only for convenience and clarity for end users: the choice of
	// explicit vs. implicit dependency has no impact on MVS selection (for itself
	// or any other module).
	if go117LazyTODO {
		// Update the above comment to reflect lazy loading once implemented.
	}
	keep := append(mg.BuildList()[1:], add...)
	for _, m := range keep {
		if direct[m.Path] && !inRootPaths[m.Path] {
			rootPaths = append(rootPaths, m.Path)
			inRootPaths[m.Path] = true
		}
	}

	min, err := mvs.Req(Target, rootPaths, &mvsReqs{roots: keep})
	if err != nil {
		return rs, err
	}

	// Note: if it turns out that we spend a lot of time reconstructing module
	// graphs after this point, we could make some effort here to detect whether
	// the root set is the same as the original root set in rs and recycle its
	// module graph and build list, if they have already been loaded.

	return newRequirements(rs.depth, min, direct), nil
}

// checkMultiplePaths verifies that a given module path is used as itself
// or as a replacement for another module, but not both at the same time.
//
// (See https://golang.org/issue/26607 and https://golang.org/issue/34650.)
func checkMultiplePaths(rs *Requirements) {
	mods := rs.rootModules
	if cached := rs.graph.Load(); cached != nil {
		if mg := cached.(cachedGraph).mg; mg != nil {
			mods = mg.BuildList()
		}
	}

	firstPath := map[module.Version]string{}
	for _, mod := range mods {
		src := resolveReplacement(mod)
		if prev, ok := firstPath[src]; !ok {
			firstPath[src] = mod.Path
		} else if prev != mod.Path {
			base.Errorf("go: %s@%s used for two different module paths (%s and %s)", src.Path, src.Version, prev, mod.Path)
		}
	}
	base.ExitIfErrors()
}
