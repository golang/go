// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/mvs"
	"cmd/go/internal/par"

	"golang.org/x/mod/module"
)

// A Requirements represents a logically-immutable set of root module requirements.
type Requirements struct {
	// pruning is the pruning at which the requirement graph is computed.
	//
	// If unpruned, the graph includes all transitive requirements regardless
	// of whether the requiring module supports pruning.
	//
	// If pruned, the graph includes only the root modules, the explicit
	// requirements of those root modules, and the transitive requirements of only
	// the root modules that do not support pruning.
	//
	// If workspace, the graph includes only the workspace modules, the explicit
	// requirements of the workspace modules, and the transitive requirements of
	// the workspace modules that do not support pruning.
	pruning modPruning

	// rootModules is the set of root modules of the graph, sorted and capped to
	// length. It may contain duplicates, and may contain multiple versions for a
	// given module path. The root modules of the graph are the set of main
	// modules in workspace mode, and the main module's direct requirements
	// outside workspace mode.
	//
	// The roots are always expected to contain an entry for the "go" module,
	// indicating the Go language version in use.
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

	graphOnce sync.Once // guards writes to (but not reads from) graph
	graph     atomic.Pointer[cachedGraph]
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

func mustHaveGoRoot(roots []module.Version) {
	for _, m := range roots {
		if m.Path == "go" {
			return
		}
	}
	panic("go: internal error: missing go root module")
}

// newRequirements returns a new requirement set with the given root modules.
// The dependencies of the roots will be loaded lazily at the first call to the
// Graph method.
//
// The rootModules slice must be sorted according to gover.ModSort.
// The caller must not modify the rootModules slice or direct map after passing
// them to newRequirements.
//
// If vendoring is in effect, the caller must invoke initVendor on the returned
// *Requirements before any other method.
func newRequirements(pruning modPruning, rootModules []module.Version, direct map[string]bool) *Requirements {
	mustHaveGoRoot(rootModules)

	if pruning != workspace {
		if workFilePath != "" {
			panic("in workspace mode, but pruning is not workspace in newRequirements")
		}
	}

	if pruning != workspace {
		if workFilePath != "" {
			panic("in workspace mode, but pruning is not workspace in newRequirements")
		}
		for i, m := range rootModules {
			if m.Version == "" && MainModules.Contains(m.Path) {
				panic(fmt.Sprintf("newRequirements called with untrimmed build list: rootModules[%v] is a main module", i))
			}
			if m.Path == "" || m.Version == "" {
				panic(fmt.Sprintf("bad requirement: rootModules[%v] = %v", i, m))
			}
		}
	}

	rs := &Requirements{
		pruning:        pruning,
		rootModules:    rootModules,
		maxRootVersion: make(map[string]string, len(rootModules)),
		direct:         direct,
	}

	for i, m := range rootModules {
		if i > 0 {
			prev := rootModules[i-1]
			if prev.Path > m.Path || (prev.Path == m.Path && gover.ModCompare(m.Path, prev.Version, m.Version) > 0) {
				panic(fmt.Sprintf("newRequirements called with unsorted roots: %v", rootModules))
			}
		}

		if v, ok := rs.maxRootVersion[m.Path]; ok && gover.ModCompare(m.Path, v, m.Version) >= 0 {
			continue
		}
		rs.maxRootVersion[m.Path] = m.Version
	}

	if rs.maxRootVersion["go"] == "" {
		panic(`newRequirements called without a "go" version`)
	}
	return rs
}

// String returns a string describing the Requirements for debugging.
func (rs *Requirements) String() string {
	return fmt.Sprintf("{%v %v}", rs.pruning, rs.rootModules)
}

// initVendor initializes rs.graph from the given list of vendored module
// dependencies, overriding the graph that would normally be loaded from module
// requirements.
func (rs *Requirements) initVendor(vendorList []module.Version) {
	rs.graphOnce.Do(func() {
		roots := MainModules.Versions()
		if inWorkspaceMode() {
			// Use rs.rootModules to pull in the go and toolchain roots
			// from the go.work file and preserve the invariant that all
			// of rs.rootModules are in mg.g.
			roots = rs.rootModules
		}
		mg := &ModuleGraph{
			g: mvs.NewGraph(cmpVersion, roots),
		}

		if rs.pruning == pruned {
			mainModule := MainModules.mustGetSingleMainModule()
			// The roots of a single pruned module should already include every module in the
			// vendor list, because the vendored modules are the same as those needed
			// for graph pruning.
			//
			// Just to be sure, we'll double-check that here.
			inconsistent := false
			for _, m := range vendorList {
				if v, ok := rs.rootSelected(m.Path); !ok || v != m.Version {
					base.Errorf("go: vendored module %v should be required explicitly in go.mod", m)
					inconsistent = true
				}
			}
			if inconsistent {
				base.Fatal(errGoModDirty)
			}

			// Now we can treat the rest of the module graph as effectively “pruned
			// out”, as though we are viewing the main module from outside: in vendor
			// mode, the root requirements *are* the complete module graph.
			mg.g.Require(mainModule, rs.rootModules)
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
			if inWorkspaceMode() {
				for _, m := range MainModules.Versions() {
					reqs, _ := rootsFromModFile(m, MainModules.ModFile(m), omitToolchainRoot)
					mg.g.Require(m, append(reqs, vendorMod))
				}
				mg.g.Require(vendorMod, vendorList)

			} else {
				mainModule := MainModules.mustGetSingleMainModule()
				mg.g.Require(mainModule, append(rs.rootModules, vendorMod))
				mg.g.Require(vendorMod, vendorList)
			}
		}

		rs.graph.Store(&cachedGraph{mg, nil})
	})
}

// GoVersion returns the Go language version for the Requirements.
func (rs *Requirements) GoVersion() string {
	v, _ := rs.rootSelected("go")
	if v == "" {
		panic("internal error: missing go version in modload.Requirements")
	}
	return v
}

// rootSelected returns the version of the root dependency with the given module
// path, or the zero module.Version and ok=false if the module is not a root
// dependency.
func (rs *Requirements) rootSelected(path string) (version string, ok bool) {
	if MainModules.Contains(path) {
		return "", true
	}
	if v, ok := rs.maxRootVersion[path]; ok {
		return v, true
	}
	return "", false
}

// hasRedundantRoot returns true if the root list contains multiple requirements
// of the same module or a requirement on any version of the main module.
// Redundant requirements should be pruned, but they may influence version
// selection.
func (rs *Requirements) hasRedundantRoot() bool {
	for i, m := range rs.rootModules {
		if MainModules.Contains(m.Path) || (i > 0 && m.Path == rs.rootModules[i-1].Path) {
			return true
		}
	}
	return false
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
		mg, mgErr := readModGraph(ctx, rs.pruning, rs.rootModules, nil)
		rs.graph.Store(&cachedGraph{mg, mgErr})
	})
	cached := rs.graph.Load()
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
// If the main module supports module graph pruning, the graph does not include
// transitive dependencies of non-root (implicit) dependencies.
type ModuleGraph struct {
	g         *mvs.Graph
	loadCache par.ErrCache[module.Version, *modFileSummary]

	buildListOnce sync.Once
	buildList     []module.Version
}

var readModGraphDebugOnce sync.Once

// readModGraph reads and returns the module dependency graph starting at the
// given roots.
//
// The requirements of the module versions found in the unprune map are included
// in the graph even if they would normally be pruned out.
//
// Unlike LoadModGraph, readModGraph does not attempt to diagnose or update
// inconsistent roots.
func readModGraph(ctx context.Context, pruning modPruning, roots []module.Version, unprune map[module.Version]bool) (*ModuleGraph, error) {
	mustHaveGoRoot(roots)
	if pruning == pruned {
		// Enable diagnostics for lazy module loading
		// (https://golang.org/ref/mod#lazy-loading) only if the module graph is
		// pruned.
		//
		// In unpruned modules,we load the module graph much more aggressively (in
		// order to detect inconsistencies that wouldn't be feasible to spot-check),
		// so it wouldn't be useful to log when that occurs (because it happens in
		// normal operation all the time).
		readModGraphDebugOnce.Do(func() {
			for _, f := range strings.Split(os.Getenv("GODEBUG"), ",") {
				switch f {
				case "lazymod=log":
					debug.PrintStack()
					fmt.Fprintf(os.Stderr, "go: read full module graph.\n")
				case "lazymod=strict":
					debug.PrintStack()
					base.Fatalf("go: read full module graph (forbidden by GODEBUG=lazymod=strict).")
				}
			}
		})
	}

	var graphRoots []module.Version
	if inWorkspaceMode() {
		graphRoots = roots
	} else {
		graphRoots = MainModules.Versions()
	}
	var (
		mu       sync.Mutex // guards mg.g and hasError during loading
		hasError bool
		mg       = &ModuleGraph{
			g: mvs.NewGraph(cmpVersion, graphRoots),
		}
	)

	if pruning != workspace {
		if inWorkspaceMode() {
			panic("pruning is not workspace in workspace mode")
		}
		mg.g.Require(MainModules.mustGetSingleMainModule(), roots)
	}

	type dedupKey struct {
		m       module.Version
		pruning modPruning
	}
	var (
		loadQueue = par.NewQueue(runtime.GOMAXPROCS(0))
		loading   sync.Map // dedupKey → nil; the set of modules that have been or are being loaded
	)

	// loadOne synchronously loads the explicit requirements for module m.
	// It does not load the transitive requirements of m even if the go version in
	// m's go.mod file indicates that it supports graph pruning.
	loadOne := func(m module.Version) (*modFileSummary, error) {
		return mg.loadCache.Do(m, func() (*modFileSummary, error) {
			summary, err := goModSummary(m)

			mu.Lock()
			if err == nil {
				mg.g.Require(m, summary.require)
			} else {
				hasError = true
			}
			mu.Unlock()

			return summary, err
		})
	}

	var enqueue func(m module.Version, pruning modPruning)
	enqueue = func(m module.Version, pruning modPruning) {
		if m.Version == "none" {
			return
		}

		if _, dup := loading.LoadOrStore(dedupKey{m, pruning}, nil); dup {
			// m has already been enqueued for loading. Since unpruned loading may
			// follow cycles in the requirement graph, we need to return early
			// to avoid making the load queue infinitely long.
			return
		}

		loadQueue.Add(func() {
			summary, err := loadOne(m)
			if err != nil {
				return // findError will report the error later.
			}

			// If the version in m's go.mod file does not support pruning, then we
			// cannot assume that the explicit requirements of m (added by loadOne)
			// are sufficient to build the packages it contains. We must load its full
			// transitive dependency graph to be sure that we see all relevant
			// dependencies. In addition, we must load the requirements of any module
			// that is explicitly marked as unpruned.
			nextPruning := summary.pruning
			if pruning == unpruned {
				nextPruning = unpruned
			}
			for _, r := range summary.require {
				if pruning != pruned || summary.pruning == unpruned || unprune[r] {
					enqueue(r, nextPruning)
				}
			}
		})
	}

	mustHaveGoRoot(roots)
	for _, m := range roots {
		enqueue(m, pruning)
	}
	<-loadQueue.Idle()

	// Reload any dependencies of the main modules which are not
	// at their selected versions at workspace mode, because the
	// requirements don't accurately reflect the transitive imports.
	if pruning == workspace {
		// hasDepsInAll contains the set of modules that need to be loaded
		// at workspace pruning because any of their dependencies may
		// provide packages in all.
		hasDepsInAll := make(map[string]bool)
		seen := map[module.Version]bool{}
		for _, m := range roots {
			hasDepsInAll[m.Path] = true
		}
		// This loop will terminate because it will call enqueue on each version of
		// each dependency of the modules in hasDepsInAll at most once (and only
		// calls enqueue on successively increasing versions of each dependency).
		for {
			needsEnqueueing := map[module.Version]bool{}
			for p := range hasDepsInAll {
				m := module.Version{Path: p, Version: mg.g.Selected(p)}
				if !seen[m] {
					needsEnqueueing[m] = true
					continue
				}
				reqs, _ := mg.g.RequiredBy(m)
				for _, r := range reqs {
					s := module.Version{Path: r.Path, Version: mg.g.Selected(r.Path)}
					if gover.ModCompare(r.Path, s.Version, r.Version) > 0 && !seen[s] {
						needsEnqueueing[s] = true
					}
				}
			}
			// add all needs enqueueing to paths we care about
			if len(needsEnqueueing) == 0 {
				break
			}

			for p := range needsEnqueueing {
				enqueue(p, workspace)
				seen[p] = true
				hasDepsInAll[p.Path] = true
			}
			<-loadQueue.Idle()
		}
	}

	if hasError {
		return mg, mg.findError()
	}
	return mg, nil
}

// RequiredBy returns the dependencies required by module m in the graph,
// or ok=false if module m's dependencies are pruned out.
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
// beginning with the main modules.
//
// The order of the remaining elements in the list is deterministic
// but arbitrary.
//
// The caller must not modify the returned list, but may safely append to it
// and may rely on it not to be modified.
func (mg *ModuleGraph) BuildList() []module.Version {
	mg.buildListOnce.Do(func() {
		mg.buildList = slices.Clip(mg.g.BuildList())
	})
	return mg.buildList
}

func (mg *ModuleGraph) findError() error {
	errStack := mg.g.FindPath(func(m module.Version) bool {
		_, err := mg.loadCache.Get(m)
		return err != nil && err != par.ErrCacheEntryNotFound
	})
	if len(errStack) > 0 {
		_, err := mg.loadCache.Get(errStack[len(errStack)-1])
		var noUpgrade func(from, to module.Version) bool
		return mvs.NewBuildListError(err, errStack, noUpgrade)
	}

	return nil
}

func (mg *ModuleGraph) allRootsSelected() bool {
	var roots []module.Version
	if inWorkspaceMode() {
		roots = MainModules.Versions()
	} else {
		roots, _ = mg.g.RequiredBy(MainModules.mustGetSingleMainModule())
	}
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
// If the goVersion string is non-empty, the returned graph is the graph
// as interpreted by the given Go version (instead of the version indicated
// in the go.mod file).
//
// Modules are loaded automatically (and lazily) in LoadPackages:
// LoadModGraph need only be called if LoadPackages is not,
// typically in commands that care about modules but no particular package.
func LoadModGraph(ctx context.Context, goVersion string) (*ModuleGraph, error) {
	rs, err := loadModFile(ctx, nil)
	if err != nil {
		return nil, err
	}

	if goVersion != "" {
		v, _ := rs.rootSelected("go")
		if gover.Compare(v, gover.GoStrictVersion) >= 0 && gover.Compare(goVersion, v) < 0 {
			return nil, fmt.Errorf("requested Go version %s cannot load module graph (requires Go >= %s)", goVersion, v)
		}

		pruning := pruningForGoVersion(goVersion)
		if pruning == unpruned && rs.pruning != unpruned {
			// Use newRequirements instead of convertDepth because convertDepth
			// also updates roots; here, we want to report the unmodified roots
			// even though they may seem inconsistent.
			rs = newRequirements(unpruned, rs.rootModules, rs.direct)
		}

		return rs.Graph(ctx)
	}

	rs, mg, err := expandGraph(ctx, rs)
	if err != nil {
		return nil, err
	}
	requirements = rs
	return mg, err
}

// expandGraph loads the complete module graph from rs.
//
// If the complete graph reveals that some root of rs is not actually the
// selected version of its path, expandGraph computes a new set of roots that
// are consistent. (With a pruned module graph, this may result in upgrades to
// other modules due to requirements that were previously pruned out.)
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
		// them. In an unpruned module this is a no-op for the build list as a whole —
		// it just promotes what were previously transitive requirements to be
		// roots — but in a pruned module it may pull in previously-irrelevant
		// transitive dependencies.

		newRS, rsErr := updateRoots(ctx, rs.direct, rs, nil, nil, false)
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
	requirements = rs
	return changed, err
}

// OverrideRoots edits the global requirement roots by replacing the specific module versions.
func OverrideRoots(ctx context.Context, replace []module.Version) {
	requirements = overrideRoots(ctx, requirements, replace)
}

func overrideRoots(ctx context.Context, rs *Requirements, replace []module.Version) *Requirements {
	drop := make(map[string]bool)
	for _, m := range replace {
		drop[m.Path] = true
	}
	var roots []module.Version
	for _, m := range rs.rootModules {
		if !drop[m.Path] {
			roots = append(roots, m)
		}
	}
	roots = append(roots, replace...)
	gover.ModSort(roots)
	return newRequirements(rs.pruning, roots, rs.direct)
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
		fmt.Fprintf(b, "\n\t%s", c.Summary())
	}
	return b.String()
}

// A Conflict is a path of requirements starting at a root or proposed root in
// the requirement graph, explaining why that root either causes a module passed
// in the mustSelect list to EditBuildList to be unattainable, or introduces an
// unresolvable error in loading the requirement graph.
type Conflict struct {
	// Path is a path of requirements starting at some module version passed in
	// the mustSelect argument and ending at a module whose requirements make that
	// version unacceptable. (Path always has len ≥ 1.)
	Path []module.Version

	// If Err is nil, Constraint is a module version passed in the mustSelect
	// argument that has the same module path as, and a lower version than,
	// the last element of the Path slice.
	Constraint module.Version

	// If Constraint is unset, Err is an error encountered when loading the
	// requirements of the last element in Path.
	Err error
}

// UnwrapModuleError returns c.Err, but unwraps it if it is a module.ModuleError
// with a version and path matching the last entry in the Path slice.
func (c Conflict) UnwrapModuleError() error {
	me, ok := c.Err.(*module.ModuleError)
	if ok && len(c.Path) > 0 {
		last := c.Path[len(c.Path)-1]
		if me.Path == last.Path && me.Version == last.Version {
			return me.Err
		}
	}
	return c.Err
}

// Summary returns a string that describes only the first and last modules in
// the conflict path.
func (c Conflict) Summary() string {
	if len(c.Path) == 0 {
		return "(internal error: invalid Conflict struct)"
	}
	first := c.Path[0]
	last := c.Path[len(c.Path)-1]
	if len(c.Path) == 1 {
		if c.Err != nil {
			return fmt.Sprintf("%s: %v", first, c.UnwrapModuleError())
		}
		return fmt.Sprintf("%s is above %s", first, c.Constraint.Version)
	}

	adverb := ""
	if len(c.Path) > 2 {
		adverb = "indirectly "
	}
	if c.Err != nil {
		return fmt.Sprintf("%s %srequires %s: %v", first, adverb, last, c.UnwrapModuleError())
	}
	return fmt.Sprintf("%s %srequires %s, but %s is requested", first, adverb, last, c.Constraint.Version)
}

// String returns a string that describes the full conflict path.
func (c Conflict) String() string {
	if len(c.Path) == 0 {
		return "(internal error: invalid Conflict struct)"
	}
	b := new(strings.Builder)
	fmt.Fprintf(b, "%v", c.Path[0])
	if len(c.Path) == 1 {
		fmt.Fprintf(b, " found")
	} else {
		for _, r := range c.Path[1:] {
			fmt.Fprintf(b, " requires\n\t%v", r)
		}
	}
	if c.Constraint != (module.Version{}) {
		fmt.Fprintf(b, ", but %v is requested", c.Constraint.Version)
	}
	if c.Err != nil {
		fmt.Fprintf(b, ": %v", c.UnwrapModuleError())
	}
	return b.String()
}

// tidyRoots trims the root dependencies to the minimal requirements needed to
// both retain the same versions of all packages in pkgs and satisfy the
// graph-pruning invariants (if applicable).
func tidyRoots(ctx context.Context, rs *Requirements, pkgs []*loadPkg) (*Requirements, error) {
	mainModule := MainModules.mustGetSingleMainModule()
	if rs.pruning == unpruned {
		return tidyUnprunedRoots(ctx, mainModule, rs, pkgs)
	}
	return tidyPrunedRoots(ctx, mainModule, rs, pkgs)
}

func updateRoots(ctx context.Context, direct map[string]bool, rs *Requirements, pkgs []*loadPkg, add []module.Version, rootsImported bool) (*Requirements, error) {
	switch rs.pruning {
	case unpruned:
		return updateUnprunedRoots(ctx, direct, rs, add)
	case pruned:
		return updatePrunedRoots(ctx, direct, rs, pkgs, add, rootsImported)
	case workspace:
		return updateWorkspaceRoots(ctx, rs, add)
	default:
		panic(fmt.Sprintf("unsupported pruning mode: %v", rs.pruning))
	}
}

func updateWorkspaceRoots(ctx context.Context, rs *Requirements, add []module.Version) (*Requirements, error) {
	if len(add) != 0 {
		// add should be empty in workspace mode because workspace mode implies
		// -mod=readonly, which in turn implies no new requirements. The code path
		// that would result in add being non-empty returns an error before it
		// reaches this point: The set of modules to add comes from
		// resolveMissingImports, which in turn resolves each package by calling
		// queryImport. But queryImport explicitly checks for -mod=readonly, and
		// return an error.
		panic("add is not empty")
	}
	return rs, nil
}

// tidyPrunedRoots returns a minimal set of root requirements that maintains the
// invariants of the go.mod file needed to support graph pruning for the given
// packages:
//
//  1. For each package marked with pkgInAll, the module path that provided that
//     package is included as a root.
//  2. For all packages, the module that provided that package either remains
//     selected at the same version or is upgraded by the dependencies of a
//     root.
//
// If any module that provided a package has been upgraded above its previous
// version, the caller may need to reload and recompute the package graph.
//
// To ensure that the loading process eventually converges, the caller should
// add any needed roots from the tidy root set (without removing existing untidy
// roots) until the set of roots has converged.
func tidyPrunedRoots(ctx context.Context, mainModule module.Version, old *Requirements, pkgs []*loadPkg) (*Requirements, error) {
	var (
		roots      []module.Version
		pathIsRoot = map[string]bool{mainModule.Path: true}
	)
	if v, ok := old.rootSelected("go"); ok {
		roots = append(roots, module.Version{Path: "go", Version: v})
		pathIsRoot["go"] = true
	}
	if v, ok := old.rootSelected("toolchain"); ok {
		roots = append(roots, module.Version{Path: "toolchain", Version: v})
		pathIsRoot["toolchain"] = true
	}
	// We start by adding roots for every package in "all".
	//
	// Once that is done, we may still need to add more roots to cover upgraded or
	// otherwise-missing test dependencies for packages in "all". For those test
	// dependencies, we prefer to add roots for packages with shorter import
	// stacks first, on the theory that the module requirements for those will
	// tend to fill in the requirements for their transitive imports (which have
	// deeper import stacks). So we add the missing dependencies for one depth at
	// a time, starting with the packages actually in "all" and expanding outwards
	// until we have scanned every package that was loaded.
	var (
		queue  []*loadPkg
		queued = map[*loadPkg]bool{}
	)
	for _, pkg := range pkgs {
		if !pkg.flags.has(pkgInAll) {
			continue
		}
		if pkg.fromExternalModule() && !pathIsRoot[pkg.mod.Path] {
			roots = append(roots, pkg.mod)
			pathIsRoot[pkg.mod.Path] = true
		}
		queue = append(queue, pkg)
		queued[pkg] = true
	}
	gover.ModSort(roots)
	tidy := newRequirements(pruned, roots, old.direct)

	for len(queue) > 0 {
		roots = tidy.rootModules
		mg, err := tidy.Graph(ctx)
		if err != nil {
			return nil, err
		}

		prevQueue := queue
		queue = nil
		for _, pkg := range prevQueue {
			m := pkg.mod
			if m.Path == "" {
				continue
			}
			for _, dep := range pkg.imports {
				if !queued[dep] {
					queue = append(queue, dep)
					queued[dep] = true
				}
			}
			if pkg.test != nil && !queued[pkg.test] {
				queue = append(queue, pkg.test)
				queued[pkg.test] = true
			}

			if !pathIsRoot[m.Path] {
				if s := mg.Selected(m.Path); gover.ModCompare(m.Path, s, m.Version) < 0 {
					roots = append(roots, m)
					pathIsRoot[m.Path] = true
				}
			}
		}

		if len(roots) > len(tidy.rootModules) {
			gover.ModSort(roots)
			tidy = newRequirements(pruned, roots, tidy.direct)
		}
	}

	roots = tidy.rootModules
	_, err := tidy.Graph(ctx)
	if err != nil {
		return nil, err
	}

	// We try to avoid adding explicit requirements for test-only dependencies of
	// packages in external modules. However, if we drop the explicit
	// requirements, that may change an import from unambiguous (due to lazy
	// module loading) to ambiguous (because lazy module loading no longer
	// disambiguates it). For any package that has become ambiguous, we try
	// to fix it by promoting its module to an explicit root.
	// (See https://go.dev/issue/60313.)
	q := par.NewQueue(runtime.GOMAXPROCS(0))
	for {
		var disambiguateRoot sync.Map
		for _, pkg := range pkgs {
			if pkg.mod.Path == "" || pathIsRoot[pkg.mod.Path] {
				// Lazy module loading will cause pkg.mod to be checked before any other modules
				// that are only indirectly required. It is as unambiguous as possible.
				continue
			}
			pkg := pkg
			q.Add(func() {
				skipModFile := true
				_, _, _, _, err := importFromModules(ctx, pkg.path, tidy, nil, skipModFile)
				if aie := (*AmbiguousImportError)(nil); errors.As(err, &aie) {
					disambiguateRoot.Store(pkg.mod, true)
				}
			})
		}
		<-q.Idle()

		disambiguateRoot.Range(func(k, _ any) bool {
			m := k.(module.Version)
			roots = append(roots, m)
			pathIsRoot[m.Path] = true
			return true
		})

		if len(roots) > len(tidy.rootModules) {
			module.Sort(roots)
			tidy = newRequirements(pruned, roots, tidy.direct)
			_, err = tidy.Graph(ctx)
			if err != nil {
				return nil, err
			}
			// Adding these roots may have pulled additional modules into the module
			// graph, causing additional packages to become ambiguous. Keep iterating
			// until we reach a fixed point.
			continue
		}

		break
	}

	return tidy, nil
}

// updatePrunedRoots returns a set of root requirements that maintains the
// invariants of the go.mod file needed to support graph pruning:
//
//  1. The selected version of the module providing each package marked with
//     either pkgInAll or pkgIsRoot is included as a root.
//     Note that certain root patterns (such as '...') may explode the root set
//     to contain every module that provides any package imported (or merely
//     required) by any other module.
//  2. Each root appears only once, at the selected version of its path
//     (if rs.graph is non-nil) or at the highest version otherwise present as a
//     root (otherwise).
//  3. Every module path that appears as a root in rs remains a root.
//  4. Every version in add is selected at its given version unless upgraded by
//     (the dependencies of) an existing root or another module in add.
//
// The packages in pkgs are assumed to have been loaded from either the roots of
// rs or the modules selected in the graph of rs.
//
// The above invariants together imply the graph-pruning invariants for the
// go.mod file:
//
//  1. (The import invariant.) Every module that provides a package transitively
//     imported by any package or test in the main module is included as a root.
//     This follows by induction from (1) and (3) above. Transitively-imported
//     packages loaded during this invocation are marked with pkgInAll (1),
//     and by hypothesis any transitively-imported packages loaded in previous
//     invocations were already roots in rs (3).
//
//  2. (The argument invariant.) Every module that provides a package matching
//     an explicit package pattern is included as a root. This follows directly
//     from (1): packages matching explicit package patterns are marked with
//     pkgIsRoot.
//
//  3. (The completeness invariant.) Every module that contributed any package
//     to the build is required by either the main module or one of the modules
//     it requires explicitly. This invariant is left up to the caller, who must
//     not load packages from outside the module graph but may add roots to the
//     graph, but is facilitated by (3). If the caller adds roots to the graph in
//     order to resolve missing packages, then updatePrunedRoots will retain them,
//     the selected versions of those roots cannot regress, and they will
//     eventually be written back to the main module's go.mod file.
//
// (See https://golang.org/design/36460-lazy-module-loading#invariants for more
// detail.)
func updatePrunedRoots(ctx context.Context, direct map[string]bool, rs *Requirements, pkgs []*loadPkg, add []module.Version, rootsImported bool) (*Requirements, error) {
	roots := rs.rootModules
	rootsUpgraded := false

	spotCheckRoot := map[module.Version]bool{}

	// “The selected version of the module providing each package marked with
	// either pkgInAll or pkgIsRoot is included as a root.”
	needSort := false
	for _, pkg := range pkgs {
		if !pkg.fromExternalModule() {
			// pkg was not loaded from a module dependency, so we don't need
			// to do anything special to maintain that dependency.
			continue
		}

		switch {
		case pkg.flags.has(pkgInAll):
			// pkg is transitively imported by a package or test in the main module.
			// We need to promote the module that maintains it to a root: if some
			// other module depends on the main module, and that other module also
			// uses a pruned module graph, it will expect to find all of our
			// transitive dependencies by reading just our go.mod file, not the go.mod
			// files of everything we depend on.
			//
			// (This is the “import invariant” that makes graph pruning possible.)

		case rootsImported && pkg.flags.has(pkgFromRoot):
			// pkg is a transitive dependency of some root, and we are treating the
			// roots as if they are imported by the main module (as in 'go get').

		case pkg.flags.has(pkgIsRoot):
			// pkg is a root of the package-import graph. (Generally this means that
			// it matches a command-line argument.) We want future invocations of the
			// 'go' command — such as 'go test' on the same package — to continue to
			// use the same versions of its dependencies that we are using right now.
			// So we need to bring this package's dependencies inside the pruned
			// module graph.
			//
			// Making the module containing this package a root of the module graph
			// does exactly that: if the module containing the package supports graph
			// pruning then it should satisfy the import invariant itself, so all of
			// its dependencies should be in its go.mod file, and if the module
			// containing the package does not support pruning then if we make it a
			// root we will load all of its (unpruned) transitive dependencies into
			// the module graph.
			//
			// (This is the “argument invariant”, and is important for
			// reproducibility.)

		default:
			// pkg is a dependency of some other package outside of the main module.
			// As far as we know it's not relevant to the main module (and thus not
			// relevant to consumers of the main module either), and its dependencies
			// should already be in the module graph — included in the dependencies of
			// the package that imported it.
			continue
		}

		if _, ok := rs.rootSelected(pkg.mod.Path); ok {
			// It is possible that the main module's go.mod file is incomplete or
			// otherwise erroneous — for example, perhaps the author forgot to 'git
			// add' their updated go.mod file after adding a new package import, or
			// perhaps they made an edit to the go.mod file using a third-party tool
			// ('git merge'?) that doesn't maintain consistency for module
			// dependencies. If that happens, ideally we want to detect the missing
			// requirements and fix them up here.
			//
			// However, we also need to be careful not to be too aggressive. For
			// transitive dependencies of external tests, the go.mod file for the
			// module containing the test itself is expected to provide all of the
			// relevant dependencies, and we explicitly don't want to pull in
			// requirements on *irrelevant* requirements that happen to occur in the
			// go.mod files for these transitive-test-only dependencies. (See the test
			// in mod_lazy_test_horizon.txt for a concrete example).
			//
			// The “goldilocks zone” seems to be to spot-check exactly the same
			// modules that we promote to explicit roots: namely, those that provide
			// packages transitively imported by the main module, and those that
			// provide roots of the package-import graph. That will catch erroneous
			// edits to the main module's go.mod file and inconsistent requirements in
			// dependencies that provide imported packages, but will ignore erroneous
			// or misleading requirements in dependencies that aren't obviously
			// relevant to the packages in the main module.
			spotCheckRoot[pkg.mod] = true
		} else {
			roots = append(roots, pkg.mod)
			rootsUpgraded = true
			// The roots slice was initially sorted because rs.rootModules was sorted,
			// but the root we just added could be out of order.
			needSort = true
		}
	}

	for _, m := range add {
		if v, ok := rs.rootSelected(m.Path); !ok || gover.ModCompare(m.Path, v, m.Version) < 0 {
			roots = append(roots, m)
			rootsUpgraded = true
			needSort = true
		}
	}
	if needSort {
		gover.ModSort(roots)
	}

	// "Each root appears only once, at the selected version of its path ….”
	for {
		var mg *ModuleGraph
		if rootsUpgraded {
			// We've added or upgraded one or more roots, so load the full module
			// graph so that we can update those roots to be consistent with other
			// requirements.
			if mustHaveCompleteRequirements() {
				// Our changes to the roots may have moved dependencies into or out of
				// the graph-pruning horizon, which could in turn change the selected
				// versions of other modules. (For pruned modules adding or removing an
				// explicit root is a semantic change, not just a cosmetic one.)
				return rs, errGoModDirty
			}

			rs = newRequirements(pruned, roots, direct)
			var err error
			mg, err = rs.Graph(ctx)
			if err != nil {
				return rs, err
			}
		} else {
			// Since none of the roots have been upgraded, we have no reason to
			// suspect that they are inconsistent with the requirements of any other
			// roots. Only look at the full module graph if we've already loaded it;
			// otherwise, just spot-check the explicit requirements of the roots from
			// which we loaded packages.
			if rs.graph.Load() != nil {
				// We've already loaded the full module graph, which includes the
				// requirements of all of the root modules — even the transitive
				// requirements, if they are unpruned!
				mg, _ = rs.Graph(ctx)
			} else if cfg.BuildMod == "vendor" {
				// We can't spot-check the requirements of other modules because we
				// don't in general have their go.mod files available in the vendor
				// directory. (Fortunately this case is impossible, because mg.graph is
				// always non-nil in vendor mode!)
				panic("internal error: rs.graph is unexpectedly nil with -mod=vendor")
			} else if !spotCheckRoots(ctx, rs, spotCheckRoot) {
				// We spot-checked the explicit requirements of the roots that are
				// relevant to the packages we've loaded. Unfortunately, they're
				// inconsistent in some way; we need to load the full module graph
				// so that we can fix the roots properly.
				var err error
				mg, err = rs.Graph(ctx)
				if err != nil {
					return rs, err
				}
			}
		}

		roots = make([]module.Version, 0, len(rs.rootModules))
		rootsUpgraded = false
		inRootPaths := make(map[string]bool, len(rs.rootModules)+1)
		for _, mm := range MainModules.Versions() {
			inRootPaths[mm.Path] = true
		}
		for _, m := range rs.rootModules {
			if inRootPaths[m.Path] {
				// This root specifies a redundant path. We already retained the
				// selected version of this path when we saw it before, so omit the
				// redundant copy regardless of its version.
				//
				// When we read the full module graph, we include the dependencies of
				// every root even if that root is redundant. That better preserves
				// reproducibility if, say, some automated tool adds a redundant
				// 'require' line and then runs 'go mod tidy' to try to make everything
				// consistent, since the requirements of the older version are carried
				// over.
				//
				// So omitting a root that was previously present may *reduce* the
				// selected versions of non-roots, but merely removing a requirement
				// cannot *increase* the selected versions of other roots as a result —
				// we don't need to mark this change as an upgrade. (This particular
				// change cannot invalidate any other roots.)
				continue
			}

			var v string
			if mg == nil {
				v, _ = rs.rootSelected(m.Path)
			} else {
				v = mg.Selected(m.Path)
			}
			roots = append(roots, module.Version{Path: m.Path, Version: v})
			inRootPaths[m.Path] = true
			if v != m.Version {
				rootsUpgraded = true
			}
		}
		// Note that rs.rootModules was already sorted by module path and version,
		// and we appended to the roots slice in the same order and guaranteed that
		// each path has only one version, so roots is also sorted by module path
		// and (trivially) version.

		if !rootsUpgraded {
			if cfg.BuildMod != "mod" {
				// The only changes to the root set (if any) were to remove duplicates.
				// The requirements are consistent (if perhaps redundant), so keep the
				// original rs to preserve its ModuleGraph.
				return rs, nil
			}
			// The root set has converged: every root going into this iteration was
			// already at its selected version, although we have have removed other
			// (redundant) roots for the same path.
			break
		}
	}

	if rs.pruning == pruned && reflect.DeepEqual(roots, rs.rootModules) && reflect.DeepEqual(direct, rs.direct) {
		// The root set is unchanged and rs was already pruned, so keep rs to
		// preserve its cached ModuleGraph (if any).
		return rs, nil
	}
	return newRequirements(pruned, roots, direct), nil
}

// spotCheckRoots reports whether the versions of the roots in rs satisfy the
// explicit requirements of the modules in mods.
func spotCheckRoots(ctx context.Context, rs *Requirements, mods map[module.Version]bool) bool {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	work := par.NewQueue(runtime.GOMAXPROCS(0))
	for m := range mods {
		m := m
		work.Add(func() {
			if ctx.Err() != nil {
				return
			}

			summary, err := goModSummary(m)
			if err != nil {
				cancel()
				return
			}

			for _, r := range summary.require {
				if v, ok := rs.rootSelected(r.Path); ok && gover.ModCompare(r.Path, v, r.Version) < 0 {
					cancel()
					return
				}
			}
		})
	}
	<-work.Idle()

	if ctx.Err() != nil {
		// Either we failed a spot-check, or the caller no longer cares about our
		// answer anyway.
		return false
	}

	return true
}

// tidyUnprunedRoots returns a minimal set of root requirements that maintains
// the selected version of every module that provided or lexically could have
// provided a package in pkgs, and includes the selected version of every such
// module in direct as a root.
func tidyUnprunedRoots(ctx context.Context, mainModule module.Version, old *Requirements, pkgs []*loadPkg) (*Requirements, error) {
	var (
		// keep is a set of of modules that provide packages or are needed to
		// disambiguate imports.
		keep     []module.Version
		keptPath = map[string]bool{}

		// rootPaths is a list of module paths that provide packages directly
		// imported from the main module. They should be included as roots.
		rootPaths   []string
		inRootPaths = map[string]bool{}

		// altMods is a set of paths of modules that lexically could have provided
		// imported packages. It may be okay to remove these from the list of
		// explicit requirements if that removes them from the module graph. If they
		// are present in the module graph reachable from rootPaths, they must not
		// be at a lower version. That could cause a missing sum error or a new
		// import ambiguity.
		//
		// For example, suppose a developer rewrites imports from example.com/m to
		// example.com/m/v2, then runs 'go mod tidy'. Tidy may delete the
		// requirement on example.com/m if there is no other transitive requirement
		// on it. However, if example.com/m were downgraded to a version not in
		// go.sum, when package example.com/m/v2/p is loaded, we'd get an error
		// trying to disambiguate the import, since we can't check example.com/m
		// without its sum. See #47738.
		altMods = map[string]string{}
	)
	if v, ok := old.rootSelected("go"); ok {
		keep = append(keep, module.Version{Path: "go", Version: v})
		keptPath["go"] = true
	}
	if v, ok := old.rootSelected("toolchain"); ok {
		keep = append(keep, module.Version{Path: "toolchain", Version: v})
		keptPath["toolchain"] = true
	}
	for _, pkg := range pkgs {
		if !pkg.fromExternalModule() {
			continue
		}
		if m := pkg.mod; !keptPath[m.Path] {
			keep = append(keep, m)
			keptPath[m.Path] = true
			if old.direct[m.Path] && !inRootPaths[m.Path] {
				rootPaths = append(rootPaths, m.Path)
				inRootPaths[m.Path] = true
			}
		}
		for _, m := range pkg.altMods {
			altMods[m.Path] = m.Version
		}
	}

	// Construct a build list with a minimal set of roots.
	// This may remove or downgrade modules in altMods.
	reqs := &mvsReqs{roots: keep}
	min, err := mvs.Req(mainModule, rootPaths, reqs)
	if err != nil {
		return nil, err
	}
	buildList, err := mvs.BuildList([]module.Version{mainModule}, reqs)
	if err != nil {
		return nil, err
	}

	// Check if modules in altMods were downgraded but not removed.
	// If so, add them to roots, which will retain an "// indirect" requirement
	// in go.mod. See comment on altMods above.
	keptAltMod := false
	for _, m := range buildList {
		if v, ok := altMods[m.Path]; ok && gover.ModCompare(m.Path, m.Version, v) < 0 {
			keep = append(keep, module.Version{Path: m.Path, Version: v})
			keptAltMod = true
		}
	}
	if keptAltMod {
		// We must run mvs.Req again instead of simply adding altMods to min.
		// It's possible that a requirement in altMods makes some other
		// explicit indirect requirement unnecessary.
		reqs.roots = keep
		min, err = mvs.Req(mainModule, rootPaths, reqs)
		if err != nil {
			return nil, err
		}
	}

	return newRequirements(unpruned, min, old.direct), nil
}

// updateUnprunedRoots returns a set of root requirements that includes the selected
// version of every module path in direct as a root, and maintains the selected
// version of every module selected in the graph of rs.
//
// The roots are updated such that:
//
//  1. The selected version of every module path in direct is included as a root
//     (if it is not "none").
//  2. Each root is the selected version of its path. (We say that such a root
//     set is “consistent”.)
//  3. Every version selected in the graph of rs remains selected unless upgraded
//     by a dependency in add.
//  4. Every version in add is selected at its given version unless upgraded by
//     (the dependencies of) an existing root or another module in add.
func updateUnprunedRoots(ctx context.Context, direct map[string]bool, rs *Requirements, add []module.Version) (*Requirements, error) {
	mg, err := rs.Graph(ctx)
	if err != nil {
		// We can't ignore errors in the module graph even if the user passed the -e
		// flag to try to push past them. If we can't load the complete module
		// dependencies, then we can't reliably compute a minimal subset of them.
		return rs, err
	}

	if mustHaveCompleteRequirements() {
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
	// This is only for convenience and clarity for end users: in an unpruned module,
	// the choice of explicit vs. implicit dependency has no impact on MVS
	// selection (for itself or any other module).
	keep := append(mg.BuildList()[MainModules.Len():], add...)
	for _, m := range keep {
		if direct[m.Path] && !inRootPaths[m.Path] {
			rootPaths = append(rootPaths, m.Path)
			inRootPaths[m.Path] = true
		}
	}

	var roots []module.Version
	for _, mainModule := range MainModules.Versions() {
		min, err := mvs.Req(mainModule, rootPaths, &mvsReqs{roots: keep})
		if err != nil {
			return rs, err
		}
		roots = append(roots, min...)
	}
	if MainModules.Len() > 1 {
		gover.ModSort(roots)
	}
	if rs.pruning == unpruned && reflect.DeepEqual(roots, rs.rootModules) && reflect.DeepEqual(direct, rs.direct) {
		// The root set is unchanged and rs was already unpruned, so keep rs to
		// preserve its cached ModuleGraph (if any).
		return rs, nil
	}

	return newRequirements(unpruned, roots, direct), nil
}

// convertPruning returns a version of rs with the given pruning behavior.
// If rs already has the given pruning, convertPruning returns rs unmodified.
func convertPruning(ctx context.Context, rs *Requirements, pruning modPruning) (*Requirements, error) {
	if rs.pruning == pruning {
		return rs, nil
	} else if rs.pruning == workspace || pruning == workspace {
		panic("attempting to convert to/from workspace pruning and another pruning type")
	}

	if pruning == unpruned {
		// We are converting a pruned module to an unpruned one. The roots of a
		// pruned module graph are a superset of the roots of an unpruned one, so
		// we don't need to add any new roots — we just need to drop the ones that
		// are redundant, which is exactly what updateUnprunedRoots does.
		return updateUnprunedRoots(ctx, rs.direct, rs, nil)
	}

	// We are converting an unpruned module to a pruned one.
	//
	// An unpruned module graph includes the transitive dependencies of every
	// module in the build list. As it turns out, we can express that as a pruned
	// root set! “Include the transitive dependencies of every module in the build
	// list” is exactly what happens in a pruned module if we promote every module
	// in the build list to a root.
	mg, err := rs.Graph(ctx)
	if err != nil {
		return rs, err
	}
	return newRequirements(pruned, mg.BuildList()[MainModules.Len():], rs.direct), nil
}
