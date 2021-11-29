// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"cmd/go/internal/mvs"
	"context"
	"reflect"
	"sort"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// editRequirements returns an edited version of rs such that:
//
// 	1. Each module version in mustSelect is selected.
//
// 	2. Each module version in tryUpgrade is upgraded toward the indicated
// 	   version as far as can be done without violating (1).
//
// 	3. Each module version in rs.rootModules (or rs.graph, if rs is unpruned)
// 	   is downgraded from its original version only to the extent needed to
// 	   satisfy (1), or upgraded only to the extent needed to satisfy (1) and
// 	   (2).
//
// 	4. No module is upgraded above the maximum version of its path found in the
// 	   dependency graph of rs, the combined dependency graph of the versions in
// 	   mustSelect, or the dependencies of each individual module version in
// 	   tryUpgrade.
//
// Generally, the module versions in mustSelect are due to the module or a
// package within the module matching an explicit command line argument to 'go
// get', and the versions in tryUpgrade are transitive dependencies that are
// either being upgraded by 'go get -u' or being added to satisfy some
// otherwise-missing package import.
func editRequirements(ctx context.Context, rs *Requirements, tryUpgrade, mustSelect []module.Version) (edited *Requirements, changed bool, err error) {
	limiter, err := limiterForEdit(ctx, rs, tryUpgrade, mustSelect)
	if err != nil {
		return rs, false, err
	}

	var conflicts []Conflict
	for _, m := range mustSelect {
		conflict, err := limiter.Select(m)
		if err != nil {
			return rs, false, err
		}
		if conflict.Path != "" {
			conflicts = append(conflicts, Conflict{
				Source: m,
				Dep:    conflict,
				Constraint: module.Version{
					Path:    conflict.Path,
					Version: limiter.max[conflict.Path],
				},
			})
		}
	}
	if len(conflicts) > 0 {
		return rs, false, &ConstraintError{Conflicts: conflicts}
	}

	mods, changed, err := selectPotentiallyImportedModules(ctx, limiter, rs, tryUpgrade)
	if err != nil {
		return rs, false, err
	}

	var roots []module.Version
	if rs.pruning == unpruned {
		// In a module without graph pruning, modules that provide packages imported
		// by the main module may either be explicit roots or implicit transitive
		// dependencies. We promote the modules in mustSelect to be explicit
		// requirements.
		var rootPaths []string
		for _, m := range mustSelect {
			if m.Version != "none" && !MainModules.Contains(m.Path) {
				rootPaths = append(rootPaths, m.Path)
			}
		}
		if !changed && len(rootPaths) == 0 {
			// The build list hasn't changed and we have no new roots to add.
			// We don't need to recompute the minimal roots for the module.
			return rs, false, nil
		}

		for _, m := range mods {
			if v, ok := rs.rootSelected(m.Path); ok && (v == m.Version || rs.direct[m.Path]) {
				// m.Path was formerly a root, and either its version hasn't changed or
				// we believe that it provides a package directly imported by a package
				// or test in the main module. For now we'll assume that it is still
				// relevant enough to remain a root. If we actually load all of the
				// packages and tests in the main module (which we are not doing here),
				// we can revise the explicit roots at that point.
				rootPaths = append(rootPaths, m.Path)
			}
		}

		roots, err = mvs.Req(MainModules.mustGetSingleMainModule(), rootPaths, &mvsReqs{roots: mods})
		if err != nil {
			return nil, false, err
		}
	} else {
		// In a module with a pruned graph, every module that provides a package
		// imported by the main module must be retained as a root.
		roots = mods
		if !changed {
			// Because the roots we just computed are unchanged, the entire graph must
			// be the same as it was before. Save the original rs, since we have
			// probably already loaded its requirement graph.
			return rs, false, nil
		}
	}

	// A module that is not even in the build list necessarily cannot provide
	// any imported packages. Mark as direct only the direct modules that are
	// still in the build list.
	//
	// TODO(bcmills): Would it make more sense to leave the direct map as-is
	// but allow it to refer to modules that are no longer in the build list?
	// That might complicate updateRoots, but it may be cleaner in other ways.
	direct := make(map[string]bool, len(rs.direct))
	for _, m := range roots {
		if rs.direct[m.Path] {
			direct[m.Path] = true
		}
	}
	return newRequirements(rs.pruning, roots, direct), changed, nil
}

// limiterForEdit returns a versionLimiter with its max versions set such that
// the max version for every module path in mustSelect is the version listed
// there, and the max version for every other module path is the maximum version
// of its path found in the dependency graph of rs, the combined dependency
// graph of the versions in mustSelect, or the dependencies of each individual
// module version in tryUpgrade.
func limiterForEdit(ctx context.Context, rs *Requirements, tryUpgrade, mustSelect []module.Version) (*versionLimiter, error) {
	mg, err := rs.Graph(ctx)
	if err != nil {
		return nil, err
	}

	maxVersion := map[string]string{} // module path → version
	restrictTo := func(m module.Version) {
		v, ok := maxVersion[m.Path]
		if !ok || cmpVersion(v, m.Version) > 0 {
			maxVersion[m.Path] = m.Version
		}
	}

	if rs.pruning == unpruned {
		// go.mod files that do not support graph pruning don't indicate which
		// transitive dependencies are actually relevant to the main module, so we
		// have to assume that any module that could have provided any package —
		// that is, any module whose selected version was not "none" — may be
		// relevant.
		for _, m := range mg.BuildList() {
			restrictTo(m)
		}
	} else {
		// The go.mod file explicitly records every module that provides a package
		// imported by the main module.
		//
		// If we need to downgrade an existing root or a new root found in
		// tryUpgrade, we don't want to allow that downgrade to incidentally upgrade
		// a module imported by the main module to some arbitrary version.
		// However, we don't particularly care about arbitrary upgrades to modules
		// that are (at best) only providing packages imported by tests of
		// dependencies outside the main module.
		for _, m := range rs.rootModules {
			restrictTo(module.Version{
				Path:    m.Path,
				Version: mg.Selected(m.Path),
			})
		}
	}

	if err := raiseLimitsForUpgrades(ctx, maxVersion, rs.pruning, tryUpgrade, mustSelect); err != nil {
		return nil, err
	}

	// The versions in mustSelect override whatever we would naively select —
	// we will downgrade other modules as needed in order to meet them.
	for _, m := range mustSelect {
		restrictTo(m)
	}

	return newVersionLimiter(rs.pruning, maxVersion), nil
}

// raiseLimitsForUpgrades increases the module versions in maxVersions to the
// versions that would be needed to allow each of the modules in tryUpgrade
// (individually or in any combination) and all of the modules in mustSelect
// (simultaneously) to be added as roots.
//
// Versions not present in maxVersion are unrestricted, and it is assumed that
// they will not be promoted to root requirements (and thus will not contribute
// their own dependencies if the main module supports graph pruning).
//
// These limits provide an upper bound on how far a module may be upgraded as
// part of an incidental downgrade, if downgrades are needed in order to select
// the versions in mustSelect.
func raiseLimitsForUpgrades(ctx context.Context, maxVersion map[string]string, pruning modPruning, tryUpgrade []module.Version, mustSelect []module.Version) error {
	// allow raises the limit for m.Path to at least m.Version.
	// If m.Path was already unrestricted, it remains unrestricted.
	allow := func(m module.Version) {
		v, ok := maxVersion[m.Path]
		if !ok {
			return // m.Path is unrestricted.
		}
		if cmpVersion(v, m.Version) < 0 {
			maxVersion[m.Path] = m.Version
		}
	}

	var (
		unprunedUpgrades []module.Version
		isPrunedRootPath map[string]bool
	)
	if pruning == unpruned {
		unprunedUpgrades = tryUpgrade
	} else {
		isPrunedRootPath = make(map[string]bool, len(maxVersion))
		for p := range maxVersion {
			isPrunedRootPath[p] = true
		}
		for _, m := range tryUpgrade {
			isPrunedRootPath[m.Path] = true
		}
		for _, m := range mustSelect {
			isPrunedRootPath[m.Path] = true
		}

		allowedRoot := map[module.Version]bool{}

		var allowRoot func(m module.Version) error
		allowRoot = func(m module.Version) error {
			if allowedRoot[m] {
				return nil
			}
			allowedRoot[m] = true

			if MainModules.Contains(m.Path) {
				// The main module versions are already considered to be higher than any
				// possible m, so m cannot be selected as a root and there is no point
				// scanning its dependencies.
				return nil
			}

			allow(m)

			summary, err := goModSummary(m)
			if err != nil {
				return err
			}
			if summary.pruning == unpruned {
				// For efficiency, we'll load all of the unpruned upgrades as one big
				// graph, rather than loading the (potentially-overlapping) subgraph for
				// each upgrade individually.
				unprunedUpgrades = append(unprunedUpgrades, m)
				return nil
			}
			for _, r := range summary.require {
				if isPrunedRootPath[r.Path] {
					// r could become a root as the result of an upgrade or downgrade,
					// in which case its dependencies will not be pruned out.
					// We need to allow those dependencies to be upgraded too.
					if err := allowRoot(r); err != nil {
						return err
					}
				} else {
					// r will not become a root, so its dependencies don't matter.
					// Allow only r itself.
					allow(r)
				}
			}
			return nil
		}

		for _, m := range tryUpgrade {
			allowRoot(m)
		}
	}

	if len(unprunedUpgrades) > 0 {
		// Compute the max versions for unpruned upgrades all together.
		// Since these modules are unpruned, we'll end up scanning all of their
		// transitive dependencies no matter which versions end up selected,
		// and since we have a large dependency graph to scan we might get
		// a significant benefit from not revisiting dependencies that are at
		// common versions among multiple upgrades.
		upgradeGraph, err := readModGraph(ctx, unpruned, unprunedUpgrades)
		if err != nil {
			// Compute the requirement path from a module path in tryUpgrade to the
			// error, and the requirement path (if any) from rs.rootModules to the
			// tryUpgrade module path. Return a *mvs.BuildListError showing the
			// concatenation of the paths (with an upgrade in the middle).
			return err
		}

		for _, r := range upgradeGraph.BuildList() {
			// Upgrading to m would upgrade to r, and the caller requested that we
			// try to upgrade to m, so it's ok to upgrade to r.
			allow(r)
		}
	}

	// Explicitly allow any (transitive) upgrades implied by mustSelect.
	nextRoots := append([]module.Version(nil), mustSelect...)
	for nextRoots != nil {
		module.Sort(nextRoots)
		rs := newRequirements(pruning, nextRoots, nil)
		nextRoots = nil

		rs, mustGraph, err := expandGraph(ctx, rs)
		if err != nil {
			return err
		}

		for _, r := range mustGraph.BuildList() {
			// Some module in mustSelect requires r, so we must allow at least
			// r.Version (unless it conflicts with another entry in mustSelect, in
			// which case we will error out either way).
			allow(r)

			if isPrunedRootPath[r.Path] {
				if v, ok := rs.rootSelected(r.Path); ok && r.Version == v {
					// r is already a root, so its requirements are already included in
					// the build list.
					continue
				}

				// The dependencies in mustSelect may upgrade (or downgrade) an existing
				// root to match r, which will remain as a root. However, since r is not
				// a root of rs, its dependencies have been pruned out of this build
				// list. We need to add it back explicitly so that we allow any
				// transitive upgrades that r will pull in.
				if nextRoots == nil {
					nextRoots = rs.rootModules // already capped
				}
				nextRoots = append(nextRoots, r)
			}
		}
	}

	return nil
}

// selectPotentiallyImportedModules increases the limiter-selected version of
// every module in rs that potentially provides a package imported (directly or
// indirectly) by the main module, and every module in tryUpgrade, toward the
// highest version seen in rs or tryUpgrade, but not above the maximums enforced
// by the limiter.
//
// It returns the list of module versions selected by the limiter, sorted by
// path, along with a boolean indicating whether that list is different from the
// list of modules read from rs.
func selectPotentiallyImportedModules(ctx context.Context, limiter *versionLimiter, rs *Requirements, tryUpgrade []module.Version) (mods []module.Version, changed bool, err error) {
	for _, m := range tryUpgrade {
		if err := limiter.UpgradeToward(ctx, m); err != nil {
			return nil, false, err
		}
	}

	var initial []module.Version
	if rs.pruning == unpruned {
		mg, err := rs.Graph(ctx)
		if err != nil {
			return nil, false, err
		}
		initial = mg.BuildList()[MainModules.Len():]
	} else {
		initial = rs.rootModules
	}
	for _, m := range initial {
		if err := limiter.UpgradeToward(ctx, m); err != nil {
			return nil, false, err
		}
	}

	mods = make([]module.Version, 0, len(limiter.selected))
	for path, v := range limiter.selected {
		if v != "none" && !MainModules.Contains(path) {
			mods = append(mods, module.Version{Path: path, Version: v})
		}
	}

	// We've identified acceptable versions for each of the modules, but those
	// versions are not necessarily consistent with each other: one upgraded or
	// downgraded module may require a higher (but still allowed) version of
	// another. The lower version may require extraneous dependencies that aren't
	// actually relevant, so we need to compute the actual selected versions.
	mg, err := readModGraph(ctx, rs.pruning, mods)
	if err != nil {
		return nil, false, err
	}
	mods = make([]module.Version, 0, len(limiter.selected))
	for path, _ := range limiter.selected {
		if !MainModules.Contains(path) {
			if v := mg.Selected(path); v != "none" {
				mods = append(mods, module.Version{Path: path, Version: v})
			}
		}
	}
	module.Sort(mods)

	changed = !reflect.DeepEqual(mods, initial)

	return mods, changed, err
}

// A versionLimiter tracks the versions that may be selected for each module
// subject to constraints on the maximum versions of transitive dependencies.
type versionLimiter struct {
	// pruning is the pruning at which the dependencies of the modules passed to
	// Select and UpgradeToward are loaded.
	pruning modPruning

	// max maps each module path to the maximum version that may be selected for
	// that path.
	//
	// Paths with no entry are unrestricted, and we assume that they will not be
	// promoted to root dependencies (so will not contribute dependencies if the
	// main module supports graph pruning).
	max map[string]string

	// selected maps each module path to a version of that path (if known) whose
	// transitive dependencies do not violate any max version. The version kept
	// is the highest one found during any call to UpgradeToward for the given
	// module path.
	//
	// If a higher acceptable version is found during a call to UpgradeToward for
	// some *other* module path, that does not update the selected version.
	// Ignoring those versions keeps the downgrades computed for two modules
	// together close to the individual downgrades that would be computed for each
	// module in isolation. (The only way one module can affect another is if the
	// final downgraded version of the one module explicitly requires a higher
	// version of the other.)
	//
	// Version "none" of every module is always known not to violate any max
	// version, so paths at version "none" are omitted.
	selected map[string]string

	// dqReason records whether and why each each encountered version is
	// disqualified.
	dqReason map[module.Version]dqState

	// requiring maps each not-yet-disqualified module version to the versions
	// that directly require it. If that version becomes disqualified, the
	// disqualification will be propagated to all of the versions in the list.
	requiring map[module.Version][]module.Version
}

// A dqState indicates whether and why a module version is “disqualified” from
// being used in a way that would incorporate its requirements.
//
// The zero dqState indicates that the module version is not known to be
// disqualified, either because it is ok or because we are currently traversing
// a cycle that includes it.
type dqState struct {
	err      error          // if non-nil, disqualified because the requirements of the module could not be read
	conflict module.Version // disqualified because the module (transitively) requires dep, which exceeds the maximum version constraint for its path
}

func (dq dqState) isDisqualified() bool {
	return dq != dqState{}
}

// newVersionLimiter returns a versionLimiter that restricts the module paths
// that appear as keys in max.
//
// max maps each module path to its maximum version; paths that are not present
// in the map are unrestricted. The limiter assumes that unrestricted paths will
// not be promoted to root dependencies.
//
// If module graph pruning is in effect, then if a module passed to
// UpgradeToward or Select supports pruning, its unrestricted dependencies are
// skipped when scanning requirements.
func newVersionLimiter(pruning modPruning, max map[string]string) *versionLimiter {
	selected := make(map[string]string)
	for _, m := range MainModules.Versions() {
		selected[m.Path] = m.Version
	}
	return &versionLimiter{
		pruning:   pruning,
		max:       max,
		selected:  selected,
		dqReason:  map[module.Version]dqState{},
		requiring: map[module.Version][]module.Version{},
	}
}

// UpgradeToward attempts to upgrade the selected version of m.Path as close as
// possible to m.Version without violating l's maximum version limits.
//
// If module graph pruning is in effect and m itself supports pruning, the
// dependencies of unrestricted dependencies of m will not be followed.
func (l *versionLimiter) UpgradeToward(ctx context.Context, m module.Version) error {
	selected, ok := l.selected[m.Path]
	if ok {
		if cmpVersion(selected, m.Version) >= 0 {
			// The selected version is already at least m, so no upgrade is needed.
			return nil
		}
	} else {
		selected = "none"
	}

	if l.check(m, l.pruning).isDisqualified() {
		candidates, err := versions(ctx, m.Path, CheckAllowed)
		if err != nil {
			// This is likely a transient error reaching the repository,
			// rather than a permanent error with the retrieved version.
			//
			// TODO(golang.org/issue/31730, golang.org/issue/30134):
			// decode what to do based on the actual error.
			return err
		}

		// Skip to candidates < m.Version.
		i := sort.Search(len(candidates), func(i int) bool {
			return semver.Compare(candidates[i], m.Version) >= 0
		})
		candidates = candidates[:i]

		for l.check(m, l.pruning).isDisqualified() {
			n := len(candidates)
			if n == 0 || cmpVersion(selected, candidates[n-1]) >= 0 {
				// We couldn't find a suitable candidate above the already-selected version.
				// Retain that version unmodified.
				return nil
			}
			m.Version, candidates = candidates[n-1], candidates[:n-1]
		}
	}

	l.selected[m.Path] = m.Version
	return nil
}

// Select attempts to set the selected version of m.Path to exactly m.Version.
func (l *versionLimiter) Select(m module.Version) (conflict module.Version, err error) {
	dq := l.check(m, l.pruning)
	if !dq.isDisqualified() {
		l.selected[m.Path] = m.Version
	}
	return dq.conflict, dq.err
}

// check determines whether m (or its transitive dependencies) would violate l's
// maximum version limits if added to the module requirement graph.
//
// If pruning is in effect and m itself supports graph pruning, the dependencies
// of unrestricted dependencies of m will not be followed. If the graph-pruning
// invariants hold for the main module up to this point, the packages in those
// modules are at best only imported by tests of dependencies that are
// themselves loaded from outside modules. Although we would like to keep
// 'go test all' as reproducible as is feasible, we don't want to retain test
// dependencies that are only marginally relevant at best.
func (l *versionLimiter) check(m module.Version, pruning modPruning) dqState {
	if m.Version == "none" || m == MainModules.mustGetSingleMainModule() {
		// version "none" has no requirements, and the dependencies of Target are
		// tautological.
		return dqState{}
	}

	if dq, seen := l.dqReason[m]; seen {
		return dq
	}
	l.dqReason[m] = dqState{}

	if max, ok := l.max[m.Path]; ok && cmpVersion(m.Version, max) > 0 {
		return l.disqualify(m, dqState{conflict: m})
	}

	summary, err := goModSummary(m)
	if err != nil {
		// If we can't load the requirements, we couldn't load the go.mod file.
		// There are a number of reasons this can happen, but this usually
		// means an older version of the module had a missing or invalid
		// go.mod file. For example, if example.com/mod released v2.0.0 before
		// migrating to modules (v2.0.0+incompatible), then added a valid go.mod
		// in v2.0.1, downgrading from v2.0.1 would cause this error.
		//
		// TODO(golang.org/issue/31730, golang.org/issue/30134): if the error
		// is transient (we couldn't download go.mod), return the error from
		// Downgrade. Currently, we can't tell what kind of error it is.
		return l.disqualify(m, dqState{err: err})
	}

	if summary.pruning == unpruned {
		pruning = unpruned
	}
	for _, r := range summary.require {
		if pruning == pruned {
			if _, restricted := l.max[r.Path]; !restricted {
				// r.Path is unrestricted, so we don't care at what version it is
				// selected. We assume that r.Path will not become a root dependency, so
				// since m supports pruning, r's dependencies won't be followed.
				continue
			}
		}

		if dq := l.check(r, pruning); dq.isDisqualified() {
			return l.disqualify(m, dq)
		}

		// r and its dependencies are (perhaps provisionally) ok.
		//
		// However, if there are cycles in the requirement graph, we may have only
		// checked a portion of the requirement graph so far, and r (and thus m) may
		// yet be disqualified by some path we have not yet visited. Remember this edge
		// so that we can disqualify m and its dependents if that occurs.
		l.requiring[r] = append(l.requiring[r], m)
	}

	return dqState{}
}

// disqualify records that m (or one of its transitive dependencies)
// violates l's maximum version limits.
func (l *versionLimiter) disqualify(m module.Version, dq dqState) dqState {
	if dq := l.dqReason[m]; dq.isDisqualified() {
		return dq
	}
	l.dqReason[m] = dq

	for _, p := range l.requiring[m] {
		l.disqualify(p, dqState{conflict: m})
	}
	// Now that we have disqualified the modules that depend on m, we can forget
	// about them — we won't need to disqualify them again.
	delete(l.requiring, m)
	return dq
}
