// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/mvs"
	"cmd/internal/par"
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"slices"

	"golang.org/x/mod/module"
)

// editRequirements returns an edited version of rs such that:
//
//  1. Each module version in mustSelect is selected.
//
//  2. Each module version in tryUpgrade is upgraded toward the indicated
//     version as far as can be done without violating (1).
//     (Other upgrades are also allowed if they are caused by
//     transitive requirements of versions in mustSelect or
//     tryUpgrade.)
//
//  3. Each module version in rs.rootModules (or rs.graph, if rs is unpruned)
//     is downgraded or upgraded from its original version only to the extent
//     needed to satisfy (1) and (2).
//
// Generally, the module versions in mustSelect are due to the module or a
// package within the module matching an explicit command line argument to 'go
// get', and the versions in tryUpgrade are transitive dependencies that are
// either being upgraded by 'go get -u' or being added to satisfy some
// otherwise-missing package import.
//
// If pruning is enabled, the roots of the edited requirements include an
// explicit entry for each module path in tryUpgrade, mustSelect, and the roots
// of rs, unless the selected version for the module path is "none".
func editRequirements(ctx context.Context, rs *Requirements, tryUpgrade, mustSelect []module.Version) (edited *Requirements, changed bool, err error) {
	if rs.pruning == workspace {
		panic("editRequirements cannot edit workspace requirements")
	}

	orig := rs
	// If we already know what go version we will end up on after the edit, and
	// the pruning for that version is different, go ahead and apply it now.
	//
	// If we are changing from pruned to unpruned, then we MUST check the unpruned
	// graph for conflicts from the start. (Checking only for pruned conflicts
	// would miss some that would be introduced later.)
	//
	// If we are changing from unpruned to pruned, then we would like to avoid
	// unnecessary downgrades due to conflicts that would be pruned out of the
	// final graph anyway.
	//
	// Note that even if we don't find a go version in mustSelect, it is possible
	// that we will switch from unpruned to pruned (but not the other way around!)
	// after applying the edits if we find a dependency that requires a high
	// enough go version to trigger an upgrade.
	rootPruning := orig.pruning
	for _, m := range mustSelect {
		if m.Path == "go" {
			rootPruning = pruningForGoVersion(m.Version)
			break
		} else if m.Path == "toolchain" && pruningForGoVersion(gover.FromToolchain(m.Version)) == unpruned {
			// We don't know exactly what go version we will end up at, but we know
			// that it must be a version supported by the requested toolchain, and
			// that toolchain does not support pruning.
			//
			// TODO(bcmills): 'go get' ought to reject explicit toolchain versions
			// older than gover.GoStrictVersion. Once that is fixed, is this still
			// needed?
			rootPruning = unpruned
			break
		}
	}

	if rootPruning != rs.pruning {
		rs, err = convertPruning(ctx, rs, rootPruning)
		if err != nil {
			return orig, false, err
		}
	}

	// selectedRoot records the edited version (possibly "none") for each module
	// path that would be a root in the edited requirements.
	var selectedRoot map[string]string // module path → edited version
	if rootPruning == pruned {
		selectedRoot = maps.Clone(rs.maxRootVersion)
	} else {
		// In a module without graph pruning, modules that provide packages imported
		// by the main module may either be explicit roots or implicit transitive
		// dependencies. To the extent possible, we want to preserve those implicit
		// dependencies, so we need to treat everything in the build list as
		// potentially relevant — that is, as what would be a “root” in a module
		// with graph pruning enabled.
		mg, err := rs.Graph(ctx)
		if err != nil {
			// If we couldn't load the graph, we don't know what its requirements were
			// to begin with, so we can't edit those requirements in a coherent way.
			return orig, false, err
		}
		bl := mg.BuildList()[MainModules.Len():]
		selectedRoot = make(map[string]string, len(bl))
		for _, m := range bl {
			selectedRoot[m.Path] = m.Version
		}
	}

	for _, r := range tryUpgrade {
		if v, ok := selectedRoot[r.Path]; ok && gover.ModCompare(r.Path, v, r.Version) >= 0 {
			continue
		}
		if cfg.BuildV {
			fmt.Fprintf(os.Stderr, "go: trying upgrade to %v\n", r)
		}
		selectedRoot[r.Path] = r.Version
	}

	// conflicts is a list of conflicts that we cannot resolve without violating
	// some version in mustSelect. It may be incomplete, but we want to report
	// as many conflicts as we can so that the user can solve more of them at once.
	var conflicts []Conflict

	// mustSelectVersion is an index of the versions in mustSelect.
	mustSelectVersion := make(map[string]string, len(mustSelect))
	for _, r := range mustSelect {
		if v, ok := mustSelectVersion[r.Path]; ok && v != r.Version {
			prev := module.Version{Path: r.Path, Version: v}
			if gover.ModCompare(r.Path, v, r.Version) > 0 {
				conflicts = append(conflicts, Conflict{Path: []module.Version{prev}, Constraint: r})
			} else {
				conflicts = append(conflicts, Conflict{Path: []module.Version{r}, Constraint: prev})
			}
			continue
		}

		mustSelectVersion[r.Path] = r.Version
		selectedRoot[r.Path] = r.Version
	}

	// We've indexed all of the data we need and we've computed the initial
	// versions of the roots. Now we need to load the actual module graph and
	// restore the invariant that every root is the selected version of its path.
	//
	// For 'go mod tidy' we would do that using expandGraph, which upgrades the
	// roots until their requirements are internally consistent and then drops out
	// the old roots. However, here we need to do more: we also need to make sure
	// the modules in mustSelect don't get upgraded above their intended versions.
	// To do that, we repeatedly walk the module graph, identify paths of
	// requirements that result in versions that are too high, and downgrade the
	// roots that lead to those paths. When no conflicts remain, we're done.
	//
	// Since we want to report accurate paths to each conflict, we don't drop out
	// older-than-selected roots until the process completes. That might mean that
	// we do some extra downgrades when they could be skipped, but for the benefit
	// of being able to explain the reason for every downgrade that seems
	// worthwhile.
	//
	// Graph pruning adds an extra wrinkle: a given node in the module graph
	// may be reached from a root whose dependencies are pruned, and from a root
	// whose dependencies are not pruned. It may be the case that the path from
	// the unpruned root leads to a conflict, while the path from the pruned root
	// prunes out the requirements that would lead to that conflict.
	// So we need to track the two kinds of paths independently.
	// They join back together at the roots of the graph: if a root r1 with pruned
	// requirements depends on a root r2 with unpruned requirements, then
	// selecting r1 would cause r2 to become a root and pull in all of its
	// unpruned dependencies.
	//
	// The dqTracker type implements the logic for propagating conflict paths
	// through the pruned and unpruned parts of the module graph.
	//
	// We make a best effort to fix incompatibilities, subject to two properties:
	//
	// 	1. If the user runs 'go get' with a set of mutually-compatible module
	// 	versions, we should accept those versions.
	//
	// 	2. If we end up upgrading or downgrading a module, it should be
	// 	clear why we did so.
	//
	// We don't try to find an optimal SAT solution,
	// especially given the complex interactions with graph pruning.

	var (
		roots      []module.Version // the current versions in selectedRoot, in sorted order
		rootsDirty = true           // true if roots does not match selectedRoot
	)

	// rejectedRoot records the set of module versions that have been disqualified
	// as roots of the module graph. When downgrading due to a conflict or error,
	// we skip any version that has already been rejected.
	//
	// NOTE(bcmills): I am not sure that the rejectedRoot map is really necessary,
	// since we normally only downgrade roots or accept indirect upgrades to
	// known-good versions. However, I am having trouble proving that accepting an
	// indirect upgrade never introduces a conflict that leads to further
	// downgrades. I really want to be able to prove that editRequirements
	// terminates, and the easiest way to prove it is to add this map.
	//
	// Then the proof of termination is this:
	// On every iteration where we mark the roots as dirty, we add some new module
	// version to the map. The universe of module versions is finite, so we must
	// eventually reach a state in which we do not add any version to the map.
	// In that state, we either report a conflict or succeed in the edit.
	rejectedRoot := map[module.Version]bool{}

	for rootsDirty && len(conflicts) == 0 {
		roots = roots[:0]
		for p, v := range selectedRoot {
			if v != "none" {
				roots = append(roots, module.Version{Path: p, Version: v})
			}
		}
		gover.ModSort(roots)

		// First, we extend the graph so that it includes the selected version
		// of every root. The upgraded roots are in addition to the original
		// roots, so we will have enough information to trace a path to each
		// conflict we discover from one or more of the original roots.
		mg, upgradedRoots, err := extendGraph(ctx, rootPruning, roots, selectedRoot)
		if err != nil {
			var tooNew *gover.TooNewError
			if mg == nil || errors.As(err, &tooNew) {
				return orig, false, err
			}
			// We're about to walk the entire extended module graph, so we will find
			// any error then — and we will either try to resolve it by downgrading
			// something or report it as a conflict with more detail.
		}

		// extendedRootPruning is an index of the pruning used to load each root in
		// the extended module graph.
		extendedRootPruning := make(map[module.Version]modPruning, len(roots)+len(upgradedRoots))
		findPruning := func(m module.Version) modPruning {
			if rootPruning == pruned {
				summary, _ := mg.loadCache.Get(m)
				if summary != nil && summary.pruning == unpruned {
					return unpruned
				}
			}
			return rootPruning
		}
		for _, m := range roots {
			extendedRootPruning[m] = findPruning(m)
		}
		for m := range upgradedRoots {
			extendedRootPruning[m] = findPruning(m)
		}

		// Now check the resulting extended graph for errors and incompatibilities.
		t := dqTracker{extendedRootPruning: extendedRootPruning}
		mg.g.WalkBreadthFirst(func(m module.Version) {
			if max, ok := mustSelectVersion[m.Path]; ok && gover.ModCompare(m.Path, m.Version, max) > 0 {
				// m itself violates mustSelect, so it cannot appear in the module graph
				// even if its transitive dependencies would be pruned out.
				t.disqualify(m, pruned, dqState{dep: m})
				return
			}

			summary, err := mg.loadCache.Get(m)
			if err != nil && err != par.ErrCacheEntryNotFound {
				// We can't determine the requirements of m, so we don't know whether
				// they would be allowed. This may be a transient error reaching the
				// repository, rather than a permanent error with the retrieved version.
				//
				// TODO(golang.org/issue/31730, golang.org/issue/30134):
				// decide what to do based on the actual error.
				t.disqualify(m, pruned, dqState{err: err})
				return
			}

			reqs, ok := mg.RequiredBy(m)
			if !ok {
				// The dependencies of m do not appear in the module graph, so they
				// can't be causing any problems this time.
				return
			}

			if summary == nil {
				if m.Version != "" {
					panic(fmt.Sprintf("internal error: %d reqs present for %v, but summary is nil", len(reqs), m))
				}
				// m is the main module: we are editing its dependencies, so it cannot
				// become disqualified.
				return
			}

			// Before we check for problems due to transitive dependencies, first
			// check m's direct requirements. A requirement on a version r that
			// violates mustSelect disqualifies m, even if the requirements of r are
			// themselves pruned out.
			for _, r := range reqs {
				if max, ok := mustSelectVersion[r.Path]; ok && gover.ModCompare(r.Path, r.Version, max) > 0 {
					t.disqualify(m, pruned, dqState{dep: r})
					return
				}
			}
			for _, r := range reqs {
				if !t.require(m, r) {
					break
				}
			}
		})

		// We have now marked all of the versions in the graph that have conflicts,
		// with a path to each conflict from one or more roots that introduce it.
		// Now we need to identify those roots and change their versions
		// (if possible) in order to resolve the conflicts.
		rootsDirty = false
		for _, m := range roots {
			path, err := t.path(m, extendedRootPruning[m])
			if len(path) == 0 && err == nil {
				continue // Nothing wrong with m; we can keep it.
			}

			// path leads to a module with a problem: either it violates a constraint,
			// or some error prevents us from determining whether it violates a
			// constraint. We might end up logging or returning the conflict
			// information, so go ahead and fill in the details about it.
			conflict := Conflict{
				Path: path,
				Err:  err,
			}
			if err == nil {
				var last module.Version = path[len(path)-1]
				mustV, ok := mustSelectVersion[last.Path]
				if !ok {
					fmt.Fprintf(os.Stderr, "go: %v\n", conflict)
					panic("internal error: found a version conflict, but no constraint it violates")
				}
				conflict.Constraint = module.Version{
					Path:    last.Path,
					Version: mustV,
				}
			}

			if v, ok := mustSelectVersion[m.Path]; ok && v == m.Version {
				// m is in mustSelect, but is marked as disqualified due to a transitive
				// dependency.
				//
				// In theory we could try removing module paths that don't appear in
				// mustSelect (added by tryUpgrade or already present in rs) in order to
				// get graph pruning to take effect, but (a) it is likely that 'go mod
				// tidy' would re-add those roots and reintroduce unwanted upgrades,
				// causing confusion, and (b) deciding which roots to try to eliminate
				// would add a lot of complexity.
				//
				// Instead, we report the path to the conflict as an error.
				// If users want to explicitly prune out nodes from the dependency
				// graph, they can always add an explicit 'exclude' directive.
				conflicts = append(conflicts, conflict)
				continue
			}

			// If m is not the selected version of its path, we have two options: we
			// can either upgrade to the version that actually is selected (dropping m
			// itself out of the bottom of the module graph), or we can try
			// downgrading it.
			//
			// If the version we would be upgrading to is ok to use, we will just plan
			// to do that and avoid the overhead of trying to find some lower version
			// to downgrade to.
			//
			// However, it is possible that m depends on something that leads to its
			// own upgrade, so if the upgrade isn't viable we should go ahead and try
			// to downgrade (like with any other root).
			if v := mg.Selected(m.Path); v != m.Version {
				u := module.Version{Path: m.Path, Version: v}
				uPruning, ok := t.extendedRootPruning[m]
				if !ok {
					fmt.Fprintf(os.Stderr, "go: %v\n", conflict)
					panic(fmt.Sprintf("internal error: selected version of root %v is %v, but it was not expanded as a new root", m, u))
				}
				if !t.check(u, uPruning).isDisqualified() && !rejectedRoot[u] {
					// Applying the upgrade from m to u will resolve the conflict,
					// so plan to do that if there are no other conflicts to resolve.
					continue
				}
			}

			// Figure out what version of m's path was present before we started
			// the edit. We want to make sure we consider keeping it as-is,
			// even if it wouldn't normally be included. (For example, it might
			// be a pseudo-version or pre-release.)
			origMG, _ := orig.Graph(ctx)
			origV := origMG.Selected(m.Path)

			if conflict.Err != nil && origV == m.Version {
				// This version of m.Path was already in the module graph before we
				// started editing, and the problem with it is that we can't load its
				// (transitive) requirements.
				//
				// If this conflict was just one step in a longer chain of downgrades,
				// then we would want to keep going past it until we find a version
				// that doesn't have that problem. However, we only want to downgrade
				// away from an *existing* requirement if we can confirm that it actually
				// conflicts with mustSelect. (For example, we don't want
				// 'go get -u ./...' to incidentally downgrade some dependency whose
				// go.mod file is unavailable or has a bad checksum.)
				conflicts = append(conflicts, conflict)
				continue
			}

			// We need to downgrade m's path to some lower version to try to resolve
			// the conflict. Find the next-lowest candidate and apply it.
			rejectedRoot[m] = true
			prev := m
			for {
				prev, err = previousVersion(ctx, prev)
				if gover.ModCompare(m.Path, m.Version, origV) > 0 && (gover.ModCompare(m.Path, prev.Version, origV) < 0 || err != nil) {
					// previousVersion skipped over origV. Insert it into the order.
					prev.Version = origV
				} else if err != nil {
					// We don't know the next downgrade to try. Give up.
					return orig, false, err
				}
				if rejectedRoot[prev] {
					// We already rejected prev in a previous round.
					// To ensure that this algorithm terminates, don't try it again.
					continue
				}
				pruning := rootPruning
				if pruning == pruned {
					if summary, err := mg.loadCache.Get(m); err == nil {
						pruning = summary.pruning
					}
				}
				if t.check(prev, pruning).isDisqualified() {
					// We found a problem with prev this round that would also disqualify
					// it as a root. Don't bother trying it next round.
					rejectedRoot[prev] = true
					continue
				}
				break
			}
			selectedRoot[m.Path] = prev.Version
			rootsDirty = true

			// If this downgrade is potentially interesting, log the reason for it.
			if conflict.Err != nil || cfg.BuildV {
				var action string
				if prev.Version == "none" {
					action = fmt.Sprintf("removing %s", m)
				} else if prev.Version == origV {
					action = fmt.Sprintf("restoring %s", prev)
				} else {
					action = fmt.Sprintf("trying %s", prev)
				}
				fmt.Fprintf(os.Stderr, "go: %s\n\t%s\n", conflict.Summary(), action)
			}
		}
		if rootsDirty {
			continue
		}

		// We didn't resolve any issues by downgrading, but we may still need to
		// resolve some conflicts by locking in upgrades. Do that now.
		//
		// We don't do these upgrades until we're done downgrading because the
		// downgrade process might reveal or remove conflicts (by changing which
		// requirement edges are pruned out).
		var upgradedFrom []module.Version // for logging only
		for p, v := range selectedRoot {
			if _, ok := mustSelectVersion[p]; !ok {
				if actual := mg.Selected(p); actual != v {
					if cfg.BuildV {
						upgradedFrom = append(upgradedFrom, module.Version{Path: p, Version: v})
					}
					selectedRoot[p] = actual
					// Accepting the upgrade to m.Path might cause the selected versions
					// of other modules to fall, because they were being increased by
					// dependencies of m that are no longer present in the graph.
					//
					// TODO(bcmills): Can removing m as a root also cause the selected
					// versions of other modules to rise? I think not: we're strictly
					// removing non-root nodes from the module graph, which can't cause
					// any root to decrease (because they're roots), and the dependencies
					// of non-roots don't matter because they're either always unpruned or
					// always pruned out.
					//
					// At any rate, it shouldn't cost much to reload the module graph one
					// last time and confirm that it is stable.
					rootsDirty = true
				}
			}
		}
		if rootsDirty {
			if cfg.BuildV {
				gover.ModSort(upgradedFrom) // Make logging deterministic.
				for _, m := range upgradedFrom {
					fmt.Fprintf(os.Stderr, "go: accepting indirect upgrade from %v to %s\n", m, selectedRoot[m.Path])
				}
			}
			continue
		}
		break
	}
	if len(conflicts) > 0 {
		return orig, false, &ConstraintError{Conflicts: conflicts}
	}

	if rootPruning == unpruned {
		// An unpruned go.mod file lists only a subset of the requirements needed
		// for building packages. Figure out which requirements need to be explicit.
		var rootPaths []string

		// The modules in mustSelect are always promoted to be explicit.
		for _, m := range mustSelect {
			if m.Version != "none" && !MainModules.Contains(m.Path) {
				rootPaths = append(rootPaths, m.Path)
			}
		}

		for _, m := range roots {
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

		roots, err = mvs.Req(MainModules.mustGetSingleMainModule(), rootPaths, &mvsReqs{roots: roots})
		if err != nil {
			return nil, false, err
		}
	}

	changed = rootPruning != orig.pruning || !slices.Equal(roots, orig.rootModules)
	if !changed {
		// Because the roots we just computed are unchanged, the entire graph must
		// be the same as it was before. Save the original rs, since we have
		// probably already loaded its requirement graph.
		return orig, false, nil
	}

	// A module that is not even in the build list necessarily cannot provide
	// any imported packages. Mark as direct only the direct modules that are
	// still in the build list. (We assume that any module path that provided a
	// direct import before the edit continues to do so after. There are a few
	// edge cases where that can change, such as if a package moves into or out of
	// a nested module or disappears entirely. If that happens, the user can run
	// 'go mod tidy' to clean up the direct/indirect annotations.)
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
	edited = newRequirements(rootPruning, roots, direct)

	// If we ended up adding a dependency that upgrades our go version far enough
	// to activate pruning, we must convert the edited Requirements in order to
	// avoid dropping transitive dependencies from the build list the next time
	// someone uses the updated go.mod file.
	//
	// Note that it isn't possible to go in the other direction (from pruned to
	// unpruned) unless the "go" or "toolchain" module is explicitly listed in
	// mustSelect, which we already handled at the very beginning of the edit.
	// That is because the virtual "go" module only requires a "toolchain",
	// and the "toolchain" module never requires anything else, which means that
	// those two modules will never be downgraded due to a conflict with any other
	// constraint.
	if rootPruning == unpruned {
		if v, ok := edited.rootSelected("go"); ok && pruningForGoVersion(v) == pruned {
			// Since we computed the edit with the unpruned graph, and the pruned
			// graph is a strict subset of the unpruned graph, this conversion
			// preserves the exact (edited) build list that we already computed.
			//
			// However, it does that by shoving the whole build list into the roots of
			// the graph. 'go get' will check for that sort of transition and log a
			// message reminding the user how to clean up this mess we're about to
			// make. 😅
			edited, err = convertPruning(ctx, edited, pruned)
			if err != nil {
				return orig, false, err
			}
		}
	}
	return edited, true, nil
}

// extendGraph loads the module graph from roots, and iteratively extends it by
// unpruning the selected version of each module path that is a root in rs or in
// the roots slice until the graph reaches a fixed point.
//
// The graph is guaranteed to converge to a fixed point because unpruning a
// module version can only increase (never decrease) the selected versions,
// and the set of versions for each module is finite.
//
// The extended graph is useful for diagnosing version conflicts: for each
// selected module version, it can provide a complete path of requirements from
// some root to that version.
func extendGraph(ctx context.Context, rootPruning modPruning, roots []module.Version, selectedRoot map[string]string) (mg *ModuleGraph, upgradedRoot map[module.Version]bool, err error) {
	for {
		mg, err = readModGraph(ctx, rootPruning, roots, upgradedRoot)
		// We keep on going even if err is non-nil until we reach a steady state.
		// (Note that readModGraph returns a non-nil *ModuleGraph even in case of
		// errors.) The caller may be able to fix the errors by adjusting versions,
		// so we really want to return as complete a result as we can.

		if rootPruning == unpruned {
			// Everything is already unpruned, so there isn't anything we can do to
			// extend it further.
			break
		}

		nPrevRoots := len(upgradedRoot)
		for p := range selectedRoot {
			// Since p is a root path, when we fix up the module graph to be
			// consistent with the selected versions, p will be promoted to a root,
			// which will pull in its dependencies. Ensure that its dependencies are
			// included in the module graph.
			v := mg.g.Selected(p)
			if v == "none" {
				// Version “none” always has no requirements, so it doesn't need
				// an explicit node in the module graph.
				continue
			}
			m := module.Version{Path: p, Version: v}
			if _, ok := mg.g.RequiredBy(m); !ok && !upgradedRoot[m] {
				// The dependencies of the selected version of p were not loaded.
				// Mark it as an upgrade so that we will load its dependencies
				// in the next iteration.
				//
				// Note that we don't remove any of the existing roots, even if they are
				// no longer the selected version: with graph pruning in effect this may
				// leave some spurious dependencies in the graph, but it at least
				// preserves enough of the graph to explain why each upgrade occurred:
				// this way, we can report a complete path from the passed-in roots
				// to every node in the module graph.
				//
				// This process is guaranteed to reach a fixed point: since we are only
				// adding roots (never removing them), the selected version of each module
				// can only increase, never decrease, and the set of module versions in the
				// universe is finite.
				if upgradedRoot == nil {
					upgradedRoot = make(map[module.Version]bool)
				}
				upgradedRoot[m] = true
			}
		}
		if len(upgradedRoot) == nPrevRoots {
			break
		}
	}

	return mg, upgradedRoot, err
}

type perPruning[T any] struct {
	pruned   T
	unpruned T
}

func (pp perPruning[T]) from(p modPruning) T {
	if p == unpruned {
		return pp.unpruned
	}
	return pp.pruned
}

// A dqTracker tracks and propagates the reason that each module version
// cannot be included in the module graph.
type dqTracker struct {
	// extendedRootPruning is the modPruning given the go.mod file for each root
	// in the extended module graph.
	extendedRootPruning map[module.Version]modPruning

	// dqReason records whether and why each each encountered version is
	// disqualified in a pruned or unpruned context.
	dqReason map[module.Version]perPruning[dqState]

	// requiring maps each not-yet-disqualified module version to the versions
	// that would cause that module's requirements to be included in a pruned or
	// unpruned context. If that version becomes disqualified, the
	// disqualification will be propagated to all of the versions in the
	// corresponding list.
	//
	// This map is similar to the module requirement graph, but includes more
	// detail about whether a given dependency edge appears in a pruned or
	// unpruned context. (Other commands do not need this level of detail.)
	requiring map[module.Version][]module.Version
}

// A dqState indicates whether and why a module version is “disqualified” from
// being used in a way that would incorporate its requirements.
//
// The zero dqState indicates that the module version is not known to be
// disqualified, either because it is ok or because we are currently traversing
// a cycle that includes it.
type dqState struct {
	err error          // if non-nil, disqualified because the requirements of the module could not be read
	dep module.Version // disqualified because the module is or requires dep
}

func (dq dqState) isDisqualified() bool {
	return dq != dqState{}
}

func (dq dqState) String() string {
	if dq.err != nil {
		return dq.err.Error()
	}
	if dq.dep != (module.Version{}) {
		return dq.dep.String()
	}
	return "(no conflict)"
}

// require records that m directly requires r, in case r becomes disqualified.
// (These edges are in the opposite direction from the edges in an mvs.Graph.)
//
// If r is already disqualified, require propagates the disqualification to m
// and returns the reason for the disqualification.
func (t *dqTracker) require(m, r module.Version) (ok bool) {
	rdq := t.dqReason[r]
	rootPruning, isRoot := t.extendedRootPruning[r]
	if isRoot && rdq.from(rootPruning).isDisqualified() {
		// When we pull in m's dependencies, we will have an edge from m to r, and r
		// is disqualified (it is a root, which causes its problematic dependencies
		// to always be included). So we cannot pull in m's dependencies at all:
		// m is completely disqualified.
		t.disqualify(m, pruned, dqState{dep: r})
		return false
	}

	if dq := rdq.from(unpruned); dq.isDisqualified() {
		t.disqualify(m, unpruned, dqState{dep: r})
		if _, ok := t.extendedRootPruning[m]; !ok {
			// Since m is not a root, its dependencies can't be included in the pruned
			// part of the module graph, and will never be disqualified from a pruned
			// reason. We've already disqualified everything that matters.
			return false
		}
	}

	// Record that m is a dependant of r, so that if r is later disqualified
	// m will be disqualified as well.
	if t.requiring == nil {
		t.requiring = make(map[module.Version][]module.Version)
	}
	t.requiring[r] = append(t.requiring[r], m)
	return true
}

// disqualify records why the dependencies of m cannot be included in the module
// graph if reached from a part of the graph with the given pruning.
//
// Since the pruned graph is a subgraph of the unpruned graph, disqualifying a
// module from a pruned part of the graph also disqualifies it in the unpruned
// parts.
func (t *dqTracker) disqualify(m module.Version, fromPruning modPruning, reason dqState) {
	if !reason.isDisqualified() {
		panic("internal error: disqualify called with a non-disqualifying dqState")
	}

	dq := t.dqReason[m]
	if dq.from(fromPruning).isDisqualified() {
		return // Already disqualified for some other reason; don't overwrite it.
	}
	rootPruning, isRoot := t.extendedRootPruning[m]
	if fromPruning == pruned {
		dq.pruned = reason
		if !dq.unpruned.isDisqualified() {
			// Since the pruned graph of m is a subgraph of the unpruned graph, if it
			// is disqualified due to something in the pruned graph, it is certainly
			// disqualified in the unpruned graph from the same reason.
			dq.unpruned = reason
		}
	} else {
		dq.unpruned = reason
		if dq.pruned.isDisqualified() {
			panic(fmt.Sprintf("internal error: %v is marked as disqualified when pruned, but not when unpruned", m))
		}
		if isRoot && rootPruning == unpruned {
			// Since m is a root that is always unpruned, any other roots — even
			// pruned ones! — that cause it to be selected would also cause the reason
			// for is disqualification to be included in the module graph.
			dq.pruned = reason
		}
	}
	if t.dqReason == nil {
		t.dqReason = make(map[module.Version]perPruning[dqState])
	}
	t.dqReason[m] = dq

	if isRoot && (fromPruning == pruned || rootPruning == unpruned) {
		// Either m is disqualified even when its dependencies are pruned,
		// or m's go.mod file causes its dependencies to *always* be unpruned.
		// Everything that depends on it must be disqualified.
		for _, p := range t.requiring[m] {
			t.disqualify(p, pruned, dqState{dep: m})
			// Note that since the pruned graph is a subset of the unpruned graph,
			// disqualifying p in the pruned graph also disqualifies it in the
			// unpruned graph.
		}
		// Everything in t.requiring[m] is now fully disqualified.
		// We won't need to use it again.
		delete(t.requiring, m)
		return
	}

	// Either m is not a root, or it is a pruned root but only being disqualified
	// when reached from the unpruned parts of the module graph.
	// Either way, the reason for this disqualification is only visible to the
	// unpruned parts of the module graph.
	for _, p := range t.requiring[m] {
		t.disqualify(p, unpruned, dqState{dep: m})
	}
	if !isRoot {
		// Since m is not a root, its dependencies can't be included in the pruned
		// part of the module graph, and will never be disqualified from a pruned
		// reason. We've already disqualified everything that matters.
		delete(t.requiring, m)
	}
}

// check reports whether m is disqualified in the given pruning context.
func (t *dqTracker) check(m module.Version, pruning modPruning) dqState {
	return t.dqReason[m].from(pruning)
}

// path returns the path from m to the reason it is disqualified, which may be
// either a module that violates constraints or an error in loading
// requirements.
//
// If m is not disqualified, path returns (nil, nil).
func (t *dqTracker) path(m module.Version, pruning modPruning) (path []module.Version, err error) {
	for {
		if rootPruning, isRoot := t.extendedRootPruning[m]; isRoot && rootPruning == unpruned {
			// Since m is a root, any other module that requires it would cause
			// its full unpruned dependencies to be included in the module graph.
			// Those dependencies must also be considered as part of the path to the conflict.
			pruning = unpruned
		}
		dq := t.dqReason[m].from(pruning)
		if !dq.isDisqualified() {
			return path, nil
		}
		path = append(path, m)
		if dq.err != nil || dq.dep == m {
			return path, dq.err // m itself is the conflict.
		}
		m = dq.dep
	}
}
