// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"sort"

	"cmd/go/internal/mvs"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// editBuildList returns an edited version of initial such that:
//
// 	1. Each module version in mustSelect is selected, unless it is upgraded
// 	   by the transitive requirements of another version in mustSelect.
//
// 	2. Each module version in tryUpgrade is upgraded toward the indicated
// 	   version as far as can be done without violating (1).
//
// 	3. Each module version in initial is downgraded from its original version
// 	   only to the extent needed to satisfy (1), or upgraded only to the extent
// 	   needed to satisfy (1) and (2).
//
// 	4. No module is upgraded above the maximum version of its path found in the
// 	   combined dependency graph of list, tryUpgrade, and mustSelect.
func editBuildList(ctx context.Context, initial, tryUpgrade, mustSelect []module.Version) ([]module.Version, error) {
	// Per https://research.swtch.com/vgo-mvs#algorithm_4:
	// “To avoid an unnecessary downgrade to E 1.1, we must also add a new
	// requirement on E 1.2. We can apply Algorithm R to find the minimal set of
	// new requirements to write to go.mod.”
	//
	// In order to generate those new requirements, we need consider versions for
	// every module in the existing build list, plus every module being directly
	// added by the edit. However, modules added only as dependencies of tentative
	// versions should not be retained if they end up being upgraded or downgraded
	// away due to versions in mustSelect.

	// When we downgrade modules in order to reach mustSelect, we don't want to
	// upgrade any existing module above the version that would be selected if we
	// just added all of the new requirements and *didn't* downgrade.
	//
	// So we'll do exactly that: just add all of the new requirements and not
	// downgrade, and return the resulting versions as an upper bound. This
	// intentionally limits our solution space so that edits that the user
	// percieves as “downgrades” will not also result in upgrades.
	max := make(map[string]string)
	maxes, err := mvs.Upgrade(Target, &mvsReqs{
		buildList: append(capVersionSlice(initial), mustSelect...),
	}, tryUpgrade...)
	if err != nil {
		return nil, err
	}
	for _, m := range maxes {
		max[m.Path] = m.Version
	}
	// The versions in mustSelect override whatever we would naively select —
	// we will downgrade other modules as needed in order to meet them.
	for _, m := range mustSelect {
		max[m.Path] = m.Version
	}

	limiter := newVersionLimiter(max)

	// Force the selected versions in mustSelect, even if they conflict.
	//
	// TODO(bcmills): Instead of forcing these versions, record conflicts
	// so that the caller doesn't have to recompute them.
	for _, m := range mustSelect {
		limiter.selected[m.Path] = m.Version
	}

	// For each module, we want to get as close as we can to either the upgrade
	// version or the previously-selected version in the build list, whichever is
	// higher. We can compute those in either order, but the upgrades will tend to
	// be higher than the build list, so we arbitrarily start with those.
	for _, m := range tryUpgrade {
		if err := limiter.upgradeToward(ctx, m); err != nil {
			return nil, err
		}
	}
	for _, m := range initial {
		if err := limiter.upgradeToward(ctx, m); err != nil {
			return nil, err
		}
	}

	// We've identified acceptable versions for each of the modules, but those
	// versions are not necessarily consistent with each other: one upgraded or
	// downgraded module may require a higher (but still allowed) version of
	// another. The lower version may require extraneous dependencies that aren't
	// actually relevant, so we need to compute the actual selected versions.
	adjusted := make([]module.Version, 0, len(maxes))
	for _, m := range maxes {
		if v, ok := limiter.selected[m.Path]; ok {
			adjusted = append(adjusted, module.Version{Path: m.Path, Version: v})
		}
	}
	consistent, err := mvs.BuildList(Target, &mvsReqs{buildList: adjusted})
	if err != nil {
		return nil, err
	}

	// We have the correct selected versions. Now we need to re-run MVS with only
	// the actually-selected versions in order to eliminate extraneous
	// dependencies from lower-than-selected ones.
	compacted := consistent[:0]
	for _, m := range consistent {
		if _, ok := limiter.selected[m.Path]; ok {
			// The fact that the limiter has a version for m.Path indicates that we
			// care about retaining that path, even if the version was upgraded for
			// consistency.
			compacted = append(compacted, m)
		}
	}

	return mvs.BuildList(Target, &mvsReqs{buildList: compacted})
}

// A versionLimiter tracks the versions that may be selected for each module
// subject to constraints on the maximum versions of transitive dependencies.
type versionLimiter struct {
	// max maps each module path to the maximum version that may be selected for
	// that path. Paths with no entry are unrestricted.
	max map[string]string

	// selected maps each module path to a version of that path (if known) whose
	// transitive dependencies do not violate any max version. The version kept
	// is the highest one found during any call to upgradeToward for the given
	// module path.
	//
	// If a higher acceptable version is found during a call to upgradeToward for
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

	// disqualified maps each encountered version to either true (if that version
	// is known to be disqualified due to a conflict with a max version) or false
	// (if that version is not known to be disqualified, either because it is ok
	// or because we are currently traversing a cycle that includes it).
	disqualified map[module.Version]bool

	// requiredBy maps each not-yet-disqualified module version to the versions
	// that directly require it. If that version becomes disqualified, the
	// disqualification will be propagated to all of the versions in the list.
	requiredBy map[module.Version][]module.Version
}

func newVersionLimiter(max map[string]string) *versionLimiter {
	return &versionLimiter{
		selected:     map[string]string{Target.Path: Target.Version},
		max:          max,
		disqualified: map[module.Version]bool{Target: false},
		requiredBy:   map[module.Version][]module.Version{},
	}
}

// upgradeToward attempts to upgrade the selected version of m.Path as close as
// possible to m.Version without violating l's maximum version limits.
func (l *versionLimiter) upgradeToward(ctx context.Context, m module.Version) error {
	selected, ok := l.selected[m.Path]
	if ok {
		if cmpVersion(selected, m.Version) >= 0 {
			// The selected version is already at least m, so no upgrade is needed.
			return nil
		}
	} else {
		selected = "none"
	}

	if l.isDisqualified(m) {
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

		for l.isDisqualified(m) {
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

// isDisqualified reports whether m (or its transitive dependencies) would
// violate l's maximum version limits if added to the module requirement graph.
func (l *versionLimiter) isDisqualified(m module.Version) bool {
	if m.Version == "none" || m == Target {
		// version "none" has no requirements, and the dependencies of Target are
		// tautological.
		return false
	}

	if dq, seen := l.disqualified[m]; seen {
		return dq
	}
	l.disqualified[m] = false

	if max, ok := l.max[m.Path]; ok && cmpVersion(m.Version, max) > 0 {
		l.disqualify(m)
		return true
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
		l.disqualify(m)
		return true
	}

	for _, r := range summary.require {
		if l.isDisqualified(r) {
			l.disqualify(m)
			return true
		}

		// r and its dependencies are (perhaps provisionally) ok.
		//
		// However, if there are cycles in the requirement graph, we may have only
		// checked a portion of the requirement graph so far, and r (and thus m) may
		// yet be disqualified by some path we have not yet visited. Remember this edge
		// so that we can disqualify m and its dependents if that occurs.
		l.requiredBy[r] = append(l.requiredBy[r], m)
	}

	return false
}

// disqualify records that m (or one of its transitive dependencies)
// violates l's maximum version limits.
func (l *versionLimiter) disqualify(m module.Version) {
	if l.disqualified[m] {
		return
	}
	l.disqualified[m] = true

	for _, p := range l.requiredBy[m] {
		l.disqualify(p)
	}
	// Now that we have disqualified the modules that depend on m, we can forget
	// about them — we won't need to disqualify them again.
	delete(l.requiredBy, m)
}
