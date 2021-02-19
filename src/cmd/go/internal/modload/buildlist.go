// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
	"cmd/go/internal/mvs"
	"context"
	"fmt"
	"os"
	"strings"

	"golang.org/x/mod/module"
)

// buildList is the list of modules to use for building packages.
// It is initialized by calling LoadPackages or ImportFromFiles,
// each of which uses loaded.load.
//
// Ideally, exactly ONE of those functions would be called,
// and exactly once. Most of the time, that's true.
// During "go get" it may not be. TODO(rsc): Figure out if
// that restriction can be established, or else document why not.
//
var buildList []module.Version

// additionalExplicitRequirements is a list of modules paths for which
// WriteGoMod should record explicit requirements, even if they would be
// selected without those requirements. Each path must also appear in buildList.
var additionalExplicitRequirements []string

// capVersionSlice returns s with its cap reduced to its length.
func capVersionSlice(s []module.Version) []module.Version {
	return s[:len(s):len(s)]
}

// LoadAllModules loads and returns the list of modules matching the "all"
// module pattern, starting with the Target module and in a deterministic
// (stable) order, without loading any packages.
//
// Modules are loaded automatically (and lazily) in LoadPackages:
// LoadAllModules need only be called if LoadPackages is not,
// typically in commands that care about modules but no particular package.
//
// The caller must not modify the returned list, but may append to it.
func LoadAllModules(ctx context.Context) []module.Version {
	LoadModFile(ctx)
	ReloadBuildList()
	WriteGoMod()
	return capVersionSlice(buildList)
}

// Selected returns the selected version of the module with the given path, or
// the empty string if the given module has no selected version
// (either because it is not required or because it is the Target module).
func Selected(path string) (version string) {
	if path == Target.Path {
		return ""
	}
	for _, m := range buildList {
		if m.Path == path {
			return m.Version
		}
	}
	return ""
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
func EditBuildList(ctx context.Context, add, mustSelect []module.Version) error {
	var upgraded = capVersionSlice(buildList)
	if len(add) > 0 {
		// First, upgrade the build list with any additions.
		// In theory we could just append the additions to the build list and let
		// mvs.Downgrade take care of resolving the upgrades too, but the
		// diagnostics from Upgrade are currently much better in case of errors.
		var err error
		upgraded, err = mvs.Upgrade(Target, &mvsReqs{buildList: upgraded}, add...)
		if err != nil {
			return err
		}
	}

	downgraded, err := mvs.Downgrade(Target, &mvsReqs{buildList: append(upgraded, mustSelect...)}, mustSelect...)
	if err != nil {
		return err
	}

	final, err := mvs.Upgrade(Target, &mvsReqs{buildList: downgraded}, mustSelect...)
	if err != nil {
		return err
	}

	selected := make(map[string]module.Version, len(final))
	for _, m := range final {
		selected[m.Path] = m
	}
	inconsistent := false
	for _, m := range mustSelect {
		s, ok := selected[m.Path]
		if !ok {
			if m.Version != "none" {
				panic(fmt.Sprintf("internal error: mvs.BuildList lost %v", m))
			}
			continue
		}
		if s.Version != m.Version {
			inconsistent = true
			break
		}
	}

	if !inconsistent {
		buildList = final
		additionalExplicitRequirements = make([]string, 0, len(mustSelect))
		for _, m := range mustSelect {
			if m.Version != "none" {
				additionalExplicitRequirements = append(additionalExplicitRequirements, m.Path)
			}
		}
		return nil
	}

	// We overshot one or more of the modules in mustSelected, which means that
	// Downgrade removed something in mustSelect because it conflicted with
	// something else in mustSelect.
	//
	// Walk the requirement graph to find the conflict.
	//
	// TODO(bcmills): Ideally, mvs.Downgrade (or a replacement for it) would do
	// this directly.

	reqs := &mvsReqs{buildList: final}
	reason := map[module.Version]module.Version{}
	for _, m := range mustSelect {
		reason[m] = m
	}
	queue := mustSelect[:len(mustSelect):len(mustSelect)]
	for len(queue) > 0 {
		var m module.Version
		m, queue = queue[0], queue[1:]
		required, err := reqs.Required(m)
		if err != nil {
			return err
		}
		for _, r := range required {
			if _, ok := reason[r]; !ok {
				reason[r] = reason[m]
				queue = append(queue, r)
			}
		}
	}

	var conflicts []Conflict
	for _, m := range mustSelect {
		s, ok := selected[m.Path]
		if !ok {
			if m.Version != "none" {
				panic(fmt.Sprintf("internal error: mvs.BuildList lost %v", m))
			}
			continue
		}
		if s.Version != m.Version {
			conflicts = append(conflicts, Conflict{
				Source:     reason[s],
				Dep:        s,
				Constraint: m,
			})
		}
	}

	return &ConstraintError{
		Conflicts: conflicts,
	}
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

// ReloadBuildList resets the state of loaded packages, then loads and returns
// the build list set by EditBuildList.
func ReloadBuildList() []module.Version {
	loaded = loadFromRoots(loaderParams{
		PackageOpts: PackageOpts{
			Tags: imports.Tags(),
		},
		listRoots:          func() []string { return nil },
		allClosesOverTests: index.allPatternClosesOverTests(), // but doesn't matter because the root list is empty.
	})
	return capVersionSlice(buildList)
}

// TidyBuildList trims the build list to the minimal requirements needed to
// retain the same versions of all packages from the preceding call to
// LoadPackages.
func TidyBuildList() {
	used := map[module.Version]bool{Target: true}
	for _, pkg := range loaded.pkgs {
		used[pkg.mod] = true
	}

	keep := []module.Version{Target}
	var direct []string
	for _, m := range buildList[1:] {
		if used[m] {
			keep = append(keep, m)
			if loaded.direct[m.Path] {
				direct = append(direct, m.Path)
			}
		} else if cfg.BuildV {
			if _, ok := index.require[m]; ok {
				fmt.Fprintf(os.Stderr, "unused %s\n", m.Path)
			}
		}
	}

	min, err := mvs.Req(Target, direct, &mvsReqs{buildList: keep})
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	buildList = append([]module.Version{Target}, min...)
}

// checkMultiplePaths verifies that a given module path is used as itself
// or as a replacement for another module, but not both at the same time.
//
// (See https://golang.org/issue/26607 and https://golang.org/issue/34650.)
func checkMultiplePaths() {
	firstPath := make(map[module.Version]string, len(buildList))
	for _, mod := range buildList {
		src := mod
		if rep := Replacement(mod); rep.Path != "" {
			src = rep
		}
		if prev, ok := firstPath[src]; !ok {
			firstPath[src] = mod.Path
		} else if prev != mod.Path {
			base.Errorf("go: %s@%s used for two different module paths (%s and %s)", src.Path, src.Version, prev, mod.Path)
		}
	}
	base.ExitIfErrors()
}
