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

	"golang.org/x/mod/module"
)

// buildList is the list of modules to use for building packages.
// It is initialized by calling ImportPaths, ImportFromFiles,
// LoadALL, or LoadBuildList, each of which uses loaded.load.
//
// Ideally, exactly ONE of those functions would be called,
// and exactly once. Most of the time, that's true.
// During "go get" it may not be. TODO(rsc): Figure out if
// that restriction can be established, or else document why not.
//
var buildList []module.Version

// LoadBuildList loads and returns the build list from go.mod.
// The loading of the build list happens automatically in ImportPaths:
// LoadBuildList need only be called if ImportPaths is not
// (typically in commands that care about the module but
// no particular package).
func LoadBuildList(ctx context.Context) []module.Version {
	InitMod(ctx)
	ReloadBuildList()
	WriteGoMod()
	return buildList
}

// ReloadBuildList resets the state of loaded packages, then loads and returns
// the build list set in SetBuildList.
func ReloadBuildList() []module.Version {
	loaded = loadFromRoots(loaderParams{
		tags:               imports.Tags(),
		listRoots:          func() []string { return nil },
		allClosesOverTests: index.allPatternClosesOverTests(), // but doesn't matter because the root list is empty.
	})
	return buildList
}

// BuildList returns the module build list,
// typically constructed by a previous call to
// LoadBuildList or ImportPaths.
// The caller must not modify the returned list.
func BuildList() []module.Version {
	return buildList
}

// SetBuildList sets the module build list.
// The caller is responsible for ensuring that the list is valid.
// SetBuildList does not retain a reference to the original list.
func SetBuildList(list []module.Version) {
	buildList = append([]module.Version{}, list...)
}

// TidyBuildList trims the build list to the minimal requirements needed to
// retain the same versions of all packages from the preceding Load* or
// ImportPaths* call.
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
