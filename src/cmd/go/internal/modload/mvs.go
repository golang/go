// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"os"
	"sort"

	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"

	"golang.org/x/mod/module"
)

// cmpVersion implements the comparison for versions in the module loader.
//
// It is consistent with gover.ModCompare except that as a special case,
// the version "" is considered higher than all other versions.
// The main module (also known as the target) has no version and must be chosen
// over other versions of the same module in the module dependency graph.
func cmpVersion(p string, v1, v2 string) int {
	if v2 == "" {
		if v1 == "" {
			return 0
		}
		return -1
	}
	if v1 == "" {
		return 1
	}
	return gover.ModCompare(p, v1, v2)
}

// mvsReqs implements mvs.Reqs for module semantic versions,
// with any exclusions or replacements applied internally.
type mvsReqs struct {
	loaderstate *State // TODO(jitsu): Is there a way we can not depend on the entire loader state?
	roots       []module.Version
}

func (r *mvsReqs) Required(mod module.Version) ([]module.Version, error) {
	if mod.Version == "" && r.loaderstate.MainModules.Contains(mod.Path) {
		// Use the build list as it existed when r was constructed, not the current
		// global build list.
		return r.roots, nil
	}

	if mod.Version == "none" {
		return nil, nil
	}

	summary, err := goModSummary(r.loaderstate, mod)
	if err != nil {
		return nil, err
	}
	return summary.require, nil
}

// Max returns the maximum of v1 and v2 according to gover.ModCompare.
//
// As a special case, the version "" is considered higher than all other
// versions. The main module (also known as the target) has no version and must
// be chosen over other versions of the same module in the module dependency
// graph.
func (*mvsReqs) Max(p, v1, v2 string) string {
	if cmpVersion(p, v1, v2) < 0 {
		return v2
	}
	return v1
}

// Upgrade is a no-op, here to implement mvs.Reqs.
// The upgrade logic for go get -u is in ../modget/get.go.
func (*mvsReqs) Upgrade(m module.Version) (module.Version, error) {
	return m, nil
}

func versions(loaderstate *State, ctx context.Context, path string, allowed AllowedFunc) (versions []string, origin *codehost.Origin, err error) {
	// Note: modfetch.Lookup and repo.Versions are cached,
	// so there's no need for us to add extra caching here.
	err = modfetch.TryProxies(func(proxy string) error {
		repo, err := lookupRepo(loaderstate, ctx, proxy, path)
		if err != nil {
			return err
		}
		allVersions, err := repo.Versions(ctx, "")
		if err != nil {
			return err
		}
		allowedVersions := make([]string, 0, len(allVersions.List))
		for _, v := range allVersions.List {
			if err := allowed(ctx, module.Version{Path: path, Version: v}); err == nil {
				allowedVersions = append(allowedVersions, v)
			} else if !errors.Is(err, ErrDisallowed) {
				return err
			}
		}
		versions = allowedVersions
		origin = allVersions.Origin
		return nil
	})
	return versions, origin, err
}

// previousVersion returns the tagged version of m.Path immediately prior to
// m.Version, or version "none" if no prior version is tagged.
//
// Since the version of a main module is not found in the version list,
// it has no previous version.
func previousVersion(loaderstate *State, ctx context.Context, m module.Version) (module.Version, error) {
	if m.Version == "" && loaderstate.MainModules.Contains(m.Path) {
		return module.Version{Path: m.Path, Version: "none"}, nil
	}

	list, _, err := versions(loaderstate, ctx, m.Path, loaderstate.CheckAllowed)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return module.Version{Path: m.Path, Version: "none"}, nil
		}
		return module.Version{}, err
	}
	i := sort.Search(len(list), func(i int) bool { return gover.ModCompare(m.Path, list[i], m.Version) >= 0 })
	if i > 0 {
		return module.Version{Path: m.Path, Version: list[i-1]}, nil
	}
	return module.Version{Path: m.Path, Version: "none"}, nil
}

func (r *mvsReqs) Previous(m module.Version) (module.Version, error) {
	// TODO(golang.org/issue/38714): thread tracing context through MVS.
	return previousVersion(r.loaderstate, context.TODO(), m)
}
