// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"os"
	"sort"

	"cmd/go/internal/modfetch"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// mvsReqs implements mvs.Reqs for module semantic versions,
// with any exclusions or replacements applied internally.
type mvsReqs struct {
	buildList []module.Version
}

func (r *mvsReqs) Required(mod module.Version) ([]module.Version, error) {
	if mod == Target {
		// Use the build list as it existed when r was constructed, not the current
		// global build list.
		return r.buildList[1:], nil
	}

	if mod.Version == "none" {
		return nil, nil
	}

	summary, err := goModSummary(mod)
	if err != nil {
		return nil, err
	}
	return summary.require, nil
}

// Max returns the maximum of v1 and v2 according to semver.Compare.
//
// As a special case, the version "" is considered higher than all other
// versions. The main module (also known as the target) has no version and must
// be chosen over other versions of the same module in the module dependency
// graph.
func (*mvsReqs) Max(v1, v2 string) string {
	if v1 != "" && (v2 == "" || semver.Compare(v1, v2) == -1) {
		return v2
	}
	return v1
}

// Upgrade is a no-op, here to implement mvs.Reqs.
// The upgrade logic for go get -u is in ../modget/get.go.
func (*mvsReqs) Upgrade(m module.Version) (module.Version, error) {
	return m, nil
}

func versions(ctx context.Context, path string, allowed AllowedFunc) ([]string, error) {
	// Note: modfetch.Lookup and repo.Versions are cached,
	// so there's no need for us to add extra caching here.
	var versions []string
	err := modfetch.TryProxies(func(proxy string) error {
		repo, err := lookupRepo(proxy, path)
		if err != nil {
			return err
		}
		allVersions, err := repo.Versions("")
		if err != nil {
			return err
		}
		allowedVersions := make([]string, 0, len(allVersions))
		for _, v := range allVersions {
			if err := allowed(ctx, module.Version{Path: path, Version: v}); err == nil {
				allowedVersions = append(allowedVersions, v)
			} else if !errors.Is(err, ErrDisallowed) {
				return err
			}
		}
		versions = allowedVersions
		return nil
	})
	return versions, err
}

// Previous returns the tagged version of m.Path immediately prior to
// m.Version, or version "none" if no prior version is tagged.
//
// Since the version of Target is not found in the version list,
// it has no previous version.
func (*mvsReqs) Previous(m module.Version) (module.Version, error) {
	// TODO(golang.org/issue/38714): thread tracing context through MVS.

	if m == Target {
		return module.Version{Path: m.Path, Version: "none"}, nil
	}

	list, err := versions(context.TODO(), m.Path, CheckAllowed)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return module.Version{Path: m.Path, Version: "none"}, nil
		}
		return module.Version{}, err
	}
	i := sort.Search(len(list), func(i int) bool { return semver.Compare(list[i], m.Version) >= 0 })
	if i > 0 {
		return module.Version{Path: m.Path, Version: list[i-1]}, nil
	}
	return module.Version{Path: m.Path, Version: "none"}, nil
}
