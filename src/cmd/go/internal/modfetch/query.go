// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
	"fmt"
	"strings"
)

// Query looks up a revision of a given module given a version query string.
// The module must be a complete module path.
// The version must take one of the following forms:
//
//	- the literal string "latest", denoting the latest available tagged version
//	- v1.2.3, a semantic version string
//	- v1 or v1.2, an abbreviated semantic version string completed by adding zeroes (v1.0.0 or v1.2.0);
//	- >v1.2.3, denoting the earliest available version after v1.2.3
//	- <v1.2.3, denoting the latest available version before v1.2.3
//	- an RFC 3339 time stamp, denoting the latest available version at that time
//	- a Unix time expressed as seconds since 1970, denoting the latest available version at that time
//	- a repository commit identifier, denoting that version
//
// The time stamps can be followed by an optional @branch suffix to limit the
// result to revisions on a particular branch name.
//
func Query(path, vers string, allowed func(module.Version) bool) (*RevInfo, error) {
	repo, err := Lookup(path)
	if err != nil {
		return nil, err
	}

	if strings.HasPrefix(vers, "v") && semver.IsValid(vers) {
		// TODO: This turns query for "v2" into Stat "v2.0.0",
		// but probably it should allow checking for a branch named "v2".
		vers = semver.Canonical(vers)
		if allowed != nil && !allowed(module.Version{Path: path, Version: vers}) {
			return nil, fmt.Errorf("%s@%s excluded", path, vers)
		}
		return repo.Stat(vers)
	}
	if strings.HasPrefix(vers, ">") || strings.HasPrefix(vers, "<") || vers == "latest" {
		var op string
		if vers != "latest" {
			if !semver.IsValid(vers[1:]) {
				return nil, fmt.Errorf("invalid semantic version in range %s", vers)
			}
			op, vers = vers[:1], vers[1:]
		}
		versions, err := repo.Versions("")
		if err != nil {
			return nil, err
		}
		if len(versions) == 0 && vers == "latest" {
			return repo.Latest()
		}
		if vers == "latest" {
			for i := len(versions) - 1; i >= 0; i-- {
				if allowed == nil || allowed(module.Version{Path: path, Version: versions[i]}) {
					return repo.Stat(versions[i])
				}
			}
		} else if op == "<" {
			for i := len(versions) - 1; i >= 0; i-- {
				if semver.Compare(versions[i], vers) < 0 && (allowed == nil || allowed(module.Version{Path: path, Version: versions[i]})) {
					return repo.Stat(versions[i])
				}
			}
		} else {
			for i := 0; i < len(versions); i++ {
				if semver.Compare(versions[i], vers) > 0 && (allowed == nil || allowed(module.Version{Path: path, Version: versions[i]})) {
					return repo.Stat(versions[i])
				}
			}
		}
		return nil, fmt.Errorf("no matching versions for %s%s", op, vers)
	}
	// TODO: Time queries, maybe.

	return repo.Stat(vers)
}
