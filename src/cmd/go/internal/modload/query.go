// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"cmd/go/internal/modfetch"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
	"fmt"
	"strings"
)

// Query looks up a revision of a given module given a version query string.
// The module must be a complete module path.
// The version must take one of the following forms:
//
//	- the literal string "latest", denoting the latest available, allowed tagged version,
//	  with non-prereleases preferred over prereleases.
//	  If there are no tagged versions in the repo, latest returns the most recent commit.
//	- v1, denoting the latest available tagged version v1.x.x.
//	- v1.2, denoting the latest available tagged version v1.2.x.
//	- v1.2.3, a semantic version string denoting that tagged version.
//	- <v1.2.3, <=v1.2.3, >v1.2.3, >=v1.2.3,
//	   denoting the version closest to the target and satisfying the given operator,
//	   with non-prereleases preferred over prereleases.
//	- a repository commit identifier, denoting that commit.
//
// If the allowed function is non-nil, Query excludes any versions for which allowed returns false.
//
func Query(path, query string, allowed func(module.Version) bool) (*modfetch.RevInfo, error) {
	if allowed == nil {
		allowed = func(module.Version) bool { return true }
	}

	// Parse query to detect parse errors (and possibly handle query)
	// before any network I/O.
	badVersion := func(v string) (*modfetch.RevInfo, error) {
		return nil, fmt.Errorf("invalid semantic version %q in range %q", v, query)
	}
	var ok func(module.Version) bool
	var preferOlder bool
	switch {
	case query == "latest":
		ok = allowed

	case strings.HasPrefix(query, "<="):
		v := query[len("<="):]
		if !semver.IsValid(v) {
			return badVersion(v)
		}
		if isSemverPrefix(v) {
			// Refuse to say whether <=v1.2 allows v1.2.3 (remember, @v1.2 might mean v1.2.3).
			return nil, fmt.Errorf("ambiguous semantic version %q in range %q", v, query)
		}
		ok = func(m module.Version) bool {
			return semver.Compare(m.Version, v) <= 0 && allowed(m)
		}

	case strings.HasPrefix(query, "<"):
		v := query[len("<"):]
		if !semver.IsValid(v) {
			return badVersion(v)
		}
		ok = func(m module.Version) bool {
			return semver.Compare(m.Version, v) < 0 && allowed(m)
		}

	case strings.HasPrefix(query, ">="):
		v := query[len(">="):]
		if !semver.IsValid(v) {
			return badVersion(v)
		}
		ok = func(m module.Version) bool {
			return semver.Compare(m.Version, v) >= 0 && allowed(m)
		}
		preferOlder = true

	case strings.HasPrefix(query, ">"):
		v := query[len(">"):]
		if !semver.IsValid(v) {
			return badVersion(v)
		}
		if isSemverPrefix(v) {
			// Refuse to say whether >v1.2 allows v1.2.3 (remember, @v1.2 might mean v1.2.3).
			return nil, fmt.Errorf("ambiguous semantic version %q in range %q", v, query)
		}
		ok = func(m module.Version) bool {
			return semver.Compare(m.Version, v) > 0 && allowed(m)
		}
		preferOlder = true

	case semver.IsValid(query) && isSemverPrefix(query):
		ok = func(m module.Version) bool {
			return matchSemverPrefix(query, m.Version) && allowed(m)
		}

	case semver.IsValid(query):
		vers := semver.Canonical(query)
		if !allowed(module.Version{Path: path, Version: vers}) {
			return nil, fmt.Errorf("%s@%s excluded", path, vers)
		}
		return modfetch.Stat(path, vers)

	default:
		// Direct lookup of semantic version or commit identifier.
		info, err := modfetch.Stat(path, query)
		if err != nil {
			return nil, err
		}
		if !allowed(module.Version{Path: path, Version: info.Version}) {
			return nil, fmt.Errorf("%s@%s excluded", path, info.Version)
		}
		return info, nil
	}

	// Load versions and execute query.
	repo, err := modfetch.Lookup(path)
	if err != nil {
		return nil, err
	}
	versions, err := repo.Versions("")
	if err != nil {
		return nil, err
	}

	if preferOlder {
		for _, v := range versions {
			if semver.Prerelease(v) == "" && ok(module.Version{Path: path, Version: v}) {
				return repo.Stat(v)
			}
		}
		for _, v := range versions {
			if semver.Prerelease(v) != "" && ok(module.Version{Path: path, Version: v}) {
				return repo.Stat(v)
			}
		}
	} else {
		for i := len(versions) - 1; i >= 0; i-- {
			v := versions[i]
			if semver.Prerelease(v) == "" && ok(module.Version{Path: path, Version: v}) {
				return repo.Stat(v)
			}
		}
		for i := len(versions) - 1; i >= 0; i-- {
			v := versions[i]
			if semver.Prerelease(v) != "" && ok(module.Version{Path: path, Version: v}) {
				return repo.Stat(v)
			}
		}
	}

	if query == "latest" {
		// Special case for "latest": if no tags match, use latest commit in repo,
		// provided it is not excluded.
		if info, err := repo.Latest(); err == nil && allowed(module.Version{Path: path, Version: info.Version}) {
			return info, nil
		}
	}

	return nil, fmt.Errorf("no matching versions for query %q", query)
}

// isSemverPrefix reports whether v is a semantic version prefix: v1 or  v1.2 (not v1.2.3).
// The caller is assumed to have checked that semver.IsValid(v) is true.
func isSemverPrefix(v string) bool {
	dots := 0
	for i := 0; i < len(v); i++ {
		switch v[i] {
		case '-', '+':
			return false
		case '.':
			dots++
			if dots >= 2 {
				return false
			}
		}
	}
	return true
}

// matchSemverPrefix reports whether the shortened semantic version p
// matches the full-width (non-shortened) semantic version v.
func matchSemverPrefix(p, v string) bool {
	return len(v) > len(p) && v[len(p)] == '.' && v[:len(p)] == p
}
