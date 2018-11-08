// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
	"fmt"
	pathpkg "path"
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
// If path is the path of the main module and the query is "latest",
// Query returns Target.Version as the version.
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
	var prefix string
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
		prefix = query + "."

	case semver.IsValid(query):
		vers := module.CanonicalVersion(query)
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

	if path == Target.Path {
		if query != "latest" {
			return nil, fmt.Errorf("can't query specific version (%q) for the main module (%s)", query, path)
		}
		if !allowed(Target) {
			return nil, fmt.Errorf("internal error: main module version is not allowed")
		}
		return &modfetch.RevInfo{Version: Target.Version}, nil
	}

	// Load versions and execute query.
	repo, err := modfetch.Lookup(path)
	if err != nil {
		return nil, err
	}
	versions, err := repo.Versions(prefix)
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

// QueryPackage looks up a revision of a module containing path.
//
// If multiple modules with revisions matching the query provide the requested
// package, QueryPackage picks the one with the longest module path.
//
// If the path is in the main module and the query is "latest",
// QueryPackage returns Target as the version.
func QueryPackage(path, query string, allowed func(module.Version) bool) (module.Version, *modfetch.RevInfo, error) {
	if HasModRoot() {
		if _, ok := dirInModule(path, Target.Path, modRoot, true); ok {
			if query != "latest" {
				return module.Version{}, nil, fmt.Errorf("can't query specific version (%q) for package %s in the main module (%s)", query, path, Target.Path)
			}
			if !allowed(Target) {
				return module.Version{}, nil, fmt.Errorf("internal error: package %s is in the main module (%s), but version is not allowed", path, Target.Path)
			}
			return Target, &modfetch.RevInfo{Version: Target.Version}, nil
		}
	}

	finalErr := errMissing
	for p := path; p != "." && p != "/"; p = pathpkg.Dir(p) {
		info, err := Query(p, query, allowed)
		if err != nil {
			if _, ok := err.(*codehost.VCSError); ok {
				// A VCSError means we know where to find the code,
				// we just can't. Abort search.
				return module.Version{}, nil, err
			}
			if finalErr == errMissing {
				finalErr = err
			}
			continue
		}
		m := module.Version{Path: p, Version: info.Version}
		root, isLocal, err := fetch(m)
		if err != nil {
			return module.Version{}, nil, err
		}
		_, ok := dirInModule(path, m.Path, root, isLocal)
		if ok {
			return m, info, nil
		}
	}

	return module.Version{}, nil, finalErr
}
