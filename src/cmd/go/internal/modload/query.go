// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"fmt"
	pathpkg "path"
	"strings"
	"sync"

	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/module"
	"cmd/go/internal/search"
	"cmd/go/internal/semver"
	"cmd/go/internal/str"
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

	if str.HasPathPrefix(path, "std") || str.HasPathPrefix(path, "cmd") {
		return nil, fmt.Errorf("explicit requirement on standard-library module %s not allowed", path)
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

	return nil, &NoMatchingVersionError{query: query}
}

// isSemverPrefix reports whether v is a semantic version prefix: v1 or  v1.2 (not wv1.2.3).
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

type QueryResult struct {
	Mod      module.Version
	Rev      *modfetch.RevInfo
	Packages []string
}

// QueryPackage looks up the module(s) containing path at a revision matching
// query. The results are sorted by module path length in descending order.
//
// If the package is in the main module, QueryPackage considers only the main
// module and only the version "latest", without checking for other possible
// modules.
func QueryPackage(path, query string, allowed func(module.Version) bool) ([]QueryResult, error) {
	if search.IsMetaPackage(path) || strings.Contains(path, "...") {
		return nil, fmt.Errorf("pattern %s is not an importable package", path)
	}
	return QueryPattern(path, query, allowed)
}

// QueryPattern looks up the module(s) containing at least one package matching
// the given pattern at the given version. The results are sorted by module path
// length in descending order.
//
// QueryPattern queries modules with package paths up to the first "..."
// in the pattern. For the pattern "example.com/a/b.../c", QueryPattern would
// consider prefixes of "example.com/a". If multiple modules have versions
// that match the query and packages that match the pattern, QueryPattern
// picks the one with the longest module path.
//
// If any matching package is in the main module, QueryPattern considers only
// the main module and only the version "latest", without checking for other
// possible modules.
func QueryPattern(pattern string, query string, allowed func(module.Version) bool) ([]QueryResult, error) {
	base := pattern
	var match func(m module.Version, root string, isLocal bool) (pkgs []string)

	if i := strings.Index(pattern, "..."); i >= 0 {
		base = pathpkg.Dir(pattern[:i+3])
		match = func(m module.Version, root string, isLocal bool) []string {
			return matchPackages(pattern, anyTags, false, []module.Version{m})
		}
	} else {
		match = func(m module.Version, root string, isLocal bool) []string {
			prefix := m.Path
			if m == Target {
				prefix = targetPrefix
			}
			if _, ok := dirInModule(pattern, prefix, root, isLocal); ok {
				return []string{pattern}
			} else {
				return nil
			}
		}
	}

	if HasModRoot() {
		pkgs := match(Target, modRoot, true)
		if len(pkgs) > 0 {
			if query != "latest" {
				return nil, fmt.Errorf("can't query specific version for package %s in the main module (%s)", pattern, Target.Path)
			}
			if !allowed(Target) {
				return nil, fmt.Errorf("internal error: package %s is in the main module (%s), but version is not allowed", pattern, Target.Path)
			}
			return []QueryResult{{
				Mod:      Target,
				Rev:      &modfetch.RevInfo{Version: Target.Version},
				Packages: pkgs,
			}}, nil
		}
	}

	// If the path we're attempting is not in the module cache and we don't have a
	// fetch result cached either, we'll end up making a (potentially slow)
	// request to the proxy or (often even slower) the origin server.
	// To minimize latency, execute all of those requests in parallel.
	type result struct {
		QueryResult
		err error
	}
	results := make([]result, strings.Count(base, "/")+1) // by descending path length
	i, p := 0, base
	var wg sync.WaitGroup
	wg.Add(len(results))
	for {
		go func(p string, r *result) (err error) {
			defer func() {
				r.err = err
				wg.Done()
			}()

			r.Mod.Path = p
			if HasModRoot() && p == Target.Path {
				r.Mod.Version = Target.Version
				r.Rev = &modfetch.RevInfo{Version: Target.Version}
				// We already know (from above) that Target does not contain any
				// packages matching pattern, so leave r.Packages empty.
			} else {
				r.Rev, err = Query(p, query, allowed)
				if err != nil {
					return err
				}
				r.Mod.Version = r.Rev.Version
				root, isLocal, err := fetch(r.Mod)
				if err != nil {
					return err
				}
				r.Packages = match(r.Mod, root, isLocal)
			}
			if len(r.Packages) == 0 {
				return &packageNotInModuleError{
					mod:     r.Mod,
					query:   query,
					pattern: pattern,
				}
			}
			return nil
		}(p, &results[i])

		j := strings.LastIndexByte(p, '/')
		if i++; i == len(results) {
			if j >= 0 {
				panic("undercounted slashes")
			}
			break
		}
		if j < 0 {
			panic("overcounted slashes")
		}
		p = p[:j]
	}
	wg.Wait()

	// Classify the results. In case of failure, identify the error that the user
	// is most likely to find helpful.
	var (
		successes  []QueryResult
		mostUseful result
	)
	for _, r := range results {
		if r.err == nil {
			successes = append(successes, r.QueryResult)
			continue
		}

		switch mostUseful.err.(type) {
		case nil:
			mostUseful = r
			continue
		case *packageNotInModuleError:
			// Any other error is more useful than one that reports that the main
			// module does not contain the requested packages.
			if mostUseful.Mod.Path == Target.Path {
				mostUseful = r
				continue
			}
		}

		switch r.err.(type) {
		case *codehost.VCSError:
			// A VCSError means that we've located a repository, but couldn't look
			// inside it for packages. That's a very strong signal, and should
			// override any others.
			return nil, r.err
		case *packageNotInModuleError:
			if r.Mod.Path == Target.Path {
				// Don't override a potentially-useful error for some other module with
				// a trivial error for the main module.
				continue
			}
			// A module with an appropriate prefix exists at the requested version,
			// but it does not contain the requested package(s).
			if _, worsePath := mostUseful.err.(*packageNotInModuleError); !worsePath {
				mostUseful = r
			}
		case *NoMatchingVersionError:
			// A module with an appropriate prefix exists, but not at the requested
			// version.
			_, worseError := mostUseful.err.(*packageNotInModuleError)
			_, worsePath := mostUseful.err.(*NoMatchingVersionError)
			if !(worseError || worsePath) {
				mostUseful = r
			}
		}
	}

	// TODO(#26232): If len(successes) == 0 and some of the errors are 4xx HTTP
	// codes, have the auth package recheck the failed paths.
	// If we obtain new credentials for any of them, re-run the above loop.

	if len(successes) == 0 {
		// All of the possible module paths either did not exist at the requested
		// version, or did not contain the requested package(s).
		return nil, mostUseful.err
	}

	// At least one module at the requested version contained the requested
	// package(s). Any remaining errors only describe the non-existence of
	// alternatives, so ignore them.
	return successes, nil
}

// A NoMatchingVersionError indicates that Query found a module at the requested
// path, but not at any versions satisfying the query string and allow-function.
type NoMatchingVersionError struct {
	query string
}

func (e *NoMatchingVersionError) Error() string {
	return fmt.Sprintf("no matching versions for query %q", e.query)
}

// A packageNotInModuleError indicates that QueryPattern found a candidate
// module at the requested version, but that module did not contain any packages
// matching the requested pattern.
type packageNotInModuleError struct {
	mod     module.Version
	query   string
	pattern string
}

func (e *packageNotInModuleError) Error() string {
	found := ""
	if e.query != e.mod.Version {
		found = fmt.Sprintf(" (%s)", e.mod.Version)
	}

	if strings.Contains(e.pattern, "...") {
		return fmt.Sprintf("module %s@%s%s found, but does not contain packages matching %s", e.mod.Path, e.query, found, e.pattern)
	}
	return fmt.Sprintf("module %s@%s%s found, but does not contain package %s", e.mod.Path, e.query, found, e.pattern)
}
