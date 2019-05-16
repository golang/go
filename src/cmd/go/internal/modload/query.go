// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"errors"
	"fmt"
	"os"
	pathpkg "path"
	"strings"
	"sync"

	"cmd/go/internal/modfetch"
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
//	- a repository commit identifier or tag, denoting that commit.
//
// If the allowed function is non-nil, Query excludes any versions for which allowed returns false.
//
// If path is the path of the main module and the query is "latest",
// Query returns Target.Version as the version.
func Query(path, query string, allowed func(module.Version) bool) (*modfetch.RevInfo, error) {
	var info *modfetch.RevInfo
	err := modfetch.TryProxies(func(proxy string) (err error) {
		info, err = queryProxy(proxy, path, query, allowed)
		return err
	})
	return info, err
}

func queryProxy(proxy, path, query string, allowed func(module.Version) bool) (*modfetch.RevInfo, error) {
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

	default:
		// Direct lookup of semantic version or commit identifier.
		//
		// If the identifier is not a canonical semver tag — including if it's a
		// semver tag with a +metadata suffix — then modfetch.Stat will populate
		// info.Version with a suitable pseudo-version.
		info, err := modfetch.Stat(proxy, path, query)
		if err != nil {
			queryErr := err
			// The full query doesn't correspond to a tag. If it is a semantic version
			// with a +metadata suffix, see if there is a tag without that suffix:
			// semantic versioning defines them to be equivalent.
			if vers := module.CanonicalVersion(query); vers != "" && vers != query {
				info, err = modfetch.Stat(proxy, path, vers)
			}
			if err != nil {
				return nil, queryErr
			}
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
	repo, err := modfetch.Lookup(proxy, path)
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

// isSemverPrefix reports whether v is a semantic version prefix: v1 or v1.2 (not v1.2.3).
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
	return len(v) > len(p) && v[len(p)] == '.' && v[:len(p)] == p && semver.Prerelease(v) == ""
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
func QueryPattern(pattern, query string, allowed func(module.Version) bool) ([]QueryResult, error) {
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

	var (
		results          []QueryResult
		candidateModules = modulePrefixesExcludingTarget(base)
	)
	if len(candidateModules) == 0 {
		return nil, fmt.Errorf("package %s is not in the main module (%s)", pattern, Target.Path)
	}

	err := modfetch.TryProxies(func(proxy string) error {
		queryModule := func(path string) (r QueryResult, err error) {
			r.Mod.Path = path
			r.Rev, err = queryProxy(proxy, path, query, allowed)
			if err != nil {
				return r, err
			}
			r.Mod.Version = r.Rev.Version
			root, isLocal, err := fetch(r.Mod)
			if err != nil {
				return r, err
			}
			r.Packages = match(r.Mod, root, isLocal)
			if len(r.Packages) == 0 {
				return r, &packageNotInModuleError{
					mod:     r.Mod,
					query:   query,
					pattern: pattern,
				}
			}
			return r, nil
		}

		var err error
		results, err = queryPrefixModules(candidateModules, queryModule)
		return err
	})

	return results, err
}

// modulePrefixesExcludingTarget returns all prefixes of path that may plausibly
// exist as a module, excluding targetPrefix but otherwise including path
// itself, sorted by descending length.
func modulePrefixesExcludingTarget(path string) []string {
	prefixes := make([]string, 0, strings.Count(path, "/")+1)

	for {
		if path != targetPrefix {
			if _, _, ok := module.SplitPathVersion(path); ok {
				prefixes = append(prefixes, path)
			}
		}

		j := strings.LastIndexByte(path, '/')
		if j < 0 {
			break
		}
		path = path[:j]
	}

	return prefixes
}

type prefixResult struct {
	QueryResult
	err error
}

func queryPrefixModules(candidateModules []string, queryModule func(path string) (QueryResult, error)) (found []QueryResult, err error) {
	// If the path we're attempting is not in the module cache and we don't have a
	// fetch result cached either, we'll end up making a (potentially slow)
	// request to the proxy or (often even slower) the origin server.
	// To minimize latency, execute all of those requests in parallel.
	type result struct {
		QueryResult
		err error
	}
	results := make([]result, len(candidateModules))
	var wg sync.WaitGroup
	wg.Add(len(candidateModules))
	for i, p := range candidateModules {
		go func(p string, r *result) {
			r.QueryResult, r.err = queryModule(p)
			wg.Done()
		}(p, &results[i])
	}
	wg.Wait()

	// Classify the results. In case of failure, identify the error that the user
	// is most likely to find helpful.
	var (
		noVersion   *NoMatchingVersionError
		noPackage   *packageNotInModuleError
		notExistErr error
	)
	for _, r := range results {
		switch rErr := r.err.(type) {
		case nil:
			found = append(found, r.QueryResult)
		case *NoMatchingVersionError:
			if noVersion == nil {
				noVersion = rErr
			}
		case *packageNotInModuleError:
			if noPackage == nil {
				noPackage = rErr
			}
		default:
			if errors.Is(rErr, os.ErrNotExist) {
				if notExistErr == nil {
					notExistErr = rErr
				}
			} else {
				err = r.err
			}
		}
	}

	// TODO(#26232): If len(found) == 0 and some of the errors are 4xx HTTP
	// codes, have the auth package recheck the failed paths.
	// If we obtain new credentials for any of them, re-run the above loop.

	if len(found) == 0 && err == nil {
		switch {
		case noPackage != nil:
			err = noPackage
		case noVersion != nil:
			err = noVersion
		case notExistErr != nil:
			err = notExistErr
		default:
			panic("queryPrefixModules: no modules found, but no error detected")
		}
	}

	return found, err
}

// A NoMatchingVersionError indicates that Query found a module at the requested
// path, but not at any versions satisfying the query string and allow-function.
//
// NOTE: NoMatchingVersionError MUST NOT implement Is(os.ErrNotExist).
//
// If the module came from a proxy, that proxy had to return a successful status
// code for the versions it knows about, and thus did not have the opportunity
// to return a non-400 status code to suppress fallback.
type NoMatchingVersionError struct {
	query string
}

func (e *NoMatchingVersionError) Error() string {
	return fmt.Sprintf("no matching versions for query %q", e.query)
}

// A packageNotInModuleError indicates that QueryPattern found a candidate
// module at the requested version, but that module did not contain any packages
// matching the requested pattern.
//
// NOTE: packageNotInModuleError MUST NOT implement Is(os.ErrNotExist).
//
// If the module came from a proxy, that proxy had to return a successful status
// code for the versions it knows about, and thus did not have the opportunity
// to return a non-400 status code to suppress fallback.
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

// ModuleHasRootPackage returns whether module m contains a package m.Path.
func ModuleHasRootPackage(m module.Version) (bool, error) {
	root, isLocal, err := fetch(m)
	if err != nil {
		return false, err
	}
	_, ok := dirInModule(m.Path, m.Path, root, isLocal)
	return ok, nil
}
