// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	pathpkg "path"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/imports"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"
	"cmd/internal/pkgpattern"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// Query looks up a revision of a given module given a version query string.
// The module must be a complete module path.
// The version must take one of the following forms:
//
//   - the literal string "latest", denoting the latest available, allowed
//     tagged version, with non-prereleases preferred over prereleases.
//     If there are no tagged versions in the repo, latest returns the most
//     recent commit.
//
//   - the literal string "upgrade", equivalent to "latest" except that if
//     current is a newer version, current will be returned (see below).
//
//   - the literal string "patch", denoting the latest available tagged version
//     with the same major and minor number as current (see below).
//
//   - v1, denoting the latest available tagged version v1.x.x.
//
//   - v1.2, denoting the latest available tagged version v1.2.x.
//
//   - v1.2.3, a semantic version string denoting that tagged version.
//
//   - <v1.2.3, <=v1.2.3, >v1.2.3, >=v1.2.3,
//     denoting the version closest to the target and satisfying the given operator,
//     with non-prereleases preferred over prereleases.
//
//   - a repository commit identifier or tag, denoting that commit.
//
// current denotes the currently-selected version of the module; it may be
// "none" if no version is currently selected, or "" if the currently-selected
// version is unknown or should not be considered. If query is
// "upgrade" or "patch", current will be returned if it is a newer
// semantic version or a chronologically later pseudo-version than the
// version that would otherwise be chosen. This prevents accidental downgrades
// from newer pre-release or development versions.
//
// The allowed function (which may be nil) is used to filter out unsuitable
// versions (see AllowedFunc documentation for details). If the query refers to
// a specific revision (for example, "master"; see IsRevisionQuery), and the
// revision is disallowed by allowed, Query returns the error. If the query
// does not refer to a specific revision (for example, "latest"), Query
// acts as if versions disallowed by allowed do not exist.
//
// If path is the path of the main module and the query is "latest",
// Query returns Target.Version as the version.
//
// Query often returns a non-nil *RevInfo with a non-nil error,
// to provide an info.Origin that can allow the error to be cached.
func Query(ctx context.Context, path, query, current string, allowed AllowedFunc) (*modfetch.RevInfo, error) {
	ctx, span := trace.StartSpan(ctx, "modload.Query "+path)
	defer span.Done()

	return queryReuse(ctx, path, query, current, allowed, nil)
}

// queryReuse is like Query but also takes a map of module info that can be reused
// if the validation criteria in Origin are met.
func queryReuse(ctx context.Context, path, query, current string, allowed AllowedFunc, reuse map[module.Version]*modinfo.ModulePublic) (*modfetch.RevInfo, error) {
	var info *modfetch.RevInfo
	err := modfetch.TryProxies(func(proxy string) (err error) {
		info, err = queryProxy(ctx, proxy, path, query, current, allowed, reuse)
		return err
	})
	return info, err
}

// checkReuse checks whether a revision of a given module
// for a given module may be reused, according to the information in origin.
func checkReuse(ctx context.Context, m module.Version, old *codehost.Origin) error {
	return modfetch.TryProxies(func(proxy string) error {
		repo, err := lookupRepo(ctx, proxy, m.Path)
		if err != nil {
			return err
		}
		return checkReuseRepo(ctx, repo, m.Path, m.Version, old)
	})
}

func checkReuseRepo(ctx context.Context, repo versionRepo, path, query string, origin *codehost.Origin) error {
	if origin == nil {
		return errors.New("nil Origin")
	}

	// Ensure that the Origin actually includes enough fields to resolve the query.
	// If we got the previous Origin data from a proxy, it may be missing something
	// that we would have needed to resolve the query directly from the repo.
	switch {
	case origin.RepoSum != "":
		// A RepoSum is always acceptable, since it incorporates everything
		// (and is often associated with an error result).

	case query == module.CanonicalVersion(query):
		// This query refers to a specific version, and Go module versions
		// are supposed to be cacheable and immutable (confirmed with checksums).
		// If the version exists at all, we shouldn't need any extra information
		// to identify which commit it resolves to.
		//
		// It may be associated with a Ref for a semantic-version tag, but if so
		// we don't expect that tag to change in the future. We also don't need a
		// TagSum: if a tag is removed from some ancestor commit, the version may
		// change from valid to invalid, but we're ok with keeping stale versions
		// as long as they were valid at some point in the past.
		//
		// If the version did not successfully resolve, the origin may indicate
		// a TagSum and/or RepoSum instead of a Hash, in which case we still need
		// to check those to ensure that the error is still applicable.
		if origin.Hash == "" && origin.Ref == "" && origin.TagSum == "" {
			return errors.New("no Origin information to check")
		}

	case IsRevisionQuery(path, query):
		// This query may refer to a branch, non-version tag, or commit ID.
		//
		// If it is a commit ID, we expect to see a Hash in the Origin data. On
		// the other hand, if it is not a commit ID, we expect to see either a Ref
		// (for a positive result) or a RepoSum (for a negative result), since
		// we don't expect refs in general to remain stable over time.
		if origin.Hash == "" && origin.Ref == "" {
			return fmt.Errorf("query %q requires a Hash or Ref", query)
		}
		// Once we resolve the query to a particular commit, we will need to
		// also identify the most appropriate version to assign to that commit.
		// (It may correspond to more than one valid version.)
		//
		// The most appropriate version depends on the tags associated with
		// both the commit itself (if the commit is a tagged version)
		// and its ancestors (if we need to produce a pseudo-version for it).
		if origin.TagSum == "" {
			return fmt.Errorf("query %q requires a TagSum", query)
		}

	default:
		// The query may be "latest" or a version inequality or prefix.
		// Its result depends on the absence of higher tags matching the query,
		// not just the state of an individual ref or tag.
		if origin.TagSum == "" {
			return fmt.Errorf("query %q requires a TagSum", query)
		}
	}

	return repo.CheckReuse(ctx, origin)
}

// AllowedFunc is used by Query and other functions to filter out unsuitable
// versions, for example, those listed in exclude directives in the main
// module's go.mod file.
//
// An AllowedFunc returns an error equivalent to ErrDisallowed for an unsuitable
// version. Any other error indicates the function was unable to determine
// whether the version should be allowed, for example, the function was unable
// to fetch or parse a go.mod file containing retractions. Typically, errors
// other than ErrDisallowed may be ignored.
type AllowedFunc func(context.Context, module.Version) error

var errQueryDisabled error = queryDisabledError{}

type queryDisabledError struct{}

func (queryDisabledError) Error() string {
	if cfg.BuildModReason == "" {
		return fmt.Sprintf("cannot query module due to -mod=%s", cfg.BuildMod)
	}
	return fmt.Sprintf("cannot query module due to -mod=%s\n\t(%s)", cfg.BuildMod, cfg.BuildModReason)
}

func queryProxy(ctx context.Context, proxy, path, query, current string, allowed AllowedFunc, reuse map[module.Version]*modinfo.ModulePublic) (*modfetch.RevInfo, error) {
	ctx, span := trace.StartSpan(ctx, "modload.queryProxy "+path+" "+query)
	defer span.Done()

	if current != "" && current != "none" && !gover.ModIsValid(path, current) {
		return nil, fmt.Errorf("invalid previous version %v@%v", path, current)
	}
	if cfg.BuildMod == "vendor" {
		return nil, errQueryDisabled
	}
	if allowed == nil {
		allowed = func(context.Context, module.Version) error { return nil }
	}

	if MainModules.Contains(path) && (query == "upgrade" || query == "patch") {
		m := module.Version{Path: path}
		if err := allowed(ctx, m); err != nil {
			return nil, fmt.Errorf("internal error: main module version is not allowed: %w", err)
		}
		return &modfetch.RevInfo{Version: m.Version}, nil
	}

	if path == "std" || path == "cmd" {
		return nil, fmt.Errorf("can't query specific version (%q) of standard-library module %q", query, path)
	}

	repo, err := lookupRepo(ctx, proxy, path)
	if err != nil {
		return nil, err
	}

	if old := reuse[module.Version{Path: path, Version: query}]; old != nil {
		if err := checkReuseRepo(ctx, repo, path, query, old.Origin); err == nil {
			info := &modfetch.RevInfo{
				Version: old.Version,
				Origin:  old.Origin,
			}
			if !old.Time.IsZero() {
				info.Time = old.Time
			}
			return info, nil
		}
	}

	// Parse query to detect parse errors (and possibly handle query)
	// before any network I/O.
	qm, err := newQueryMatcher(path, query, current, allowed)
	if (err == nil && qm.canStat) || err == errRevQuery {
		// Direct lookup of a commit identifier or complete (non-prefix) semantic
		// version.

		// If the identifier is not a canonical semver tag — including if it's a
		// semver tag with a +metadata suffix — then modfetch.Stat will populate
		// info.Version with a suitable pseudo-version.
		info, err := repo.Stat(ctx, query)
		if err != nil {
			queryErr := err
			// The full query doesn't correspond to a tag. If it is a semantic version
			// with a +metadata suffix, see if there is a tag without that suffix:
			// semantic versioning defines them to be equivalent.
			canonicalQuery := module.CanonicalVersion(query)
			if canonicalQuery != "" && query != canonicalQuery {
				info, err = repo.Stat(ctx, canonicalQuery)
				if err != nil && !errors.Is(err, fs.ErrNotExist) {
					return info, err
				}
			}
			if err != nil {
				return info, queryErr
			}
		}
		if err := allowed(ctx, module.Version{Path: path, Version: info.Version}); errors.Is(err, ErrDisallowed) {
			return nil, err
		}
		return info, nil
	} else if err != nil {
		return nil, err
	}

	// Load versions and execute query.
	versions, err := repo.Versions(ctx, qm.prefix)
	if err != nil {
		return nil, err
	}
	origin := versions.Origin

	revWithOrigin := func(rev *modfetch.RevInfo) *modfetch.RevInfo {
		if rev == nil {
			if origin == nil {
				return nil
			}
			return &modfetch.RevInfo{Origin: origin}
		}

		clone := *rev
		clone.Origin = origin
		return &clone
	}

	releases, prereleases, err := qm.filterVersions(ctx, versions.List)
	if err != nil {
		return revWithOrigin(nil), err
	}

	lookup := func(v string) (*modfetch.RevInfo, error) {
		rev, err := repo.Stat(ctx, v)
		if rev != nil {
			// Note that Stat can return a non-nil rev and a non-nil err,
			// in order to provide origin information to make the error cacheable.
			origin = mergeOrigin(origin, rev.Origin)
		}
		if err != nil {
			return revWithOrigin(nil), err
		}

		if (query == "upgrade" || query == "patch") && module.IsPseudoVersion(current) && !rev.Time.IsZero() {
			// Don't allow "upgrade" or "patch" to move from a pseudo-version
			// to a chronologically older version or pseudo-version.
			//
			// If the current version is a pseudo-version from an untagged branch, it
			// may be semantically lower than the "latest" release or the latest
			// pseudo-version on the main branch. A user on such a version is unlikely
			// to intend to “upgrade” to a version that already existed at that point
			// in time.
			//
			// We do this only if the current version is a pseudo-version: if the
			// version is tagged, the author of the dependency module has given us
			// explicit information about their intended precedence of this version
			// relative to other versions, and we shouldn't contradict that
			// information. (For example, v1.0.1 might be a backport of a fix already
			// incorporated into v1.1.0, in which case v1.0.1 would be chronologically
			// newer but v1.1.0 is still an “upgrade”; or v1.0.2 might be a revert of
			// an unsuccessful fix in v1.0.1, in which case the v1.0.2 commit may be
			// older than the v1.0.1 commit despite the tag itself being newer.)
			currentTime, err := module.PseudoVersionTime(current)
			if err == nil && rev.Time.Before(currentTime) {
				if err := allowed(ctx, module.Version{Path: path, Version: current}); errors.Is(err, ErrDisallowed) {
					return revWithOrigin(nil), err
				}
				rev, err = repo.Stat(ctx, current)
				if rev != nil {
					origin = mergeOrigin(origin, rev.Origin)
				}
				if err != nil {
					return revWithOrigin(nil), err
				}
				return revWithOrigin(rev), nil
			}
		}

		return revWithOrigin(rev), nil
	}

	if qm.preferLower {
		if len(releases) > 0 {
			return lookup(releases[0])
		}
		if len(prereleases) > 0 {
			return lookup(prereleases[0])
		}
	} else {
		if len(releases) > 0 {
			return lookup(releases[len(releases)-1])
		}
		if len(prereleases) > 0 {
			return lookup(prereleases[len(prereleases)-1])
		}
	}

	if qm.mayUseLatest {
		latest, err := repo.Latest(ctx)
		if latest != nil {
			origin = mergeOrigin(origin, latest.Origin)
		}
		if err == nil {
			if qm.allowsVersion(ctx, latest.Version) {
				return lookup(latest.Version)
			}
		} else if !errors.Is(err, fs.ErrNotExist) {
			return revWithOrigin(nil), err
		}
	}

	if (query == "upgrade" || query == "patch") && current != "" && current != "none" {
		// "upgrade" and "patch" may stay on the current version if allowed.
		if err := allowed(ctx, module.Version{Path: path, Version: current}); errors.Is(err, ErrDisallowed) {
			return revWithOrigin(nil), err
		}
		return lookup(current)
	}

	return revWithOrigin(nil), &NoMatchingVersionError{query: query, current: current}
}

// IsRevisionQuery returns true if vers is a version query that may refer to
// a particular version or revision in a repository like "v1.0.0", "master",
// or "0123abcd". IsRevisionQuery returns false if vers is a query that
// chooses from among available versions like "latest" or ">v1.0.0".
func IsRevisionQuery(path, vers string) bool {
	if vers == "latest" ||
		vers == "upgrade" ||
		vers == "patch" ||
		strings.HasPrefix(vers, "<") ||
		strings.HasPrefix(vers, ">") ||
		(gover.ModIsValid(path, vers) && gover.ModIsPrefix(path, vers)) {
		return false
	}
	return true
}

type queryMatcher struct {
	path               string
	prefix             string
	filter             func(version string) bool
	allowed            AllowedFunc
	canStat            bool // if true, the query can be resolved by repo.Stat
	preferLower        bool // if true, choose the lowest matching version
	mayUseLatest       bool
	preferIncompatible bool
}

var errRevQuery = errors.New("query refers to a non-semver revision")

// newQueryMatcher returns a new queryMatcher that matches the versions
// specified by the given query on the module with the given path.
//
// If the query can only be resolved by statting a non-SemVer revision,
// newQueryMatcher returns errRevQuery.
func newQueryMatcher(path string, query, current string, allowed AllowedFunc) (*queryMatcher, error) {
	badVersion := func(v string) (*queryMatcher, error) {
		return nil, fmt.Errorf("invalid semantic version %q in range %q", v, query)
	}

	matchesMajor := func(v string) bool {
		_, pathMajor, ok := module.SplitPathVersion(path)
		if !ok {
			return false
		}
		return module.CheckPathMajor(v, pathMajor) == nil
	}

	qm := &queryMatcher{
		path:               path,
		allowed:            allowed,
		preferIncompatible: strings.HasSuffix(current, "+incompatible"),
	}

	switch {
	case query == "latest":
		qm.mayUseLatest = true

	case query == "upgrade":
		if current == "" || current == "none" {
			qm.mayUseLatest = true
		} else {
			qm.mayUseLatest = module.IsPseudoVersion(current)
			qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, current) >= 0 }
		}

	case query == "patch":
		if current == "" || current == "none" {
			return nil, &NoPatchBaseError{path}
		}
		if current == "" {
			qm.mayUseLatest = true
		} else {
			qm.mayUseLatest = module.IsPseudoVersion(current)
			qm.prefix = gover.ModMajorMinor(qm.path, current) + "."
			qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, current) >= 0 }
		}

	case strings.HasPrefix(query, "<="):
		v := query[len("<="):]
		if !gover.ModIsValid(path, v) {
			return badVersion(v)
		}
		if gover.ModIsPrefix(path, v) {
			// Refuse to say whether <=v1.2 allows v1.2.3 (remember, @v1.2 might mean v1.2.3).
			return nil, fmt.Errorf("ambiguous semantic version %q in range %q", v, query)
		}
		qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, v) <= 0 }
		if !matchesMajor(v) {
			qm.preferIncompatible = true
		}

	case strings.HasPrefix(query, "<"):
		v := query[len("<"):]
		if !gover.ModIsValid(path, v) {
			return badVersion(v)
		}
		qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, v) < 0 }
		if !matchesMajor(v) {
			qm.preferIncompatible = true
		}

	case strings.HasPrefix(query, ">="):
		v := query[len(">="):]
		if !gover.ModIsValid(path, v) {
			return badVersion(v)
		}
		qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, v) >= 0 }
		qm.preferLower = true
		if !matchesMajor(v) {
			qm.preferIncompatible = true
		}

	case strings.HasPrefix(query, ">"):
		v := query[len(">"):]
		if !gover.ModIsValid(path, v) {
			return badVersion(v)
		}
		if gover.ModIsPrefix(path, v) {
			// Refuse to say whether >v1.2 allows v1.2.3 (remember, @v1.2 might mean v1.2.3).
			return nil, fmt.Errorf("ambiguous semantic version %q in range %q", v, query)
		}
		qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, v) > 0 }
		qm.preferLower = true
		if !matchesMajor(v) {
			qm.preferIncompatible = true
		}

	case gover.ModIsValid(path, query):
		if gover.ModIsPrefix(path, query) {
			qm.prefix = query + "."
			// Do not allow the query "v1.2" to match versions lower than "v1.2.0",
			// such as prereleases for that version. (https://golang.org/issue/31972)
			qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, query) >= 0 }
		} else {
			qm.canStat = true
			qm.filter = func(mv string) bool { return gover.ModCompare(qm.path, mv, query) == 0 }
			qm.prefix = semver.Canonical(query)
		}
		if !matchesMajor(query) {
			qm.preferIncompatible = true
		}

	default:
		return nil, errRevQuery
	}

	return qm, nil
}

// allowsVersion reports whether version v is allowed by the prefix, filter, and
// AllowedFunc of qm.
func (qm *queryMatcher) allowsVersion(ctx context.Context, v string) bool {
	if qm.prefix != "" && !strings.HasPrefix(v, qm.prefix) {
		if gover.IsToolchain(qm.path) && strings.TrimSuffix(qm.prefix, ".") == v {
			// Allow 1.21 to match "1.21." prefix.
		} else {
			return false
		}
	}
	if qm.filter != nil && !qm.filter(v) {
		return false
	}
	if qm.allowed != nil {
		if err := qm.allowed(ctx, module.Version{Path: qm.path, Version: v}); errors.Is(err, ErrDisallowed) {
			return false
		}
	}
	return true
}

// filterVersions classifies versions into releases and pre-releases, filtering
// out:
//  1. versions that do not satisfy the 'allowed' predicate, and
//  2. "+incompatible" versions, if a compatible one satisfies the predicate
//     and the incompatible version is not preferred.
//
// If the allowed predicate returns an error not equivalent to ErrDisallowed,
// filterVersions returns that error.
func (qm *queryMatcher) filterVersions(ctx context.Context, versions []string) (releases, prereleases []string, err error) {
	needIncompatible := qm.preferIncompatible

	var lastCompatible string
	for _, v := range versions {
		if !qm.allowsVersion(ctx, v) {
			continue
		}

		if !needIncompatible {
			// We're not yet sure whether we need to include +incompatible versions.
			// Keep track of the last compatible version we've seen, and use the
			// presence (or absence) of a go.mod file in that version to decide: a
			// go.mod file implies that the module author is supporting modules at a
			// compatible version (and we should ignore +incompatible versions unless
			// requested explicitly), while a lack of go.mod file implies the
			// potential for legacy (pre-modules) versioning without semantic import
			// paths (and thus *with* +incompatible versions).
			//
			// This isn't strictly accurate if the latest compatible version has been
			// replaced by a local file path, because we do not allow file-path
			// replacements without a go.mod file: the user would have needed to add
			// one. However, replacing the last compatible version while
			// simultaneously expecting to upgrade implicitly to a +incompatible
			// version seems like an extreme enough corner case to ignore for now.

			if !strings.HasSuffix(v, "+incompatible") {
				lastCompatible = v
			} else if lastCompatible != "" {
				// If the latest compatible version is allowed and has a go.mod file,
				// ignore any version with a higher (+incompatible) major version. (See
				// https://golang.org/issue/34165.) Note that we even prefer a
				// compatible pre-release over an incompatible release.
				ok, err := versionHasGoMod(ctx, module.Version{Path: qm.path, Version: lastCompatible})
				if err != nil {
					return nil, nil, err
				}
				if ok {
					// The last compatible version has a go.mod file, so that's the
					// highest version we're willing to consider. Don't bother even
					// looking at higher versions, because they're all +incompatible from
					// here onward.
					break
				}

				// No acceptable compatible release has a go.mod file, so the versioning
				// for the module might not be module-aware, and we should respect
				// legacy major-version tags.
				needIncompatible = true
			}
		}

		if gover.ModIsPrerelease(qm.path, v) {
			prereleases = append(prereleases, v)
		} else {
			releases = append(releases, v)
		}
	}

	return releases, prereleases, nil
}

type QueryResult struct {
	Mod      module.Version
	Rev      *modfetch.RevInfo
	Packages []string
}

// QueryPackages is like QueryPattern, but requires that the pattern match at
// least one package and omits the non-package result (if any).
func QueryPackages(ctx context.Context, pattern, query string, current func(string) string, allowed AllowedFunc) ([]QueryResult, error) {
	pkgMods, modOnly, err := QueryPattern(ctx, pattern, query, current, allowed)

	if len(pkgMods) == 0 && err == nil {
		replacement := Replacement(modOnly.Mod)
		return nil, &PackageNotInModuleError{
			Mod:         modOnly.Mod,
			Replacement: replacement,
			Query:       query,
			Pattern:     pattern,
		}
	}

	return pkgMods, err
}

// QueryPattern looks up the module(s) containing at least one package matching
// the given pattern at the given version. The results are sorted by module path
// length in descending order. If any proxy provides a non-empty set of candidate
// modules, no further proxies are tried.
//
// For wildcard patterns, QueryPattern looks in modules with package paths up to
// the first "..." in the pattern. For the pattern "example.com/a/b.../c",
// QueryPattern would consider prefixes of "example.com/a".
//
// If any matching package is in the main module, QueryPattern considers only
// the main module and only the version "latest", without checking for other
// possible modules.
//
// QueryPattern always returns at least one QueryResult (which may be only
// modOnly) or a non-nil error.
func QueryPattern(ctx context.Context, pattern, query string, current func(string) string, allowed AllowedFunc) (pkgMods []QueryResult, modOnly *QueryResult, err error) {
	ctx, span := trace.StartSpan(ctx, "modload.QueryPattern "+pattern+" "+query)
	defer span.Done()

	base := pattern

	firstError := func(m *search.Match) error {
		if len(m.Errs) == 0 {
			return nil
		}
		return m.Errs[0]
	}

	var match func(mod module.Version, roots []string, isLocal bool) *search.Match
	matchPattern := pkgpattern.MatchPattern(pattern)

	if i := strings.Index(pattern, "..."); i >= 0 {
		base = pathpkg.Dir(pattern[:i+3])
		if base == "." {
			return nil, nil, &WildcardInFirstElementError{Pattern: pattern, Query: query}
		}
		match = func(mod module.Version, roots []string, isLocal bool) *search.Match {
			m := search.NewMatch(pattern)
			matchPackages(ctx, m, imports.AnyTags(), omitStd, []module.Version{mod})
			return m
		}
	} else {
		match = func(mod module.Version, roots []string, isLocal bool) *search.Match {
			m := search.NewMatch(pattern)
			prefix := mod.Path
			if MainModules.Contains(mod.Path) {
				prefix = MainModules.PathPrefix(module.Version{Path: mod.Path})
			}
			for _, root := range roots {
				if _, ok, err := dirInModule(pattern, prefix, root, isLocal); err != nil {
					m.AddError(err)
				} else if ok {
					m.Pkgs = []string{pattern}
				}
			}
			return m
		}
	}

	var mainModuleMatches []module.Version
	for _, mainModule := range MainModules.Versions() {
		m := match(mainModule, modRoots, true)
		if len(m.Pkgs) > 0 {
			if query != "upgrade" && query != "patch" {
				return nil, nil, &QueryMatchesPackagesInMainModuleError{
					Pattern:  pattern,
					Query:    query,
					Packages: m.Pkgs,
				}
			}
			if err := allowed(ctx, mainModule); err != nil {
				return nil, nil, fmt.Errorf("internal error: package %s is in the main module (%s), but version is not allowed: %w", pattern, mainModule.Path, err)
			}
			return []QueryResult{{
				Mod:      mainModule,
				Rev:      &modfetch.RevInfo{Version: mainModule.Version},
				Packages: m.Pkgs,
			}}, nil, nil
		}
		if err := firstError(m); err != nil {
			return nil, nil, err
		}

		var matchesMainModule bool
		if matchPattern(mainModule.Path) {
			mainModuleMatches = append(mainModuleMatches, mainModule)
			matchesMainModule = true
		}

		if (query == "upgrade" || query == "patch") && matchesMainModule {
			if err := allowed(ctx, mainModule); err == nil {
				modOnly = &QueryResult{
					Mod: mainModule,
					Rev: &modfetch.RevInfo{Version: mainModule.Version},
				}
			}
		}
	}

	var (
		results          []QueryResult
		candidateModules = modulePrefixesExcludingTarget(base)
	)
	if len(candidateModules) == 0 {
		if modOnly != nil {
			return nil, modOnly, nil
		} else if len(mainModuleMatches) != 0 {
			return nil, nil, &QueryMatchesMainModulesError{
				MainModules: mainModuleMatches,
				Pattern:     pattern,
				Query:       query,
			}
		} else {
			return nil, nil, &PackageNotInModuleError{
				MainModules: mainModuleMatches,
				Query:       query,
				Pattern:     pattern,
			}
		}
	}

	err = modfetch.TryProxies(func(proxy string) error {
		queryModule := func(ctx context.Context, path string) (r QueryResult, err error) {
			ctx, span := trace.StartSpan(ctx, "modload.QueryPattern.queryModule ["+proxy+"] "+path)
			defer span.Done()

			pathCurrent := current(path)
			r.Mod.Path = path
			r.Rev, err = queryProxy(ctx, proxy, path, query, pathCurrent, allowed, nil)
			if err != nil {
				return r, err
			}
			r.Mod.Version = r.Rev.Version
			if gover.IsToolchain(r.Mod.Path) {
				return r, nil
			}
			root, isLocal, err := fetch(ctx, r.Mod)
			if err != nil {
				return r, err
			}
			m := match(r.Mod, []string{root}, isLocal)
			r.Packages = m.Pkgs
			if len(r.Packages) == 0 && !matchPattern(path) {
				if err := firstError(m); err != nil {
					return r, err
				}
				replacement := Replacement(r.Mod)
				return r, &PackageNotInModuleError{
					Mod:         r.Mod,
					Replacement: replacement,
					Query:       query,
					Pattern:     pattern,
				}
			}
			return r, nil
		}

		allResults, err := queryPrefixModules(ctx, candidateModules, queryModule)
		results = allResults[:0]
		for _, r := range allResults {
			if len(r.Packages) == 0 {
				modOnly = &r
			} else {
				results = append(results, r)
			}
		}
		return err
	})

	if len(mainModuleMatches) > 0 && len(results) == 0 && modOnly == nil && errors.Is(err, fs.ErrNotExist) {
		return nil, nil, &QueryMatchesMainModulesError{
			Pattern: pattern,
			Query:   query,
		}
	}
	return slices.Clip(results), modOnly, err
}

// modulePrefixesExcludingTarget returns all prefixes of path that may plausibly
// exist as a module, excluding targetPrefix but otherwise including path
// itself, sorted by descending length. Prefixes that are not valid module paths
// but are valid package paths (like "m" or "example.com/.gen") are included,
// since they might be replaced.
func modulePrefixesExcludingTarget(path string) []string {
	prefixes := make([]string, 0, strings.Count(path, "/")+1)

	mainModulePrefixes := make(map[string]bool)
	for _, m := range MainModules.Versions() {
		mainModulePrefixes[m.Path] = true
	}

	for {
		if !mainModulePrefixes[path] {
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

func queryPrefixModules(ctx context.Context, candidateModules []string, queryModule func(ctx context.Context, path string) (QueryResult, error)) (found []QueryResult, err error) {
	ctx, span := trace.StartSpan(ctx, "modload.queryPrefixModules")
	defer span.Done()

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
		ctx := trace.StartGoroutine(ctx)
		go func(p string, r *result) {
			r.QueryResult, r.err = queryModule(ctx, p)
			wg.Done()
		}(p, &results[i])
	}
	wg.Wait()

	// Classify the results. In case of failure, identify the error that the user
	// is most likely to find helpful: the most useful class of error at the
	// longest matching path.
	var (
		noPackage      *PackageNotInModuleError
		noVersion      *NoMatchingVersionError
		noPatchBase    *NoPatchBaseError
		invalidPath    *module.InvalidPathError // see comment in case below
		invalidVersion error
		notExistErr    error
	)
	for _, r := range results {
		switch rErr := r.err.(type) {
		case nil:
			found = append(found, r.QueryResult)
		case *PackageNotInModuleError:
			// Given the option, prefer to attribute “package not in module”
			// to modules other than the main one.
			if noPackage == nil || MainModules.Contains(noPackage.Mod.Path) {
				noPackage = rErr
			}
		case *NoMatchingVersionError:
			if noVersion == nil {
				noVersion = rErr
			}
		case *NoPatchBaseError:
			if noPatchBase == nil {
				noPatchBase = rErr
			}
		case *module.InvalidPathError:
			// The prefix was not a valid module path, and there was no replacement.
			// Prefixes like this may appear in candidateModules, since we handle
			// replaced modules that weren't required in the repo lookup process
			// (see lookupRepo).
			//
			// A shorter prefix may be a valid module path and may contain a valid
			// import path, so this is a low-priority error.
			if invalidPath == nil {
				invalidPath = rErr
			}
		default:
			if errors.Is(rErr, fs.ErrNotExist) {
				if notExistErr == nil {
					notExistErr = rErr
				}
			} else if iv := (*module.InvalidVersionError)(nil); errors.As(rErr, &iv) {
				if invalidVersion == nil {
					invalidVersion = rErr
				}
			} else if err == nil {
				if len(found) > 0 || noPackage != nil {
					// golang.org/issue/34094: If we have already found a module that
					// could potentially contain the target package, ignore unclassified
					// errors for modules with shorter paths.

					// golang.org/issue/34383 is a special case of this: if we have
					// already found example.com/foo/v2@v2.0.0 with a matching go.mod
					// file, ignore the error from example.com/foo@v2.0.0.
				} else {
					err = r.err
				}
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
		case noPatchBase != nil:
			err = noPatchBase
		case invalidPath != nil:
			err = invalidPath
		case invalidVersion != nil:
			err = invalidVersion
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
// NOTE: NoMatchingVersionError MUST NOT implement Is(fs.ErrNotExist).
//
// If the module came from a proxy, that proxy had to return a successful status
// code for the versions it knows about, and thus did not have the opportunity
// to return a non-400 status code to suppress fallback.
type NoMatchingVersionError struct {
	query, current string
}

func (e *NoMatchingVersionError) Error() string {
	currentSuffix := ""
	if (e.query == "upgrade" || e.query == "patch") && e.current != "" && e.current != "none" {
		currentSuffix = fmt.Sprintf(" (current version is %s)", e.current)
	}
	return fmt.Sprintf("no matching versions for query %q", e.query) + currentSuffix
}

// A NoPatchBaseError indicates that Query was called with the query "patch"
// but with a current version of "" or "none".
type NoPatchBaseError struct {
	path string
}

func (e *NoPatchBaseError) Error() string {
	return fmt.Sprintf(`can't query version "patch" of module %s: no existing version is required`, e.path)
}

// A WildcardInFirstElementError indicates that a pattern passed to QueryPattern
// had a wildcard in its first path element, and therefore had no pattern-prefix
// modules to search in.
type WildcardInFirstElementError struct {
	Pattern string
	Query   string
}

func (e *WildcardInFirstElementError) Error() string {
	return fmt.Sprintf("no modules to query for %s@%s because first path element contains a wildcard", e.Pattern, e.Query)
}

// A PackageNotInModuleError indicates that QueryPattern found a candidate
// module at the requested version, but that module did not contain any packages
// matching the requested pattern.
//
// NOTE: PackageNotInModuleError MUST NOT implement Is(fs.ErrNotExist).
//
// If the module came from a proxy, that proxy had to return a successful status
// code for the versions it knows about, and thus did not have the opportunity
// to return a non-400 status code to suppress fallback.
type PackageNotInModuleError struct {
	MainModules []module.Version
	Mod         module.Version
	Replacement module.Version
	Query       string
	Pattern     string
}

func (e *PackageNotInModuleError) Error() string {
	if len(e.MainModules) > 0 {
		prefix := "workspace modules do"
		if len(e.MainModules) == 1 {
			prefix = fmt.Sprintf("main module (%s) does", e.MainModules[0])
		}
		if strings.Contains(e.Pattern, "...") {
			return fmt.Sprintf("%s not contain packages matching %s", prefix, e.Pattern)
		}
		return fmt.Sprintf("%s not contain package %s", prefix, e.Pattern)
	}

	found := ""
	if r := e.Replacement; r.Path != "" {
		replacement := r.Path
		if r.Version != "" {
			replacement = fmt.Sprintf("%s@%s", r.Path, r.Version)
		}
		if e.Query == e.Mod.Version {
			found = fmt.Sprintf(" (replaced by %s)", replacement)
		} else {
			found = fmt.Sprintf(" (%s, replaced by %s)", e.Mod.Version, replacement)
		}
	} else if e.Query != e.Mod.Version {
		found = fmt.Sprintf(" (%s)", e.Mod.Version)
	}

	if strings.Contains(e.Pattern, "...") {
		return fmt.Sprintf("module %s@%s found%s, but does not contain packages matching %s", e.Mod.Path, e.Query, found, e.Pattern)
	}
	return fmt.Sprintf("module %s@%s found%s, but does not contain package %s", e.Mod.Path, e.Query, found, e.Pattern)
}

func (e *PackageNotInModuleError) ImportPath() string {
	if !strings.Contains(e.Pattern, "...") {
		return e.Pattern
	}
	return ""
}

// versionHasGoMod returns whether a version has a go.mod file.
//
// versionHasGoMod fetches the go.mod file (possibly a fake) and true if it
// contains anything other than a module directive with the same path. When a
// module does not have a real go.mod file, the go command acts as if it had one
// that only contained a module directive. Normal go.mod files created after
// 1.12 at least have a go directive.
//
// This function is a heuristic, since it's possible to commit a file that would
// pass this test. However, we only need a heuristic for determining whether
// +incompatible versions may be "latest", which is what this function is used
// for.
//
// This heuristic is useful for two reasons: first, when using a proxy,
// this lets us fetch from the .mod endpoint which is much faster than the .zip
// endpoint. The .mod file is used anyway, even if the .zip file contains a
// go.mod with different content. Second, if we don't fetch the .zip, then
// we don't need to verify it in go.sum. This makes 'go list -m -u' faster
// and simpler.
func versionHasGoMod(_ context.Context, m module.Version) (bool, error) {
	_, data, err := rawGoModData(m)
	if err != nil {
		return false, err
	}
	isFake := bytes.Equal(data, modfetch.LegacyGoMod(m.Path))
	return !isFake, nil
}

// A versionRepo is a subset of modfetch.Repo that can report information about
// available versions, but cannot fetch specific source files.
type versionRepo interface {
	ModulePath() string
	CheckReuse(context.Context, *codehost.Origin) error
	Versions(ctx context.Context, prefix string) (*modfetch.Versions, error)
	Stat(ctx context.Context, rev string) (*modfetch.RevInfo, error)
	Latest(context.Context) (*modfetch.RevInfo, error)
}

var _ versionRepo = modfetch.Repo(nil)

func lookupRepo(ctx context.Context, proxy, path string) (repo versionRepo, err error) {
	if path != "go" && path != "toolchain" {
		err = module.CheckPath(path)
	}
	if err == nil {
		repo = modfetch.Lookup(ctx, proxy, path)
	} else {
		repo = emptyRepo{path: path, err: err}
	}

	if MainModules == nil {
		return repo, err
	} else if _, ok := MainModules.HighestReplaced()[path]; ok {
		return &replacementRepo{repo: repo}, nil
	}

	return repo, err
}

// An emptyRepo is a versionRepo that contains no versions.
type emptyRepo struct {
	path string
	err  error
}

var _ versionRepo = emptyRepo{}

func (er emptyRepo) ModulePath() string { return er.path }
func (er emptyRepo) CheckReuse(ctx context.Context, old *codehost.Origin) error {
	return fmt.Errorf("empty repo")
}
func (er emptyRepo) Versions(ctx context.Context, prefix string) (*modfetch.Versions, error) {
	return &modfetch.Versions{}, nil
}
func (er emptyRepo) Stat(ctx context.Context, rev string) (*modfetch.RevInfo, error) {
	return nil, er.err
}
func (er emptyRepo) Latest(ctx context.Context) (*modfetch.RevInfo, error) { return nil, er.err }

// A replacementRepo augments a versionRepo to include the replacement versions
// (if any) found in the main module's go.mod file.
//
// A replacementRepo suppresses "not found" errors for otherwise-nonexistent
// modules, so a replacementRepo should only be constructed for a module that
// actually has one or more valid replacements.
type replacementRepo struct {
	repo versionRepo
}

var _ versionRepo = (*replacementRepo)(nil)

func (rr *replacementRepo) ModulePath() string { return rr.repo.ModulePath() }

func (rr *replacementRepo) CheckReuse(ctx context.Context, old *codehost.Origin) error {
	return fmt.Errorf("replacement repo")
}

// Versions returns the versions from rr.repo augmented with any matching
// replacement versions.
func (rr *replacementRepo) Versions(ctx context.Context, prefix string) (*modfetch.Versions, error) {
	repoVersions, err := rr.repo.Versions(ctx, prefix)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		repoVersions = new(modfetch.Versions)
	}

	versions := repoVersions.List
	for _, mm := range MainModules.Versions() {
		if index := MainModules.Index(mm); index != nil && len(index.replace) > 0 {
			path := rr.ModulePath()
			for m := range index.replace {
				if m.Path == path && strings.HasPrefix(m.Version, prefix) && m.Version != "" && !module.IsPseudoVersion(m.Version) {
					versions = append(versions, m.Version)
				}
			}
		}
	}

	if len(versions) == len(repoVersions.List) { // replacement versions added
		return repoVersions, nil
	}

	path := rr.ModulePath()
	sort.Slice(versions, func(i, j int) bool {
		return gover.ModCompare(path, versions[i], versions[j]) < 0
	})
	str.Uniq(&versions)
	return &modfetch.Versions{List: versions}, nil
}

func (rr *replacementRepo) Stat(ctx context.Context, rev string) (*modfetch.RevInfo, error) {
	info, err := rr.repo.Stat(ctx, rev)
	if err == nil {
		return info, err
	}
	var hasReplacements bool
	for _, v := range MainModules.Versions() {
		if index := MainModules.Index(v); index != nil && len(index.replace) > 0 {
			hasReplacements = true
		}
	}
	if !hasReplacements {
		return info, err
	}

	v := module.CanonicalVersion(rev)
	if v != rev {
		// The replacements in the go.mod file list only canonical semantic versions,
		// so a non-canonical version can't possibly have a replacement.
		return info, err
	}

	path := rr.ModulePath()
	_, pathMajor, ok := module.SplitPathVersion(path)
	if ok && pathMajor == "" {
		if err := module.CheckPathMajor(v, pathMajor); err != nil && semver.Build(v) == "" {
			v += "+incompatible"
		}
	}

	if r := Replacement(module.Version{Path: path, Version: v}); r.Path == "" {
		return info, err
	}
	return rr.replacementStat(v)
}

func (rr *replacementRepo) Latest(ctx context.Context) (*modfetch.RevInfo, error) {
	info, err := rr.repo.Latest(ctx)
	path := rr.ModulePath()

	if v, ok := MainModules.HighestReplaced()[path]; ok {
		if v == "" {
			// The only replacement is a wildcard that doesn't specify a version, so
			// synthesize a pseudo-version with an appropriate major version and a
			// timestamp below any real timestamp. That way, if the main module is
			// used from within some other module, the user will be able to upgrade
			// the requirement to any real version they choose.
			if _, pathMajor, ok := module.SplitPathVersion(path); ok && len(pathMajor) > 0 {
				v = module.PseudoVersion(pathMajor[1:], "", time.Time{}, "000000000000")
			} else {
				v = module.PseudoVersion("v0", "", time.Time{}, "000000000000")
			}
		}

		if err != nil || gover.ModCompare(path, v, info.Version) > 0 {
			return rr.replacementStat(v)
		}
	}

	return info, err
}

func (rr *replacementRepo) replacementStat(v string) (*modfetch.RevInfo, error) {
	rev := &modfetch.RevInfo{Version: v}
	if module.IsPseudoVersion(v) {
		rev.Time, _ = module.PseudoVersionTime(v)
		rev.Short, _ = module.PseudoVersionRev(v)
	}
	return rev, nil
}

// A QueryMatchesMainModulesError indicates that a query requests
// a version of the main module that cannot be satisfied.
// (The main module's version cannot be changed.)
type QueryMatchesMainModulesError struct {
	MainModules []module.Version
	Pattern     string
	Query       string
}

func (e *QueryMatchesMainModulesError) Error() string {
	if MainModules.Contains(e.Pattern) {
		return fmt.Sprintf("can't request version %q of the main module (%s)", e.Query, e.Pattern)
	}

	plural := ""
	mainModulePaths := make([]string, len(e.MainModules))
	for i := range e.MainModules {
		mainModulePaths[i] = e.MainModules[i].Path
	}
	if len(e.MainModules) > 1 {
		plural = "s"
	}
	return fmt.Sprintf("can't request version %q of pattern %q that includes the main module%s (%s)", e.Query, e.Pattern, plural, strings.Join(mainModulePaths, ", "))
}

// A QueryUpgradesAllError indicates that a query requests
// an upgrade on the all pattern.
// (The main module's version cannot be changed.)
type QueryUpgradesAllError struct {
	MainModules []module.Version
	Query       string
}

func (e *QueryUpgradesAllError) Error() string {
	var plural string = ""
	if len(e.MainModules) != 1 {
		plural = "s"
	}

	return fmt.Sprintf("can't request version %q of pattern \"all\" that includes the main module%s", e.Query, plural)
}

// A QueryMatchesPackagesInMainModuleError indicates that a query cannot be
// satisfied because it matches one or more packages found in the main module.
type QueryMatchesPackagesInMainModuleError struct {
	Pattern  string
	Query    string
	Packages []string
}

func (e *QueryMatchesPackagesInMainModuleError) Error() string {
	if len(e.Packages) > 1 {
		return fmt.Sprintf("pattern %s matches %d packages in the main module, so can't request version %s", e.Pattern, len(e.Packages), e.Query)
	}

	if search.IsMetaPackage(e.Pattern) || strings.Contains(e.Pattern, "...") {
		return fmt.Sprintf("pattern %s matches package %s in the main module, so can't request version %s", e.Pattern, e.Packages[0], e.Query)
	}

	return fmt.Sprintf("package %s is in the main module, so can't request version %s", e.Packages[0], e.Query)
}
