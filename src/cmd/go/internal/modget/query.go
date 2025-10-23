// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modget

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"

	"cmd/go/internal/base"
	"cmd/go/internal/gover"
	"cmd/go/internal/modload"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/internal/pkgpattern"

	"golang.org/x/mod/module"
)

// A query describes a command-line argument and the modules and/or packages
// to which that argument may resolve..
type query struct {
	// raw is the original argument, to be printed in error messages.
	raw string

	// rawVersion is the portion of raw corresponding to version, if any
	rawVersion string

	// pattern is the part of the argument before "@" (or the whole argument
	// if there is no "@"), which may match either packages (preferred) or
	// modules (if no matching packages).
	//
	// The pattern may also be "-u", for the synthetic query representing the -u
	// (“upgrade”)flag.
	pattern string

	// patternIsLocal indicates whether pattern is restricted to match only paths
	// local to the main module, such as absolute filesystem paths or paths
	// beginning with './'.
	//
	// A local pattern must resolve to one or more packages in the main module.
	patternIsLocal bool

	// version is the part of the argument after "@", or an implied
	// "upgrade" or "patch" if there is no "@". version specifies the
	// module version to get.
	version string

	// matchWildcard, if non-nil, reports whether pattern, which must be a
	// wildcard (with the substring "..."), matches the given package or module
	// path.
	matchWildcard func(path string) bool

	// canMatchWildcardInModule, if non-nil, reports whether the module with the given
	// path could lexically contain a package matching pattern, which must be a
	// wildcard.
	canMatchWildcardInModule func(mPath string) bool

	// conflict is the first query identified as incompatible with this one.
	// conflict forces one or more of the modules matching this query to a
	// version that does not match version.
	conflict *query

	// candidates is a list of sets of alternatives for a path that matches (or
	// contains packages that match) the pattern. The query can be resolved by
	// choosing exactly one alternative from each set in the list.
	//
	// A path-literal query results in only one set: the path itself, which
	// may resolve to either a package path or a module path.
	//
	// A wildcard query results in one set for each matching module path, each
	// module for which the matching version contains at least one matching
	// package, and (if no other modules match) one candidate set for the pattern
	// overall if no existing match is identified in the build list.
	//
	// A query for pattern "all" results in one set for each package transitively
	// imported by the main module.
	//
	// The special query for the "-u" flag results in one set for each
	// otherwise-unconstrained package that has available upgrades.
	candidates   []pathSet
	candidatesMu sync.Mutex

	// pathSeen ensures that only one pathSet is added to the query per
	// unique path.
	pathSeen sync.Map

	// resolved contains the set of modules whose versions have been determined by
	// this query, in the order in which they were determined.
	//
	// The resolver examines the candidate sets for each query, resolving one
	// module per candidate set in a way that attempts to avoid obvious conflicts
	// between the versions resolved by different queries.
	resolved []module.Version

	// matchesPackages is true if the resolved modules provide at least one
	// package matching q.pattern.
	matchesPackages bool
}

// A pathSet describes the possible options for resolving a specific path
// to a package and/or module.
type pathSet struct {
	// path is a package (if "all" or "-u" or a non-wildcard) or module (if
	// wildcard) path that could be resolved by adding any of the modules in this
	// set. For a wildcard pattern that so far matches no packages, the path is
	// the wildcard pattern itself.
	//
	// Each path must occur only once in a query's candidate sets, and the path is
	// added implicitly to each pathSet returned to pathOnce.
	path string

	// pkgMods is a set of zero or more modules, each of which contains the
	// package with the indicated path. Due to the requirement that imports be
	// unambiguous, only one such module can be in the build list, and all others
	// must be excluded.
	pkgMods []module.Version

	// mod is either the zero Version, or a module that does not contain any
	// packages matching the query but for which the module path itself
	// matches the query pattern.
	//
	// We track this module separately from pkgMods because, all else equal, we
	// prefer to match a query to a package rather than just a module. Also,
	// unlike the modules in pkgMods, this module does not inherently exclude
	// any other module in pkgMods.
	mod module.Version

	err error
}

// errSet returns a pathSet containing the given error.
func errSet(err error) pathSet { return pathSet{err: err} }

// newQuery returns a new query parsed from the raw argument,
// which must be either path or path@version.
func newQuery(loaderstate *modload.State, raw string) (*query, error) {
	pattern, rawVers, found := strings.Cut(raw, "@")
	if found && (strings.Contains(rawVers, "@") || rawVers == "") {
		return nil, fmt.Errorf("invalid module version syntax %q", raw)
	}

	// If no version suffix is specified, assume @upgrade.
	// If -u=patch was specified, assume @patch instead.
	version := rawVers
	if version == "" {
		if getU.version == "" {
			version = "upgrade"
		} else {
			version = getU.version
		}
	}

	q := &query{
		raw:            raw,
		rawVersion:     rawVers,
		pattern:        pattern,
		patternIsLocal: filepath.IsAbs(pattern) || search.IsRelativePath(pattern),
		version:        version,
	}
	if strings.Contains(q.pattern, "...") {
		q.matchWildcard = pkgpattern.MatchPattern(q.pattern)
		q.canMatchWildcardInModule = pkgpattern.TreeCanMatchPattern(q.pattern)
	}
	if err := q.validate(loaderstate); err != nil {
		return q, err
	}
	return q, nil
}

// validate reports a non-nil error if q is not sensible and well-formed.
func (q *query) validate(loaderstate *modload.State) error {
	if q.patternIsLocal {
		if q.rawVersion != "" {
			return fmt.Errorf("can't request explicit version %q of path %q in main module", q.rawVersion, q.pattern)
		}
		return nil
	}

	if q.pattern == "all" {
		// If there is no main module, "all" is not meaningful.
		if !modload.HasModRoot(loaderstate) {
			return fmt.Errorf(`cannot match "all": %v`, modload.ErrNoModRoot)
		}
		if !versionOkForMainModule(q.version) {
			// TODO(bcmills): "all@none" seems like a totally reasonable way to
			// request that we remove all module requirements, leaving only the main
			// module and standard library. Perhaps we should implement that someday.
			return &modload.QueryUpgradesAllError{
				MainModules: loaderstate.MainModules.Versions(),
				Query:       q.version,
			}
		}
	}

	if search.IsMetaPackage(q.pattern) && q.pattern != "all" {
		if q.pattern != q.raw {
			if q.pattern == "tool" {
				return fmt.Errorf("can't request explicit version of \"tool\" pattern")
			}
			return fmt.Errorf("can't request explicit version of standard-library pattern %q", q.pattern)
		}
	}

	return nil
}

// String returns the original argument from which q was parsed.
func (q *query) String() string { return q.raw }

// ResolvedString returns a string describing m as a resolved match for q.
func (q *query) ResolvedString(m module.Version) string {
	if m.Path != q.pattern {
		if m.Version != q.version {
			return fmt.Sprintf("%v (matching %s@%s)", m, q.pattern, q.version)
		}
		return fmt.Sprintf("%v (matching %v)", m, q)
	}
	if m.Version != q.version {
		return fmt.Sprintf("%s@%s (%s)", q.pattern, q.version, m.Version)
	}
	return q.String()
}

// isWildcard reports whether q is a pattern that can match multiple paths.
func (q *query) isWildcard() bool {
	return q.matchWildcard != nil || (q.patternIsLocal && strings.Contains(q.pattern, "..."))
}

// matchesPath reports whether the given path matches q.pattern.
func (q *query) matchesPath(path string) bool {
	if q.matchWildcard != nil && !gover.IsToolchain(path) {
		return q.matchWildcard(path)
	}
	return path == q.pattern
}

// canMatchInModule reports whether the given module path can potentially
// contain q.pattern.
func (q *query) canMatchInModule(mPath string) bool {
	if gover.IsToolchain(mPath) {
		return false
	}
	if q.canMatchWildcardInModule != nil {
		return q.canMatchWildcardInModule(mPath)
	}
	return str.HasPathPrefix(q.pattern, mPath)
}

// pathOnce invokes f to generate the pathSet for the given path,
// if one is still needed.
//
// Note that, unlike sync.Once, pathOnce does not guarantee that a concurrent
// call to f for the given path has completed on return.
//
// pathOnce is safe for concurrent use by multiple goroutines, but note that
// multiple concurrent calls will result in the sets being added in
// nondeterministic order.
func (q *query) pathOnce(path string, f func() pathSet) {
	if _, dup := q.pathSeen.LoadOrStore(path, nil); dup {
		return
	}

	cs := f()

	if len(cs.pkgMods) > 0 || cs.mod != (module.Version{}) || cs.err != nil {
		cs.path = path
		q.candidatesMu.Lock()
		q.candidates = append(q.candidates, cs)
		q.candidatesMu.Unlock()
	}
}

// reportError logs err concisely using base.Errorf.
func reportError(q *query, err error) {
	errStr := err.Error()

	// If err already mentions all of the relevant parts of q, just log err to
	// reduce stutter. Otherwise, log both q and err.
	//
	// TODO(bcmills): Use errors.AsType to unpack these errors instead of parsing
	// strings with regular expressions.

	if !utf8.ValidString(q.pattern) || !utf8.ValidString(q.version) {
		base.Errorf("go: %s", errStr)
		return
	}

	patternRE := regexp.MustCompile("(?m)(?:[ \t(\"`]|^)" + regexp.QuoteMeta(q.pattern) + "(?:[ @:;)\"`]|$)")
	if patternRE.MatchString(errStr) {
		if q.rawVersion == "" {
			base.Errorf("go: %s", errStr)
			return
		}

		versionRE := regexp.MustCompile("(?m)(?:[ @(\"`]|^)" + regexp.QuoteMeta(q.version) + "(?:[ :;)\"`]|$)")
		if versionRE.MatchString(errStr) {
			base.Errorf("go: %s", errStr)
			return
		}
	}

	if qs := q.String(); qs != "" {
		base.Errorf("go: %s: %s", qs, errStr)
	} else {
		base.Errorf("go: %s", errStr)
	}
}

func reportConflict(pq *query, m module.Version, conflict versionReason) {
	if pq.conflict != nil {
		// We've already reported a conflict for the proposed query.
		// Don't report it again, even if it has other conflicts.
		return
	}
	pq.conflict = conflict.reason

	proposed := versionReason{
		version: m.Version,
		reason:  pq,
	}
	if pq.isWildcard() && !conflict.reason.isWildcard() {
		// Prefer to report the specific path first and the wildcard second.
		proposed, conflict = conflict, proposed
	}
	reportError(pq, &conflictError{
		mPath:    m.Path,
		proposed: proposed,
		conflict: conflict,
	})
}

type conflictError struct {
	mPath    string
	proposed versionReason
	conflict versionReason
}

func (e *conflictError) Error() string {
	argStr := func(q *query, v string) string {
		if v != q.version {
			return fmt.Sprintf("%s@%s (%s)", q.pattern, q.version, v)
		}
		return q.String()
	}

	pq := e.proposed.reason
	rq := e.conflict.reason
	modDetail := ""
	if e.mPath != pq.pattern {
		modDetail = fmt.Sprintf("for module %s, ", e.mPath)
	}

	return fmt.Sprintf("%s%s conflicts with %s",
		modDetail,
		argStr(pq, e.proposed.version),
		argStr(rq, e.conflict.version))
}

func versionOkForMainModule(version string) bool {
	return version == "upgrade" || version == "patch"
}
