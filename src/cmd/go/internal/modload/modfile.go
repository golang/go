// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/par"
	"cmd/go/internal/trace"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

const (
	// narrowAllVersionV is the Go version (plus leading "v") at which the
	// module-module "all" pattern no longer closes over the dependencies of
	// tests outside of the main module.
	narrowAllVersionV = "v1.16"

	// lazyLoadingVersionV is the Go version (plus leading "v") at which a
	// module's go.mod file is expected to list explicit requirements on every
	// module that provides any package transitively imported by that module.
	lazyLoadingVersionV = "v1.17"

	// separateIndirectVersionV is the Go version (plus leading "v") at which
	// "// indirect" dependencies are added in a block separate from the direct
	// ones. See https://golang.org/issue/45965.
	separateIndirectVersionV = "v1.17"
)

const (
	// go117EnableLazyLoading toggles whether lazy-loading code paths should be
	// active. It will be removed once the lazy loading implementation is stable
	// and well-tested.
	go117EnableLazyLoading = true

	// go1117LazyTODO is a constant that exists only until lazy loading is
	// implemented. Its use indicates a condition that will need to change if the
	// main module is lazy.
	go117LazyTODO = false
)

var modFile *modfile.File

// modFileGoVersion returns the (non-empty) Go version at which the requirements
// in modFile are intepreted, or the latest Go version if modFile is nil.
func modFileGoVersion() string {
	if modFile == nil {
		return LatestGoVersion()
	}
	if modFile.Go == nil || modFile.Go.Version == "" {
		// The main module necessarily has a go.mod file, and that file lacks a
		// 'go' directive. The 'go' command has been adding that directive
		// automatically since Go 1.12, so this module either dates to Go 1.11 or
		// has been erroneously hand-edited.
		//
		// The semantics of the go.mod file are more-or-less the same from Go 1.11
		// through Go 1.16, changing at 1.17 for lazy loading. So even though a
		// go.mod file without a 'go' directive is theoretically a Go 1.11 file,
		// scripts may assume that it ends up as a Go 1.16 module.
		return "1.16"
	}
	return modFile.Go.Version
}

// A modFileIndex is an index of data corresponding to a modFile
// at a specific point in time.
type modFileIndex struct {
	data            []byte
	dataNeedsFix    bool // true if fixVersion applied a change while parsing data
	module          module.Version
	goVersionV      string // GoVersion with "v" prefix
	require         map[module.Version]requireMeta
	replace         map[module.Version]module.Version
	highestReplaced map[string]string // highest replaced version of each module path; empty string for wildcard-only replacements
	exclude         map[module.Version]bool
}

// index is the index of the go.mod file as of when it was last read or written.
var index *modFileIndex

type requireMeta struct {
	indirect bool
}

// A modDepth indicates which dependencies should be loaded for a go.mod file.
type modDepth uint8

const (
	lazy  modDepth = iota // load dependencies only as needed
	eager                 // load all transitive dependencies eagerly
)

func modDepthFromGoVersion(goVersion string) modDepth {
	if !go117EnableLazyLoading {
		return eager
	}
	if semver.Compare("v"+goVersion, lazyLoadingVersionV) < 0 {
		return eager
	}
	return lazy
}

// CheckAllowed returns an error equivalent to ErrDisallowed if m is excluded by
// the main module's go.mod or retracted by its author. Most version queries use
// this to filter out versions that should not be used.
func CheckAllowed(ctx context.Context, m module.Version) error {
	if err := CheckExclusions(ctx, m); err != nil {
		return err
	}
	if err := CheckRetractions(ctx, m); err != nil {
		return err
	}
	return nil
}

// ErrDisallowed is returned by version predicates passed to Query and similar
// functions to indicate that a version should not be considered.
var ErrDisallowed = errors.New("disallowed module version")

// CheckExclusions returns an error equivalent to ErrDisallowed if module m is
// excluded by the main module's go.mod file.
func CheckExclusions(ctx context.Context, m module.Version) error {
	if index != nil && index.exclude[m] {
		return module.VersionError(m, errExcluded)
	}
	return nil
}

var errExcluded = &excludedError{}

type excludedError struct{}

func (e *excludedError) Error() string     { return "excluded by go.mod" }
func (e *excludedError) Is(err error) bool { return err == ErrDisallowed }

// CheckRetractions returns an error if module m has been retracted by
// its author.
func CheckRetractions(ctx context.Context, m module.Version) (err error) {
	defer func() {
		if retractErr := (*ModuleRetractedError)(nil); err == nil || errors.As(err, &retractErr) {
			return
		}
		// Attribute the error to the version being checked, not the version from
		// which the retractions were to be loaded.
		if mErr := (*module.ModuleError)(nil); errors.As(err, &mErr) {
			err = mErr.Err
		}
		err = &retractionLoadingError{m: m, err: err}
	}()

	if m.Version == "" {
		// Main module, standard library, or file replacement module.
		// Cannot be retracted.
		return nil
	}
	if repl := Replacement(module.Version{Path: m.Path}); repl.Path != "" {
		// All versions of the module were replaced.
		// Don't load retractions, since we'd just load the replacement.
		return nil
	}

	// Find the latest available version of the module, and load its go.mod. If
	// the latest version is replaced, we'll load the replacement.
	//
	// If there's an error loading the go.mod, we'll return it here. These errors
	// should generally be ignored by callers since they happen frequently when
	// we're offline. These errors are not equivalent to ErrDisallowed, so they
	// may be distinguished from retraction errors.
	//
	// We load the raw file here: the go.mod file may have a different module
	// path that we expect if the module or its repository was renamed.
	// We still want to apply retractions to other aliases of the module.
	rm, err := queryLatestVersionIgnoringRetractions(ctx, m.Path)
	if err != nil {
		return err
	}
	summary, err := rawGoModSummary(rm)
	if err != nil {
		return err
	}

	var rationale []string
	isRetracted := false
	for _, r := range summary.retract {
		if semver.Compare(r.Low, m.Version) <= 0 && semver.Compare(m.Version, r.High) <= 0 {
			isRetracted = true
			if r.Rationale != "" {
				rationale = append(rationale, r.Rationale)
			}
		}
	}
	if isRetracted {
		return module.VersionError(m, &ModuleRetractedError{Rationale: rationale})
	}
	return nil
}

type ModuleRetractedError struct {
	Rationale []string
}

func (e *ModuleRetractedError) Error() string {
	msg := "retracted by module author"
	if len(e.Rationale) > 0 {
		// This is meant to be a short error printed on a terminal, so just
		// print the first rationale.
		msg += ": " + ShortMessage(e.Rationale[0], "retracted by module author")
	}
	return msg
}

func (e *ModuleRetractedError) Is(err error) bool {
	return err == ErrDisallowed
}

type retractionLoadingError struct {
	m   module.Version
	err error
}

func (e *retractionLoadingError) Error() string {
	return fmt.Sprintf("loading module retractions for %v: %v", e.m, e.err)
}

func (e *retractionLoadingError) Unwrap() error {
	return e.err
}

// ShortMessage returns a string from go.mod (for example, a retraction
// rationale or deprecation message) that is safe to print in a terminal.
//
// If the given string is empty, ShortMessage returns the given default. If the
// given string is too long or contains non-printable characters, ShortMessage
// returns a hard-coded string.
func ShortMessage(message, emptyDefault string) string {
	const maxLen = 500
	if i := strings.Index(message, "\n"); i >= 0 {
		message = message[:i]
	}
	message = strings.TrimSpace(message)
	if message == "" {
		return emptyDefault
	}
	if len(message) > maxLen {
		return "(message omitted: too long)"
	}
	for _, r := range message {
		if !unicode.IsGraphic(r) && !unicode.IsSpace(r) {
			return "(message omitted: contains non-printable characters)"
		}
	}
	// NOTE: the go.mod parser rejects invalid UTF-8, so we don't check that here.
	return message
}

// CheckDeprecation returns a deprecation message from the go.mod file of the
// latest version of the given module. Deprecation messages are comments
// before or on the same line as the module directives that start with
// "Deprecated:" and run until the end of the paragraph.
//
// CheckDeprecation returns an error if the message can't be loaded.
// CheckDeprecation returns "", nil if there is no deprecation message.
func CheckDeprecation(ctx context.Context, m module.Version) (deprecation string, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("loading deprecation for %s: %w", m.Path, err)
		}
	}()

	if m.Version == "" {
		// Main module, standard library, or file replacement module.
		// Don't look up deprecation.
		return "", nil
	}
	if repl := Replacement(module.Version{Path: m.Path}); repl.Path != "" {
		// All versions of the module were replaced.
		// We'll look up deprecation separately for the replacement.
		return "", nil
	}

	latest, err := queryLatestVersionIgnoringRetractions(ctx, m.Path)
	if err != nil {
		return "", err
	}
	summary, err := rawGoModSummary(latest)
	if err != nil {
		return "", err
	}
	return summary.deprecated, nil
}

// Replacement returns the replacement for mod, if any, from go.mod.
// If there is no replacement for mod, Replacement returns
// a module.Version with Path == "".
func Replacement(mod module.Version) module.Version {
	if index != nil {
		if r, ok := index.replace[mod]; ok {
			return r
		}
		if r, ok := index.replace[module.Version{Path: mod.Path}]; ok {
			return r
		}
	}
	return module.Version{}
}

// resolveReplacement returns the module actually used to load the source code
// for m: either m itself, or the replacement for m (iff m is replaced).
func resolveReplacement(m module.Version) module.Version {
	if r := Replacement(m); r.Path != "" {
		return r
	}
	return m
}

// indexModFile rebuilds the index of modFile.
// If modFile has been changed since it was first read,
// modFile.Cleanup must be called before indexModFile.
func indexModFile(data []byte, modFile *modfile.File, needsFix bool) *modFileIndex {
	i := new(modFileIndex)
	i.data = data
	i.dataNeedsFix = needsFix

	i.module = module.Version{}
	if modFile.Module != nil {
		i.module = modFile.Module.Mod
	}

	i.goVersionV = ""
	if modFile.Go == nil {
		rawGoVersion.Store(Target, "")
	} else {
		// We're going to use the semver package to compare Go versions, so go ahead
		// and add the "v" prefix it expects once instead of every time.
		i.goVersionV = "v" + modFile.Go.Version
		rawGoVersion.Store(Target, modFile.Go.Version)
	}

	i.require = make(map[module.Version]requireMeta, len(modFile.Require))
	for _, r := range modFile.Require {
		i.require[r.Mod] = requireMeta{indirect: r.Indirect}
	}

	i.replace = make(map[module.Version]module.Version, len(modFile.Replace))
	for _, r := range modFile.Replace {
		if prev, dup := i.replace[r.Old]; dup && prev != r.New {
			base.Fatalf("go: conflicting replacements for %v:\n\t%v\n\t%v", r.Old, prev, r.New)
		}
		i.replace[r.Old] = r.New
	}

	i.highestReplaced = make(map[string]string)
	for _, r := range modFile.Replace {
		v, ok := i.highestReplaced[r.Old.Path]
		if !ok || semver.Compare(r.Old.Version, v) > 0 {
			i.highestReplaced[r.Old.Path] = r.Old.Version
		}
	}

	i.exclude = make(map[module.Version]bool, len(modFile.Exclude))
	for _, x := range modFile.Exclude {
		i.exclude[x.Mod] = true
	}

	return i
}

// modFileIsDirty reports whether the go.mod file differs meaningfully
// from what was indexed.
// If modFile has been changed (even cosmetically) since it was first read,
// modFile.Cleanup must be called before modFileIsDirty.
func (i *modFileIndex) modFileIsDirty(modFile *modfile.File) bool {
	if i == nil {
		return modFile != nil
	}

	if i.dataNeedsFix {
		return true
	}

	if modFile.Module == nil {
		if i.module != (module.Version{}) {
			return true
		}
	} else if modFile.Module.Mod != i.module {
		return true
	}

	if modFile.Go == nil {
		if i.goVersionV != "" {
			return true
		}
	} else if "v"+modFile.Go.Version != i.goVersionV {
		if i.goVersionV == "" && cfg.BuildMod != "mod" {
			// go.mod files did not always require a 'go' version, so do not error out
			// if one is missing — we may be inside an older module in the module
			// cache, and should bias toward providing useful behavior.
		} else {
			return true
		}
	}

	if len(modFile.Require) != len(i.require) ||
		len(modFile.Replace) != len(i.replace) ||
		len(modFile.Exclude) != len(i.exclude) {
		return true
	}

	for _, r := range modFile.Require {
		if meta, ok := i.require[r.Mod]; !ok {
			return true
		} else if r.Indirect != meta.indirect {
			if cfg.BuildMod == "readonly" {
				// The module's requirements are consistent; only the "// indirect"
				// comments that are wrong. But those are only guaranteed to be accurate
				// after a "go mod tidy" — it's a good idea to run those before
				// committing a change, but it's certainly not mandatory.
			} else {
				return true
			}
		}
	}

	for _, r := range modFile.Replace {
		if r.New != i.replace[r.Old] {
			return true
		}
	}

	for _, x := range modFile.Exclude {
		if !i.exclude[x.Mod] {
			return true
		}
	}

	return false
}

// rawGoVersion records the Go version parsed from each module's go.mod file.
//
// If a module is replaced, the version of the replacement is keyed by the
// replacement module.Version, not the version being replaced.
var rawGoVersion sync.Map // map[module.Version]string

// A modFileSummary is a summary of a go.mod file for which we do not need to
// retain complete information — for example, the go.mod file of a dependency
// module.
type modFileSummary struct {
	module     module.Version
	goVersion  string
	depth      modDepth
	require    []module.Version
	retract    []retraction
	deprecated string
}

// A retraction consists of a retracted version interval and rationale.
// retraction is like modfile.Retract, but it doesn't point to the syntax tree.
type retraction struct {
	modfile.VersionInterval
	Rationale string
}

// goModSummary returns a summary of the go.mod file for module m,
// taking into account any replacements for m, exclusions of its dependencies,
// and/or vendoring.
//
// m must be a version in the module graph, reachable from the Target module.
// In readonly mode, the go.sum file must contain an entry for m's go.mod file
// (or its replacement). goModSummary must not be called for the Target module
// itself, as its requirements may change. Use rawGoModSummary for other
// module versions.
//
// The caller must not modify the returned summary.
func goModSummary(m module.Version) (*modFileSummary, error) {
	if m == Target {
		panic("internal error: goModSummary called on the Target module")
	}

	if cfg.BuildMod == "vendor" {
		summary := &modFileSummary{
			module: module.Version{Path: m.Path},
		}
		if vendorVersion[m.Path] != m.Version {
			// This module is not vendored, so packages cannot be loaded from it and
			// it cannot be relevant to the build.
			return summary, nil
		}

		// For every module other than the target,
		// return the full list of modules from modules.txt.
		readVendorList()

		// We don't know what versions the vendored module actually relies on,
		// so assume that it requires everything.
		summary.require = vendorList
		return summary, nil
	}

	actual := resolveReplacement(m)
	if HasModRoot() && cfg.BuildMod == "readonly" && actual.Version != "" {
		key := module.Version{Path: actual.Path, Version: actual.Version + "/go.mod"}
		if !modfetch.HaveSum(key) {
			suggestion := fmt.Sprintf("; to add it:\n\tgo mod download %s", m.Path)
			return nil, module.VersionError(actual, &sumMissingError{suggestion: suggestion})
		}
	}
	summary, err := rawGoModSummary(actual)
	if err != nil {
		return nil, err
	}

	if actual.Version == "" {
		// The actual module is a filesystem-local replacement, for which we have
		// unfortunately not enforced any sort of invariants about module lines or
		// matching module paths. Anything goes.
		//
		// TODO(bcmills): Remove this special-case, update tests, and add a
		// release note.
	} else {
		if summary.module.Path == "" {
			return nil, module.VersionError(actual, errors.New("parsing go.mod: missing module line"))
		}

		// In theory we should only allow mpath to be unequal to m.Path here if the
		// version that we fetched lacks an explicit go.mod file: if the go.mod file
		// is explicit, then it should match exactly (to ensure that imports of other
		// packages within the module are interpreted correctly). Unfortunately, we
		// can't determine that information from the module proxy protocol: we'll have
		// to leave that validation for when we load actual packages from within the
		// module.
		if mpath := summary.module.Path; mpath != m.Path && mpath != actual.Path {
			return nil, module.VersionError(actual, fmt.Errorf(`parsing go.mod:
	module declares its path as: %s
	        but was required as: %s`, mpath, m.Path))
		}
	}

	if index != nil && len(index.exclude) > 0 {
		// Drop any requirements on excluded versions.
		// Don't modify the cached summary though, since we might need the raw
		// summary separately.
		haveExcludedReqs := false
		for _, r := range summary.require {
			if index.exclude[r] {
				haveExcludedReqs = true
				break
			}
		}
		if haveExcludedReqs {
			s := new(modFileSummary)
			*s = *summary
			s.require = make([]module.Version, 0, len(summary.require))
			for _, r := range summary.require {
				if !index.exclude[r] {
					s.require = append(s.require, r)
				}
			}
			summary = s
		}
	}
	return summary, nil
}

// rawGoModSummary returns a new summary of the go.mod file for module m,
// ignoring all replacements that may apply to m and excludes that may apply to
// its dependencies.
//
// rawGoModSummary cannot be used on the Target module.
func rawGoModSummary(m module.Version) (*modFileSummary, error) {
	if m == Target {
		panic("internal error: rawGoModSummary called on the Target module")
	}

	type cached struct {
		summary *modFileSummary
		err     error
	}
	c := rawGoModSummaryCache.Do(m, func() interface{} {
		summary := new(modFileSummary)
		name, data, err := rawGoModData(m)
		if err != nil {
			return cached{nil, err}
		}
		f, err := modfile.ParseLax(name, data, nil)
		if err != nil {
			return cached{nil, module.VersionError(m, fmt.Errorf("parsing %s: %v", base.ShortPath(name), err))}
		}
		if f.Module != nil {
			summary.module = f.Module.Mod
			summary.deprecated = f.Module.Deprecated
		}
		if f.Go != nil && f.Go.Version != "" {
			rawGoVersion.LoadOrStore(m, f.Go.Version)
			summary.goVersion = f.Go.Version
			summary.depth = modDepthFromGoVersion(f.Go.Version)
		} else {
			summary.depth = eager
		}
		if len(f.Require) > 0 {
			summary.require = make([]module.Version, 0, len(f.Require))
			for _, req := range f.Require {
				summary.require = append(summary.require, req.Mod)
			}
		}
		if len(f.Retract) > 0 {
			summary.retract = make([]retraction, 0, len(f.Retract))
			for _, ret := range f.Retract {
				summary.retract = append(summary.retract, retraction{
					VersionInterval: ret.VersionInterval,
					Rationale:       ret.Rationale,
				})
			}
		}

		return cached{summary, nil}
	}).(cached)

	return c.summary, c.err
}

var rawGoModSummaryCache par.Cache // module.Version → rawGoModSummary result

// rawGoModData returns the content of the go.mod file for module m, ignoring
// all replacements that may apply to m.
//
// rawGoModData cannot be used on the Target module.
//
// Unlike rawGoModSummary, rawGoModData does not cache its results in memory.
// Use rawGoModSummary instead unless you specifically need these bytes.
func rawGoModData(m module.Version) (name string, data []byte, err error) {
	if m.Version == "" {
		// m is a replacement module with only a file path.
		dir := m.Path
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(ModRoot(), dir)
		}
		name = filepath.Join(dir, "go.mod")
		if gomodActual, ok := fsys.OverlayPath(name); ok {
			// Don't lock go.mod if it's part of the overlay.
			// On Plan 9, locking requires chmod, and we don't want to modify any file
			// in the overlay. See #44700.
			data, err = os.ReadFile(gomodActual)
		} else {
			data, err = lockedfile.Read(gomodActual)
		}
		if err != nil {
			return "", nil, module.VersionError(m, fmt.Errorf("reading %s: %v", base.ShortPath(name), err))
		}
	} else {
		if !semver.IsValid(m.Version) {
			// Disallow the broader queries supported by fetch.Lookup.
			base.Fatalf("go: internal error: %s@%s: unexpected invalid semantic version", m.Path, m.Version)
		}
		name = "go.mod"
		data, err = modfetch.GoMod(m.Path, m.Version)
	}
	return name, data, err
}

// queryLatestVersionIgnoringRetractions looks up the latest version of the
// module with the given path without considering retracted or excluded
// versions.
//
// If all versions of the module are replaced,
// queryLatestVersionIgnoringRetractions returns the replacement without making
// a query.
//
// If the queried latest version is replaced,
// queryLatestVersionIgnoringRetractions returns the replacement.
func queryLatestVersionIgnoringRetractions(ctx context.Context, path string) (latest module.Version, err error) {
	type entry struct {
		latest module.Version
		err    error
	}
	e := latestVersionIgnoringRetractionsCache.Do(path, func() interface{} {
		ctx, span := trace.StartSpan(ctx, "queryLatestVersionIgnoringRetractions "+path)
		defer span.Done()

		if repl := Replacement(module.Version{Path: path}); repl.Path != "" {
			// All versions of the module were replaced.
			// No need to query.
			return &entry{latest: repl}
		}

		// Find the latest version of the module.
		// Ignore exclusions from the main module's go.mod.
		const ignoreSelected = ""
		var allowAll AllowedFunc
		rev, err := Query(ctx, path, "latest", ignoreSelected, allowAll)
		if err != nil {
			return &entry{err: err}
		}
		latest := module.Version{Path: path, Version: rev.Version}
		if repl := resolveReplacement(latest); repl.Path != "" {
			latest = repl
		}
		return &entry{latest: latest}
	}).(*entry)
	return e.latest, e.err
}

var latestVersionIgnoringRetractionsCache par.Cache // path → queryLatestVersionIgnoringRetractions result
