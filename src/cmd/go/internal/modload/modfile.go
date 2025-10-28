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
	"cmd/go/internal/gover"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/trace"
	"cmd/internal/par"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
)

// ReadModFile reads and parses the mod file at gomod. ReadModFile properly applies the
// overlay, locks the file while reading, and applies fix, if applicable.
func ReadModFile(gomod string, fix modfile.VersionFixer) (data []byte, f *modfile.File, err error) {
	if fsys.Replaced(gomod) {
		// Don't lock go.mod if it's part of the overlay.
		// On Plan 9, locking requires chmod, and we don't want to modify any file
		// in the overlay. See #44700.
		data, err = os.ReadFile(fsys.Actual(gomod))
	} else {
		data, err = lockedfile.Read(gomod)
	}
	if err != nil {
		return nil, nil, err
	}

	f, err = modfile.Parse(gomod, data, fix)
	if err != nil {
		f, laxErr := modfile.ParseLax(gomod, data, fix)
		if laxErr == nil {
			if f.Go != nil && gover.Compare(f.Go.Version, gover.Local()) > 0 {
				toolchain := ""
				if f.Toolchain != nil {
					toolchain = f.Toolchain.Name
				}
				return nil, nil, &gover.TooNewError{What: base.ShortPath(gomod), GoVersion: f.Go.Version, Toolchain: toolchain}
			}
		}

		// Errors returned by modfile.Parse begin with file:line.
		return nil, nil, fmt.Errorf("errors parsing %s:\n%w", base.ShortPath(gomod), shortPathErrorList(err))
	}
	if f.Go != nil && gover.Compare(f.Go.Version, gover.Local()) > 0 {
		toolchain := ""
		if f.Toolchain != nil {
			toolchain = f.Toolchain.Name
		}
		return nil, nil, &gover.TooNewError{What: base.ShortPath(gomod), GoVersion: f.Go.Version, Toolchain: toolchain}
	}
	if f.Module == nil {
		// No module declaration. Must add module path.
		return nil, nil, fmt.Errorf("error reading %s: missing module declaration. To specify the module path:\n\tgo mod edit -module=example.com/mod", base.ShortPath(gomod))
	} else if err := CheckReservedModulePath(f.Module.Mod.Path); err != nil {
		return nil, nil, fmt.Errorf("error reading %s: invalid module path: %q", base.ShortPath(gomod), f.Module.Mod.Path)
	}

	return data, f, err
}

func shortPathErrorList(err error) error {
	if el, ok := errors.AsType[modfile.ErrorList](err); ok {
		for i := range el {
			el[i].Filename = base.ShortPath(el[i].Filename)
		}
	}
	return err
}

// A modFileIndex is an index of data corresponding to a modFile
// at a specific point in time.
type modFileIndex struct {
	data         []byte
	dataNeedsFix bool // true if fixVersion applied a change while parsing data
	module       module.Version
	goVersion    string // Go version (no "v" or "go" prefix)
	toolchain    string
	require      map[module.Version]requireMeta
	replace      map[module.Version]module.Version
	exclude      map[module.Version]bool
	ignore       []string
}

type requireMeta struct {
	indirect bool
}

// A modPruning indicates whether transitive dependencies of Go 1.17 dependencies
// are pruned out of the module subgraph rooted at a given module.
// (See https://golang.org/ref/mod#graph-pruning.)
type modPruning uint8

const (
	pruned    modPruning = iota // transitive dependencies of modules at go 1.17 and higher are pruned out
	unpruned                    // no transitive dependencies are pruned out
	workspace                   // pruned to the union of modules in the workspace
)

func (p modPruning) String() string {
	switch p {
	case pruned:
		return "pruned"
	case unpruned:
		return "unpruned"
	case workspace:
		return "workspace"
	default:
		return fmt.Sprintf("%T(%d)", p, p)
	}
}

func pruningForGoVersion(goVersion string) modPruning {
	if gover.Compare(goVersion, gover.ExplicitIndirectVersion) < 0 {
		// The go.mod file does not duplicate relevant information about transitive
		// dependencies, so they cannot be pruned out.
		return unpruned
	}
	return pruned
}

// CheckAllowed returns an error equivalent to ErrDisallowed if m is excluded by
// the main module's go.mod or retracted by its author. Most version queries use
// this to filter out versions that should not be used.
func (s *State) CheckAllowed(ctx context.Context, m module.Version) error {
	if err := s.CheckExclusions(ctx, m); err != nil {
		return err
	}
	if err := s.CheckRetractions(ctx, m); err != nil {
		return err
	}
	return nil
}

// ErrDisallowed is returned by version predicates passed to Query and similar
// functions to indicate that a version should not be considered.
var ErrDisallowed = errors.New("disallowed module version")

// CheckExclusions returns an error equivalent to ErrDisallowed if module m is
// excluded by the main module's go.mod file.
func (s *State) CheckExclusions(ctx context.Context, m module.Version) error {
	for _, mainModule := range s.MainModules.Versions() {
		if index := s.MainModules.Index(mainModule); index != nil && index.exclude[m] {
			return module.VersionError(m, errExcluded)
		}
	}
	return nil
}

var errExcluded = &excludedError{}

type excludedError struct{}

func (e *excludedError) Error() string     { return "excluded by go.mod" }
func (e *excludedError) Is(err error) bool { return err == ErrDisallowed }

// CheckRetractions returns an error if module m has been retracted by
// its author.
func (s *State) CheckRetractions(ctx context.Context, m module.Version) (err error) {
	defer func() {
		if err == nil {
			return
		}
		if _, ok := errors.AsType[*ModuleRetractedError](err); ok {
			return
		}
		// Attribute the error to the version being checked, not the version from
		// which the retractions were to be loaded.
		if mErr, ok := errors.AsType[*module.ModuleError](err); ok {
			err = mErr.Err
		}
		err = &retractionLoadingError{m: m, err: err}
	}()

	if m.Version == "" {
		// Main module, standard library, or file replacement module.
		// Cannot be retracted.
		return nil
	}
	if repl := Replacement(s, module.Version{Path: m.Path}); repl.Path != "" {
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
	rm, err := queryLatestVersionIgnoringRetractions(s, ctx, m.Path)
	if err != nil {
		return err
	}
	summary, err := rawGoModSummary(s, rm)
	if err != nil && !errors.Is(err, gover.ErrTooNew) {
		return err
	}

	var rationale []string
	isRetracted := false
	for _, r := range summary.retract {
		if gover.ModCompare(m.Path, r.Low, m.Version) <= 0 && gover.ModCompare(m.Path, m.Version, r.High) <= 0 {
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
func CheckDeprecation(loaderstate *State, ctx context.Context, m module.Version) (deprecation string, err error) {
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
	if repl := Replacement(loaderstate, module.Version{Path: m.Path}); repl.Path != "" {
		// All versions of the module were replaced.
		// We'll look up deprecation separately for the replacement.
		return "", nil
	}

	latest, err := queryLatestVersionIgnoringRetractions(loaderstate, ctx, m.Path)
	if err != nil {
		return "", err
	}
	summary, err := rawGoModSummary(loaderstate, latest)
	if err != nil && !errors.Is(err, gover.ErrTooNew) {
		return "", err
	}
	return summary.deprecated, nil
}

func replacement(mod module.Version, replace map[module.Version]module.Version) (fromVersion string, to module.Version, ok bool) {
	if r, ok := replace[mod]; ok {
		return mod.Version, r, true
	}
	if r, ok := replace[module.Version{Path: mod.Path}]; ok {
		return "", r, true
	}
	return "", module.Version{}, false
}

// Replacement returns the replacement for mod, if any. If the path in the
// module.Version is relative it's relative to the single main module outside
// workspace mode, or the workspace's directory in workspace mode.
func Replacement(loaderstate *State, mod module.Version) module.Version {
	r, foundModRoot, _ := replacementFrom(loaderstate, mod)
	return canonicalizeReplacePath(loaderstate, r, foundModRoot)
}

// replacementFrom returns the replacement for mod, if any, the modroot of the replacement if it appeared in a go.mod,
// and the source of the replacement. The replacement is relative to the go.work or go.mod file it appears in.
func replacementFrom(loaderstate *State, mod module.Version) (r module.Version, modroot string, fromFile string) {
	foundFrom, found, foundModRoot := "", module.Version{}, ""
	if loaderstate.MainModules == nil {
		return module.Version{}, "", ""
	} else if loaderstate.MainModules.Contains(mod.Path) && mod.Version == "" {
		// Don't replace the workspace version of the main module.
		return module.Version{}, "", ""
	}
	if _, r, ok := replacement(mod, loaderstate.MainModules.WorkFileReplaceMap()); ok {
		return r, "", loaderstate.workFilePath
	}
	for _, v := range loaderstate.MainModules.Versions() {
		if index := loaderstate.MainModules.Index(v); index != nil {
			if from, r, ok := replacement(mod, index.replace); ok {
				modRoot := loaderstate.MainModules.ModRoot(v)
				if foundModRoot != "" && foundFrom != from && found != r {
					base.Errorf("conflicting replacements found for %v in workspace modules defined by %v and %v",
						mod, modFilePath(foundModRoot), modFilePath(modRoot))
					return found, foundModRoot, modFilePath(foundModRoot)
				}
				found, foundModRoot = r, modRoot
			}
		}
	}
	return found, foundModRoot, modFilePath(foundModRoot)
}

func replaceRelativeTo(loaderstate *State) string {
	if workFilePath := WorkFilePath(loaderstate); workFilePath != "" {
		return filepath.Dir(workFilePath)
	}
	return loaderstate.MainModules.ModRoot(loaderstate.MainModules.mustGetSingleMainModule(loaderstate))
}

// canonicalizeReplacePath ensures that relative, on-disk, replaced module paths
// are relative to the workspace directory (in workspace mode) or to the module's
// directory (in module mode, as they already are).
func canonicalizeReplacePath(loaderstate *State, r module.Version, modRoot string) module.Version {
	if filepath.IsAbs(r.Path) || r.Version != "" || modRoot == "" {
		return r
	}
	workFilePath := WorkFilePath(loaderstate)
	if workFilePath == "" {
		return r
	}
	abs := filepath.Join(modRoot, r.Path)
	if rel, err := filepath.Rel(filepath.Dir(workFilePath), abs); err == nil {
		return module.Version{Path: ToDirectoryPath(rel), Version: r.Version}
	}
	// We couldn't make the version's path relative to the workspace's path,
	// so just return the absolute path. It's the best we can do.
	return module.Version{Path: ToDirectoryPath(abs), Version: r.Version}
}

// resolveReplacement returns the module actually used to load the source code
// for m: either m itself, or the replacement for m (iff m is replaced).
// It also returns the modroot of the module providing the replacement if
// one was found.
func resolveReplacement(loaderstate *State, m module.Version) module.Version {
	if r := Replacement(loaderstate, m); r.Path != "" {
		return r
	}
	return m
}

func toReplaceMap(replacements []*modfile.Replace) map[module.Version]module.Version {
	replaceMap := make(map[module.Version]module.Version, len(replacements))
	for _, r := range replacements {
		if prev, dup := replaceMap[r.Old]; dup && prev != r.New {
			base.Fatalf("go: conflicting replacements for %v:\n\t%v\n\t%v", r.Old, prev, r.New)
		}
		replaceMap[r.Old] = r.New
	}
	return replaceMap
}

// indexModFile rebuilds the index of modFile.
// If modFile has been changed since it was first read,
// modFile.Cleanup must be called before indexModFile.
func indexModFile(data []byte, modFile *modfile.File, mod module.Version, needsFix bool) *modFileIndex {
	i := new(modFileIndex)
	i.data = data
	i.dataNeedsFix = needsFix

	i.module = module.Version{}
	if modFile.Module != nil {
		i.module = modFile.Module.Mod
	}

	i.goVersion = ""
	if modFile.Go == nil {
		rawGoVersion.Store(mod, "")
	} else {
		i.goVersion = modFile.Go.Version
		rawGoVersion.Store(mod, modFile.Go.Version)
	}
	if modFile.Toolchain != nil {
		i.toolchain = modFile.Toolchain.Name
	}

	i.require = make(map[module.Version]requireMeta, len(modFile.Require))
	for _, r := range modFile.Require {
		i.require[r.Mod] = requireMeta{indirect: r.Indirect}
	}

	i.replace = toReplaceMap(modFile.Replace)

	i.exclude = make(map[module.Version]bool, len(modFile.Exclude))
	for _, x := range modFile.Exclude {
		i.exclude[x.Mod] = true
	}
	if modFile.Ignore != nil {
		for _, x := range modFile.Ignore {
			i.ignore = append(i.ignore, x.Path)
		}
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

	var goV, toolchain string
	if modFile.Go != nil {
		goV = modFile.Go.Version
	}
	if modFile.Toolchain != nil {
		toolchain = modFile.Toolchain.Name
	}

	if goV != i.goVersion ||
		toolchain != i.toolchain ||
		len(modFile.Require) != len(i.require) ||
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
	toolchain  string
	ignore     []string
	pruning    modPruning
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
func goModSummary(loaderstate *State, m module.Version) (*modFileSummary, error) {
	if m.Version == "" && !inWorkspaceMode(loaderstate) && loaderstate.MainModules.Contains(m.Path) {
		panic("internal error: goModSummary called on a main module")
	}
	if gover.IsToolchain(m.Path) {
		return rawGoModSummary(loaderstate, m)
	}

	if cfg.BuildMod == "vendor" {
		summary := &modFileSummary{
			module: module.Version{Path: m.Path},
		}

		readVendorList(VendorDir(loaderstate))
		if vendorVersion[m.Path] != m.Version {
			// This module is not vendored, so packages cannot be loaded from it and
			// it cannot be relevant to the build.
			return summary, nil
		}

		// For every module other than the target,
		// return the full list of modules from modules.txt.
		// We don't know what versions the vendored module actually relies on,
		// so assume that it requires everything.
		summary.require = vendorList
		return summary, nil
	}

	actual := resolveReplacement(loaderstate, m)
	if mustHaveSums(loaderstate) && actual.Version != "" {
		key := module.Version{Path: actual.Path, Version: actual.Version + "/go.mod"}
		if !modfetch.HaveSum(key) {
			suggestion := fmt.Sprintf(" for go.mod file; to add it:\n\tgo mod download %s", m.Path)
			return nil, module.VersionError(actual, &sumMissingError{suggestion: suggestion})
		}
	}
	summary, err := rawGoModSummary(loaderstate, actual)
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
			return nil, module.VersionError(actual,
				fmt.Errorf("parsing go.mod:\n"+
					"\tmodule declares its path as: %s\n"+
					"\t        but was required as: %s", mpath, m.Path))
		}
	}

	for _, mainModule := range loaderstate.MainModules.Versions() {
		if index := loaderstate.MainModules.Index(mainModule); index != nil && len(index.exclude) > 0 {
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
	}
	return summary, nil
}

// rawGoModSummary returns a new summary of the go.mod file for module m,
// ignoring all replacements that may apply to m and excludes that may apply to
// its dependencies.
//
// rawGoModSummary cannot be used on the main module outside of workspace mode.
// The modFileSummary can still be used for retractions and deprecations
// even if a TooNewError is returned.
func rawGoModSummary(loaderstate *State, m module.Version) (*modFileSummary, error) {
	if gover.IsToolchain(m.Path) {
		if m.Path == "go" && gover.Compare(m.Version, gover.GoStrictVersion) >= 0 {
			// Declare that go 1.21.3 requires toolchain 1.21.3,
			// so that go get knows that downgrading toolchain implies downgrading go
			// and similarly upgrading go requires upgrading the toolchain.
			return &modFileSummary{module: m, require: []module.Version{{Path: "toolchain", Version: "go" + m.Version}}}, nil
		}
		return &modFileSummary{module: m}, nil
	}
	if m.Version == "" && !inWorkspaceMode(loaderstate) && loaderstate.MainModules.Contains(m.Path) {
		// Calling rawGoModSummary implies that we are treating m as a module whose
		// requirements aren't the roots of the module graph and can't be modified.
		//
		// If we are not in workspace mode, then the requirements of the main module
		// are the roots of the module graph and we expect them to be kept consistent.
		panic("internal error: rawGoModSummary called on a main module")
	}
	if m.Version == "" && inWorkspaceMode(loaderstate) && m.Path == "command-line-arguments" {
		// "go work sync" calls LoadModGraph to make sure the module graph is valid.
		// If there are no modules in the workspace, we synthesize an empty
		// command-line-arguments module, which rawGoModData cannot read a go.mod for.
		return &modFileSummary{module: m}, nil
	} else if m.Version == "" && inWorkspaceMode(loaderstate) && loaderstate.MainModules.Contains(m.Path) {
		// When go get uses EnterWorkspace to check that the workspace loads properly,
		// it will update the contents of the workspace module's modfile in memory. To use the updated
		// contents of the modfile when doing the load, don't read from disk and instead
		// recompute a summary using the updated contents of the modfile.
		if mf := loaderstate.MainModules.ModFile(m); mf != nil {
			return summaryFromModFile(m, loaderstate.MainModules.modFiles[m])
		}
	}
	return rawGoModSummaryCache.Do(m, func() (*modFileSummary, error) {
		name, data, err := rawGoModData(loaderstate, m)
		if err != nil {
			return nil, err
		}
		f, err := modfile.ParseLax(name, data, nil)
		if err != nil {
			return nil, module.VersionError(m, fmt.Errorf("parsing %s: %v", base.ShortPath(name), err))
		}
		return summaryFromModFile(m, f)
	})
}

func summaryFromModFile(m module.Version, f *modfile.File) (*modFileSummary, error) {
	summary := new(modFileSummary)
	if f.Module != nil {
		summary.module = f.Module.Mod
		summary.deprecated = f.Module.Deprecated
	}
	if f.Go != nil {
		rawGoVersion.LoadOrStore(m, f.Go.Version)
		summary.goVersion = f.Go.Version
		summary.pruning = pruningForGoVersion(f.Go.Version)
	} else {
		summary.pruning = unpruned
	}
	if f.Toolchain != nil {
		summary.toolchain = f.Toolchain.Name
	}
	if f.Ignore != nil {
		for _, i := range f.Ignore {
			summary.ignore = append(summary.ignore, i.Path)
		}
	}
	if len(f.Require) > 0 {
		summary.require = make([]module.Version, 0, len(f.Require)+1)
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

	// This block must be kept at the end of the function because the summary may
	// be used for reading retractions or deprecations even if a TooNewError is
	// returned.
	if summary.goVersion != "" && gover.Compare(summary.goVersion, gover.GoStrictVersion) >= 0 {
		summary.require = append(summary.require, module.Version{Path: "go", Version: summary.goVersion})
		if gover.Compare(summary.goVersion, gover.Local()) > 0 {
			return summary, &gover.TooNewError{What: "module " + m.String(), GoVersion: summary.goVersion}
		}
	}

	return summary, nil
}

var rawGoModSummaryCache par.ErrCache[module.Version, *modFileSummary]

// rawGoModData returns the content of the go.mod file for module m, ignoring
// all replacements that may apply to m.
//
// rawGoModData cannot be used on the main module outside of workspace mode.
//
// Unlike rawGoModSummary, rawGoModData does not cache its results in memory.
// Use rawGoModSummary instead unless you specifically need these bytes.
func rawGoModData(loaderstate *State, m module.Version) (name string, data []byte, err error) {
	if m.Version == "" {
		dir := m.Path
		if !filepath.IsAbs(dir) {
			if inWorkspaceMode(loaderstate) && loaderstate.MainModules.Contains(m.Path) {
				dir = loaderstate.MainModules.ModRoot(m)
			} else {
				// m is a replacement module with only a file path.
				dir = filepath.Join(replaceRelativeTo(loaderstate), dir)
			}
		}
		name = filepath.Join(dir, "go.mod")
		if fsys.Replaced(name) {
			// Don't lock go.mod if it's part of the overlay.
			// On Plan 9, locking requires chmod, and we don't want to modify any file
			// in the overlay. See #44700.
			data, err = os.ReadFile(fsys.Actual(name))
		} else {
			data, err = lockedfile.Read(name)
		}
		if err != nil {
			return "", nil, module.VersionError(m, fmt.Errorf("reading %s: %v", base.ShortPath(name), err))
		}
	} else {
		if !gover.ModIsValid(m.Path, m.Version) {
			// Disallow the broader queries supported by fetch.Lookup.
			base.Fatalf("go: internal error: %s@%s: unexpected invalid semantic version", m.Path, m.Version)
		}
		name = "go.mod"
		data, err = modfetch.GoMod(context.TODO(), m.Path, m.Version)
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
func queryLatestVersionIgnoringRetractions(loaderstate *State, ctx context.Context, path string) (latest module.Version, err error) {
	return latestVersionIgnoringRetractionsCache.Do(path, func() (module.Version, error) {
		ctx, span := trace.StartSpan(ctx, "queryLatestVersionIgnoringRetractions "+path)
		defer span.Done()

		if repl := Replacement(loaderstate, module.Version{Path: path}); repl.Path != "" {
			// All versions of the module were replaced.
			// No need to query.
			return repl, nil
		}

		// Find the latest version of the module.
		// Ignore exclusions from the main module's go.mod.
		const ignoreSelected = ""
		var allowAll AllowedFunc
		rev, err := Query(loaderstate, ctx, path, "latest", ignoreSelected, allowAll)
		if err != nil {
			return module.Version{}, err
		}
		latest := module.Version{Path: path, Version: rev.Version}
		if repl := resolveReplacement(loaderstate, latest); repl.Path != "" {
			latest = repl
		}
		return latest, nil
	})
}

var latestVersionIgnoringRetractionsCache par.ErrCache[string, module.Version] // path → queryLatestVersionIgnoringRetractions result

// ToDirectoryPath adds a prefix if necessary so that path in unambiguously
// an absolute path or a relative path starting with a '.' or '..'
// path component.
func ToDirectoryPath(path string) string {
	if modfile.IsDirectoryPath(path) {
		return path
	}
	// The path is not a relative path or an absolute path, so make it relative
	// to the current directory.
	return "./" + filepath.ToSlash(filepath.Clean(path))
}
