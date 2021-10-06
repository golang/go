// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	"cmd/go/internal/modfetch/codehost"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
	modzip "golang.org/x/mod/zip"
)

// A codeRepo implements modfetch.Repo using an underlying codehost.Repo.
type codeRepo struct {
	modPath string

	// code is the repository containing this module.
	code codehost.Repo
	// codeRoot is the import path at the root of code.
	codeRoot string
	// codeDir is the directory (relative to root) at which we expect to find the module.
	// If pathMajor is non-empty and codeRoot is not the full modPath,
	// then we look in both codeDir and codeDir/pathMajor[1:].
	codeDir string

	// pathMajor is the suffix of modPath that indicates its major version,
	// or the empty string if modPath is at major version 0 or 1.
	//
	// pathMajor is typically of the form "/vN", but possibly ".vN", or
	// ".vN-unstable" for modules resolved using gopkg.in.
	pathMajor string
	// pathPrefix is the prefix of modPath that excludes pathMajor.
	// It is used only for logging.
	pathPrefix string

	// pseudoMajor is the major version prefix to require when generating
	// pseudo-versions for this module, derived from the module path. pseudoMajor
	// is empty if the module path does not include a version suffix (that is,
	// accepts either v0 or v1).
	pseudoMajor string
}

// newCodeRepo returns a Repo that reads the source code for the module with the
// given path, from the repo stored in code, with the root of the repo
// containing the path given by codeRoot.
func newCodeRepo(code codehost.Repo, codeRoot, path string) (Repo, error) {
	if !hasPathPrefix(path, codeRoot) {
		return nil, fmt.Errorf("mismatched repo: found %s for %s", codeRoot, path)
	}
	pathPrefix, pathMajor, ok := module.SplitPathVersion(path)
	if !ok {
		return nil, fmt.Errorf("invalid module path %q", path)
	}
	if codeRoot == path {
		pathPrefix = path
	}
	pseudoMajor := module.PathMajorPrefix(pathMajor)

	// Compute codeDir = bar, the subdirectory within the repo
	// corresponding to the module root.
	//
	// At this point we might have:
	//	path = github.com/rsc/foo/bar/v2
	//	codeRoot = github.com/rsc/foo
	//	pathPrefix = github.com/rsc/foo/bar
	//	pathMajor = /v2
	//	pseudoMajor = v2
	//
	// which gives
	//	codeDir = bar
	//
	// We know that pathPrefix is a prefix of path, and codeRoot is a prefix of
	// path, but codeRoot may or may not be a prefix of pathPrefix, because
	// codeRoot may be the entire path (in which case codeDir should be empty).
	// That occurs in two situations.
	//
	// One is when a go-import meta tag resolves the complete module path,
	// including the pathMajor suffix:
	//	path = nanomsg.org/go/mangos/v2
	//	codeRoot = nanomsg.org/go/mangos/v2
	//	pathPrefix = nanomsg.org/go/mangos
	//	pathMajor = /v2
	//	pseudoMajor = v2
	//
	// The other is similar: for gopkg.in only, the major version is encoded
	// with a dot rather than a slash, and thus can't be in a subdirectory.
	//	path = gopkg.in/yaml.v2
	//	codeRoot = gopkg.in/yaml.v2
	//	pathPrefix = gopkg.in/yaml
	//	pathMajor = .v2
	//	pseudoMajor = v2
	//
	codeDir := ""
	if codeRoot != path {
		if !hasPathPrefix(pathPrefix, codeRoot) {
			return nil, fmt.Errorf("repository rooted at %s cannot contain module %s", codeRoot, path)
		}
		codeDir = strings.Trim(pathPrefix[len(codeRoot):], "/")
	}

	r := &codeRepo{
		modPath:     path,
		code:        code,
		codeRoot:    codeRoot,
		codeDir:     codeDir,
		pathPrefix:  pathPrefix,
		pathMajor:   pathMajor,
		pseudoMajor: pseudoMajor,
	}

	return r, nil
}

func (r *codeRepo) ModulePath() string {
	return r.modPath
}

func (r *codeRepo) Versions(prefix string) ([]string, error) {
	// Special case: gopkg.in/macaroon-bakery.v2-unstable
	// does not use the v2 tags (those are for macaroon-bakery.v2).
	// It has no possible tags at all.
	if strings.HasPrefix(r.modPath, "gopkg.in/") && strings.HasSuffix(r.modPath, "-unstable") {
		return nil, nil
	}

	p := prefix
	if r.codeDir != "" {
		p = r.codeDir + "/" + p
	}
	tags, err := r.code.Tags(p)
	if err != nil {
		return nil, &module.ModuleError{
			Path: r.modPath,
			Err:  err,
		}
	}

	var list, incompatible []string
	for _, tag := range tags {
		if !strings.HasPrefix(tag, p) {
			continue
		}
		v := tag
		if r.codeDir != "" {
			v = v[len(r.codeDir)+1:]
		}
		if v == "" || v != module.CanonicalVersion(v) || module.IsPseudoVersion(v) {
			continue
		}

		if err := module.CheckPathMajor(v, r.pathMajor); err != nil {
			if r.codeDir == "" && r.pathMajor == "" && semver.Major(v) > "v1" {
				incompatible = append(incompatible, v)
			}
			continue
		}

		list = append(list, v)
	}
	semver.Sort(list)
	semver.Sort(incompatible)

	return r.appendIncompatibleVersions(list, incompatible)
}

// appendIncompatibleVersions appends "+incompatible" versions to list if
// appropriate, returning the final list.
//
// The incompatible list contains candidate versions without the '+incompatible'
// prefix.
//
// Both list and incompatible must be sorted in semantic order.
func (r *codeRepo) appendIncompatibleVersions(list, incompatible []string) ([]string, error) {
	if len(incompatible) == 0 || r.pathMajor != "" {
		// No +incompatible versions are possible, so no need to check them.
		return list, nil
	}

	versionHasGoMod := func(v string) (bool, error) {
		_, err := r.code.ReadFile(v, "go.mod", codehost.MaxGoMod)
		if err == nil {
			return true, nil
		}
		if !os.IsNotExist(err) {
			return false, &module.ModuleError{
				Path: r.modPath,
				Err:  err,
			}
		}
		return false, nil
	}

	if len(list) > 0 {
		ok, err := versionHasGoMod(list[len(list)-1])
		if err != nil {
			return nil, err
		}
		if ok {
			// The latest compatible version has a go.mod file, so assume that all
			// subsequent versions do as well, and do not include any +incompatible
			// versions. Even if we are wrong, the author clearly intends module
			// consumers to be on the v0/v1 line instead of a higher +incompatible
			// version. (See https://golang.org/issue/34189.)
			//
			// We know of at least two examples where this behavior is desired
			// (github.com/russross/blackfriday@v2.0.0 and
			// github.com/libp2p/go-libp2p@v6.0.23), and (as of 2019-10-29) have no
			// concrete examples for which it is undesired.
			return list, nil
		}
	}

	var (
		lastMajor         string
		lastMajorHasGoMod bool
	)
	for i, v := range incompatible {
		major := semver.Major(v)

		if major != lastMajor {
			rem := incompatible[i:]
			j := sort.Search(len(rem), func(j int) bool {
				return semver.Major(rem[j]) != major
			})
			latestAtMajor := rem[j-1]

			var err error
			lastMajor = major
			lastMajorHasGoMod, err = versionHasGoMod(latestAtMajor)
			if err != nil {
				return nil, err
			}
		}

		if lastMajorHasGoMod {
			// The latest release of this major version has a go.mod file, so it is
			// not allowed as +incompatible. It would be confusing to include some
			// minor versions of this major version as +incompatible but require
			// semantic import versioning for others, so drop all +incompatible
			// versions for this major version.
			//
			// If we're wrong about a minor version in the middle, users will still be
			// able to 'go get' specific tags for that version explicitly — they just
			// won't appear in 'go list' or as the results for queries with inequality
			// bounds.
			continue
		}
		list = append(list, v+"+incompatible")
	}

	return list, nil
}

func (r *codeRepo) Stat(rev string) (*RevInfo, error) {
	if rev == "latest" {
		return r.Latest()
	}
	codeRev := r.revToRev(rev)
	info, err := r.code.Stat(codeRev)
	if err != nil {
		return nil, &module.ModuleError{
			Path: r.modPath,
			Err: &module.InvalidVersionError{
				Version: rev,
				Err:     err,
			},
		}
	}
	return r.convert(info, rev)
}

func (r *codeRepo) Latest() (*RevInfo, error) {
	info, err := r.code.Latest()
	if err != nil {
		return nil, err
	}
	return r.convert(info, "")
}

// convert converts a version as reported by the code host to a version as
// interpreted by the module system.
//
// If statVers is a valid module version, it is used for the Version field.
// Otherwise, the Version is derived from the passed-in info and recent tags.
func (r *codeRepo) convert(info *codehost.RevInfo, statVers string) (*RevInfo, error) {
	info2 := &RevInfo{
		Name:  info.Name,
		Short: info.Short,
		Time:  info.Time,
	}

	// If this is a plain tag (no dir/ prefix)
	// and the module path is unversioned,
	// and if the underlying file tree has no go.mod,
	// then allow using the tag with a +incompatible suffix.
	var canUseIncompatible func() bool
	canUseIncompatible = func() bool {
		var ok bool
		if r.codeDir == "" && r.pathMajor == "" {
			_, errGoMod := r.code.ReadFile(info.Name, "go.mod", codehost.MaxGoMod)
			if errGoMod != nil {
				ok = true
			}
		}
		canUseIncompatible = func() bool { return ok }
		return ok
	}

	invalidf := func(format string, args ...interface{}) error {
		return &module.ModuleError{
			Path: r.modPath,
			Err: &module.InvalidVersionError{
				Version: info2.Version,
				Err:     fmt.Errorf(format, args...),
			},
		}
	}

	// checkGoMod verifies that the go.mod file for the module exists or does not
	// exist as required by info2.Version and the module path represented by r.
	checkGoMod := func() (*RevInfo, error) {
		// If r.codeDir is non-empty, then the go.mod file must exist: the module
		// author — not the module consumer, — gets to decide how to carve up the repo
		// into modules.
		//
		// Conversely, if the go.mod file exists, the module author — not the module
		// consumer — gets to determine the module's path
		//
		// r.findDir verifies both of these conditions. Execute it now so that
		// r.Stat will correctly return a notExistError if the go.mod location or
		// declared module path doesn't match.
		_, _, _, err := r.findDir(info2.Version)
		if err != nil {
			// TODO: It would be nice to return an error like "not a module".
			// Right now we return "missing go.mod", which is a little confusing.
			return nil, &module.ModuleError{
				Path: r.modPath,
				Err: &module.InvalidVersionError{
					Version: info2.Version,
					Err:     notExistError{err: err},
				},
			}
		}

		// If the version is +incompatible, then the go.mod file must not exist:
		// +incompatible is not an ongoing opt-out from semantic import versioning.
		if strings.HasSuffix(info2.Version, "+incompatible") {
			if !canUseIncompatible() {
				if r.pathMajor != "" {
					return nil, invalidf("+incompatible suffix not allowed: module path includes a major version suffix, so major version must match")
				} else {
					return nil, invalidf("+incompatible suffix not allowed: module contains a go.mod file, so semantic import versioning is required")
				}
			}

			if err := module.CheckPathMajor(strings.TrimSuffix(info2.Version, "+incompatible"), r.pathMajor); err == nil {
				return nil, invalidf("+incompatible suffix not allowed: major version %s is compatible", semver.Major(info2.Version))
			}
		}

		return info2, nil
	}

	// Determine version.
	//
	// If statVers is canonical, then the original call was repo.Stat(statVers).
	// Since the version is canonical, we must not resolve it to anything but
	// itself, possibly with a '+incompatible' annotation: we do not need to do
	// the work required to look for an arbitrary pseudo-version.
	if statVers != "" && statVers == module.CanonicalVersion(statVers) {
		info2.Version = statVers

		if module.IsPseudoVersion(info2.Version) {
			if err := r.validatePseudoVersion(info, info2.Version); err != nil {
				return nil, err
			}
			return checkGoMod()
		}

		if err := module.CheckPathMajor(info2.Version, r.pathMajor); err != nil {
			if canUseIncompatible() {
				info2.Version += "+incompatible"
				return checkGoMod()
			} else {
				if vErr, ok := err.(*module.InvalidVersionError); ok {
					// We're going to describe why the version is invalid in more detail,
					// so strip out the existing “invalid version” wrapper.
					err = vErr.Err
				}
				return nil, invalidf("module contains a go.mod file, so major version must be compatible: %v", err)
			}
		}

		return checkGoMod()
	}

	// statVers is empty or non-canonical, so we need to resolve it to a canonical
	// version or pseudo-version.

	// Derive or verify a version from a code repo tag.
	// Tag must have a prefix matching codeDir.
	tagPrefix := ""
	if r.codeDir != "" {
		tagPrefix = r.codeDir + "/"
	}

	isRetracted, err := r.retractedVersions()
	if err != nil {
		isRetracted = func(string) bool { return false }
	}

	// tagToVersion returns the version obtained by trimming tagPrefix from tag.
	// If the tag is invalid, retracted, or a pseudo-version, tagToVersion returns
	// an empty version.
	tagToVersion := func(tag string) (v string, tagIsCanonical bool) {
		if !strings.HasPrefix(tag, tagPrefix) {
			return "", false
		}
		trimmed := tag[len(tagPrefix):]
		// Tags that look like pseudo-versions would be confusing. Ignore them.
		if module.IsPseudoVersion(tag) {
			return "", false
		}

		v = semver.Canonical(trimmed) // Not module.Canonical: we don't want to pick up an explicit "+incompatible" suffix from the tag.
		if v == "" || !strings.HasPrefix(trimmed, v) {
			return "", false // Invalid or incomplete version (just vX or vX.Y).
		}
		if isRetracted(v) {
			return "", false
		}
		if v == trimmed {
			tagIsCanonical = true
		}

		if err := module.CheckPathMajor(v, r.pathMajor); err != nil {
			if canUseIncompatible() {
				return v + "+incompatible", tagIsCanonical
			}
			return "", false
		}

		return v, tagIsCanonical
	}

	// If the VCS gave us a valid version, use that.
	if v, tagIsCanonical := tagToVersion(info.Version); tagIsCanonical {
		info2.Version = v
		return checkGoMod()
	}

	// Look through the tags on the revision for either a usable canonical version
	// or an appropriate base for a pseudo-version.
	var pseudoBase string
	for _, pathTag := range info.Tags {
		v, tagIsCanonical := tagToVersion(pathTag)
		if tagIsCanonical {
			if statVers != "" && semver.Compare(v, statVers) == 0 {
				// The user requested a non-canonical version, but the tag for the
				// canonical equivalent refers to the same revision. Use it.
				info2.Version = v
				return checkGoMod()
			} else {
				// Save the highest canonical tag for the revision. If we don't find a
				// better match, we'll use it as the canonical version.
				//
				// NOTE: Do not replace this with semver.Max. Despite the name,
				// semver.Max *also* canonicalizes its arguments, which uses
				// semver.Canonical instead of module.CanonicalVersion and thereby
				// strips our "+incompatible" suffix.
				if semver.Compare(info2.Version, v) < 0 {
					info2.Version = v
				}
			}
		} else if v != "" && semver.Compare(v, statVers) == 0 {
			// The user explicitly requested something equivalent to this tag. We
			// can't use the version from the tag directly: since the tag is not
			// canonical, it could be ambiguous. For example, tags v0.0.1+a and
			// v0.0.1+b might both exist and refer to different revisions.
			//
			// The tag is otherwise valid for the module, so we can at least use it as
			// the base of an unambiguous pseudo-version.
			//
			// If multiple tags match, tagToVersion will canonicalize them to the same
			// base version.
			pseudoBase = v
		}
	}

	// If we found any canonical tag for the revision, return it.
	// Even if we found a good pseudo-version base, a canonical version is better.
	if info2.Version != "" {
		return checkGoMod()
	}

	// Find the highest tagged version in the revision's history, subject to
	// major version and +incompatible constraints. Use that version as the
	// pseudo-version base so that the pseudo-version sorts higher. Ignore
	// retracted versions.
	allowedMajor := func(major string) func(v string) bool {
		return func(v string) bool {
			return (major == "" || semver.Major(v) == major) && !isRetracted(v)
		}
	}
	if pseudoBase == "" {
		var tag string
		if r.pseudoMajor != "" || canUseIncompatible() {
			tag, _ = r.code.RecentTag(info.Name, tagPrefix, allowedMajor(r.pseudoMajor))
		} else {
			// Allow either v1 or v0, but not incompatible higher versions.
			tag, _ = r.code.RecentTag(info.Name, tagPrefix, allowedMajor("v1"))
			if tag == "" {
				tag, _ = r.code.RecentTag(info.Name, tagPrefix, allowedMajor("v0"))
			}
		}
		pseudoBase, _ = tagToVersion(tag) // empty if the tag is invalid
	}

	info2.Version = module.PseudoVersion(r.pseudoMajor, pseudoBase, info.Time, info.Short)
	return checkGoMod()
}

// validatePseudoVersion checks that version has a major version compatible with
// r.modPath and encodes a base version and commit metadata that agrees with
// info.
//
// Note that verifying a nontrivial base version in particular may be somewhat
// expensive: in order to do so, r.code.DescendsFrom will need to fetch at least
// enough of the commit history to find a path between version and its base.
// Fortunately, many pseudo-versions — such as those for untagged repositories —
// have trivial bases!
func (r *codeRepo) validatePseudoVersion(info *codehost.RevInfo, version string) (err error) {
	defer func() {
		if err != nil {
			if _, ok := err.(*module.ModuleError); !ok {
				if _, ok := err.(*module.InvalidVersionError); !ok {
					err = &module.InvalidVersionError{Version: version, Pseudo: true, Err: err}
				}
				err = &module.ModuleError{Path: r.modPath, Err: err}
			}
		}
	}()

	if err := module.CheckPathMajor(version, r.pathMajor); err != nil {
		return err
	}

	rev, err := module.PseudoVersionRev(version)
	if err != nil {
		return err
	}
	if rev != info.Short {
		switch {
		case strings.HasPrefix(rev, info.Short):
			return fmt.Errorf("revision is longer than canonical (%s)", info.Short)
		case strings.HasPrefix(info.Short, rev):
			return fmt.Errorf("revision is shorter than canonical (%s)", info.Short)
		default:
			return fmt.Errorf("does not match short name of revision (%s)", info.Short)
		}
	}

	t, err := module.PseudoVersionTime(version)
	if err != nil {
		return err
	}
	if !t.Equal(info.Time.Truncate(time.Second)) {
		return fmt.Errorf("does not match version-control timestamp (expected %s)", info.Time.UTC().Format(module.PseudoVersionTimestampFormat))
	}

	tagPrefix := ""
	if r.codeDir != "" {
		tagPrefix = r.codeDir + "/"
	}

	// A pseudo-version should have a precedence just above its parent revisions,
	// and no higher. Otherwise, it would be possible for library authors to "pin"
	// dependency versions (and bypass the usual minimum version selection) by
	// naming an extremely high pseudo-version rather than an accurate one.
	//
	// Moreover, if we allow a pseudo-version to use any arbitrary pre-release
	// tag, we end up with infinitely many possible names for each commit. Each
	// name consumes resources in the module cache and proxies, so we want to
	// restrict them to a finite set under control of the module author.
	//
	// We address both of these issues by requiring the tag upon which the
	// pseudo-version is based to refer to some ancestor of the revision. We
	// prefer the highest such tag when constructing a new pseudo-version, but do
	// not enforce that property when resolving existing pseudo-versions: we don't
	// know when the parent tags were added, and the highest-tagged parent may not
	// have existed when the pseudo-version was first resolved.
	base, err := module.PseudoVersionBase(strings.TrimSuffix(version, "+incompatible"))
	if err != nil {
		return err
	}
	if base == "" {
		if r.pseudoMajor == "" && semver.Major(version) == "v1" {
			return fmt.Errorf("major version without preceding tag must be v0, not v1")
		}
		return nil
	} else {
		for _, tag := range info.Tags {
			versionOnly := strings.TrimPrefix(tag, tagPrefix)
			if versionOnly == base {
				// The base version is canonical, so if the version from the tag is
				// literally equal (not just equivalent), then the tag is canonical too.
				//
				// We allow pseudo-versions to be derived from non-canonical tags on the
				// same commit, so that tags like "v1.1.0+some-metadata" resolve as
				// close as possible to the canonical version ("v1.1.0") while still
				// enforcing a total ordering ("v1.1.1-0.[…]" with a unique suffix).
				//
				// However, canonical tags already have a total ordering, so there is no
				// reason not to use the canonical tag directly, and we know that the
				// canonical tag must already exist because the pseudo-version is
				// derived from it. In that case, referring to the revision by a
				// pseudo-version derived from its own canonical tag is just confusing.
				return fmt.Errorf("tag (%s) found on revision %s is already canonical, so should not be replaced with a pseudo-version derived from that tag", tag, rev)
			}
		}
	}

	tags, err := r.code.Tags(tagPrefix + base)
	if err != nil {
		return err
	}

	var lastTag string // Prefer to log some real tag rather than a canonically-equivalent base.
	ancestorFound := false
	for _, tag := range tags {
		versionOnly := strings.TrimPrefix(tag, tagPrefix)
		if semver.Compare(versionOnly, base) == 0 {
			lastTag = tag
			ancestorFound, err = r.code.DescendsFrom(info.Name, tag)
			if ancestorFound {
				break
			}
		}
	}

	if lastTag == "" {
		return fmt.Errorf("preceding tag (%s) not found", base)
	}

	if !ancestorFound {
		if err != nil {
			return err
		}
		rev, err := module.PseudoVersionRev(version)
		if err != nil {
			return fmt.Errorf("not a descendent of preceding tag (%s)", lastTag)
		}
		return fmt.Errorf("revision %s is not a descendent of preceding tag (%s)", rev, lastTag)
	}
	return nil
}

func (r *codeRepo) revToRev(rev string) string {
	if semver.IsValid(rev) {
		if module.IsPseudoVersion(rev) {
			r, _ := module.PseudoVersionRev(rev)
			return r
		}
		if semver.Build(rev) == "+incompatible" {
			rev = rev[:len(rev)-len("+incompatible")]
		}
		if r.codeDir == "" {
			return rev
		}
		return r.codeDir + "/" + rev
	}
	return rev
}

func (r *codeRepo) versionToRev(version string) (rev string, err error) {
	if !semver.IsValid(version) {
		return "", &module.ModuleError{
			Path: r.modPath,
			Err: &module.InvalidVersionError{
				Version: version,
				Err:     errors.New("syntax error"),
			},
		}
	}
	return r.revToRev(version), nil
}

// findDir locates the directory within the repo containing the module.
//
// If r.pathMajor is non-empty, this can be either r.codeDir or — if a go.mod
// file exists — r.codeDir/r.pathMajor[1:].
func (r *codeRepo) findDir(version string) (rev, dir string, gomod []byte, err error) {
	rev, err = r.versionToRev(version)
	if err != nil {
		return "", "", nil, err
	}

	// Load info about go.mod but delay consideration
	// (except I/O error) until we rule out v2/go.mod.
	file1 := path.Join(r.codeDir, "go.mod")
	gomod1, err1 := r.code.ReadFile(rev, file1, codehost.MaxGoMod)
	if err1 != nil && !os.IsNotExist(err1) {
		return "", "", nil, fmt.Errorf("reading %s/%s at revision %s: %v", r.pathPrefix, file1, rev, err1)
	}
	mpath1 := modfile.ModulePath(gomod1)
	found1 := err1 == nil && (isMajor(mpath1, r.pathMajor) || r.canReplaceMismatchedVersionDueToBug(mpath1))

	var file2 string
	if r.pathMajor != "" && r.codeRoot != r.modPath && !strings.HasPrefix(r.pathMajor, ".") {
		// Suppose pathMajor is "/v2".
		// Either go.mod should claim v2 and v2/go.mod should not exist,
		// or v2/go.mod should exist and claim v2. Not both.
		// Note that we don't check the full path, just the major suffix,
		// because of replacement modules. This might be a fork of
		// the real module, found at a different path, usable only in
		// a replace directive.
		dir2 := path.Join(r.codeDir, r.pathMajor[1:])
		file2 = path.Join(dir2, "go.mod")
		gomod2, err2 := r.code.ReadFile(rev, file2, codehost.MaxGoMod)
		if err2 != nil && !os.IsNotExist(err2) {
			return "", "", nil, fmt.Errorf("reading %s/%s at revision %s: %v", r.pathPrefix, file2, rev, err2)
		}
		mpath2 := modfile.ModulePath(gomod2)
		found2 := err2 == nil && isMajor(mpath2, r.pathMajor)

		if found1 && found2 {
			return "", "", nil, fmt.Errorf("%s/%s and ...%s/go.mod both have ...%s module paths at revision %s", r.pathPrefix, file1, r.pathMajor, r.pathMajor, rev)
		}
		if found2 {
			return rev, dir2, gomod2, nil
		}
		if err2 == nil {
			if mpath2 == "" {
				return "", "", nil, fmt.Errorf("%s/%s is missing module path at revision %s", r.pathPrefix, file2, rev)
			}
			return "", "", nil, fmt.Errorf("%s/%s has non-...%s module path %q at revision %s", r.pathPrefix, file2, r.pathMajor, mpath2, rev)
		}
	}

	// Not v2/go.mod, so it's either go.mod or nothing. Which is it?
	if found1 {
		// Explicit go.mod with matching major version ok.
		return rev, r.codeDir, gomod1, nil
	}
	if err1 == nil {
		// Explicit go.mod with non-matching major version disallowed.
		suffix := ""
		if file2 != "" {
			suffix = fmt.Sprintf(" (and ...%s/go.mod does not exist)", r.pathMajor)
		}
		if mpath1 == "" {
			return "", "", nil, fmt.Errorf("%s is missing module path%s at revision %s", file1, suffix, rev)
		}
		if r.pathMajor != "" { // ".v1", ".v2" for gopkg.in
			return "", "", nil, fmt.Errorf("%s has non-...%s module path %q%s at revision %s", file1, r.pathMajor, mpath1, suffix, rev)
		}
		if _, _, ok := module.SplitPathVersion(mpath1); !ok {
			return "", "", nil, fmt.Errorf("%s has malformed module path %q%s at revision %s", file1, mpath1, suffix, rev)
		}
		return "", "", nil, fmt.Errorf("%s has post-%s module path %q%s at revision %s", file1, semver.Major(version), mpath1, suffix, rev)
	}

	if r.codeDir == "" && (r.pathMajor == "" || strings.HasPrefix(r.pathMajor, ".")) {
		// Implicit go.mod at root of repo OK for v0/v1 and for gopkg.in.
		return rev, "", nil, nil
	}

	// Implicit go.mod below root of repo or at v2+ disallowed.
	// Be clear about possibility of using either location for v2+.
	if file2 != "" {
		return "", "", nil, fmt.Errorf("missing %s/go.mod and ...%s/go.mod at revision %s", r.pathPrefix, r.pathMajor, rev)
	}
	return "", "", nil, fmt.Errorf("missing %s/go.mod at revision %s", r.pathPrefix, rev)
}

// isMajor reports whether the versions allowed for mpath are compatible with
// the major version(s) implied by pathMajor, or false if mpath has an invalid
// version suffix.
func isMajor(mpath, pathMajor string) bool {
	if mpath == "" {
		// If we don't have a path, we don't know what version(s) it is compatible with.
		return false
	}
	_, mpathMajor, ok := module.SplitPathVersion(mpath)
	if !ok {
		// An invalid module path is not compatible with any version.
		return false
	}
	if pathMajor == "" {
		// All of the valid versions for a gopkg.in module that requires major
		// version v0 or v1 are compatible with the "v0 or v1" implied by an empty
		// pathMajor.
		switch module.PathMajorPrefix(mpathMajor) {
		case "", "v0", "v1":
			return true
		default:
			return false
		}
	}
	if mpathMajor == "" {
		// Even if pathMajor is ".v0" or ".v1", we can't be sure that a module
		// without a suffix is tagged appropriately. Besides, we don't expect clones
		// of non-gopkg.in modules to have gopkg.in paths, so a non-empty,
		// non-gopkg.in mpath is probably the wrong module for any such pathMajor
		// anyway.
		return false
	}
	// If both pathMajor and mpathMajor are non-empty, then we only care that they
	// have the same major-version validation rules. A clone fetched via a /v2
	// path might replace a module with path gopkg.in/foo.v2-unstable, and that's
	// ok.
	return pathMajor[1:] == mpathMajor[1:]
}

// canReplaceMismatchedVersionDueToBug reports whether versions of r
// could replace versions of mpath with otherwise-mismatched major versions
// due to a historical bug in the Go command (golang.org/issue/34254).
func (r *codeRepo) canReplaceMismatchedVersionDueToBug(mpath string) bool {
	// The bug caused us to erroneously accept unversioned paths as replacements
	// for versioned gopkg.in paths.
	unversioned := r.pathMajor == ""
	replacingGopkgIn := strings.HasPrefix(mpath, "gopkg.in/")
	return unversioned && replacingGopkgIn
}

func (r *codeRepo) GoMod(version string) (data []byte, err error) {
	if version != module.CanonicalVersion(version) {
		return nil, fmt.Errorf("version %s is not canonical", version)
	}

	if module.IsPseudoVersion(version) {
		// findDir ignores the metadata encoded in a pseudo-version,
		// only using the revision at the end.
		// Invoke Stat to verify the metadata explicitly so we don't return
		// a bogus file for an invalid version.
		_, err := r.Stat(version)
		if err != nil {
			return nil, err
		}
	}

	rev, dir, gomod, err := r.findDir(version)
	if err != nil {
		return nil, err
	}
	if gomod != nil {
		return gomod, nil
	}
	data, err = r.code.ReadFile(rev, path.Join(dir, "go.mod"), codehost.MaxGoMod)
	if err != nil {
		if os.IsNotExist(err) {
			return LegacyGoMod(r.modPath), nil
		}
		return nil, err
	}
	return data, nil
}

// LegacyGoMod generates a fake go.mod file for a module that doesn't have one.
// The go.mod file contains a module directive and nothing else: no go version,
// no requirements.
//
// We used to try to build a go.mod reflecting pre-existing
// package management metadata files, but the conversion
// was inherently imperfect (because those files don't have
// exactly the same semantics as go.mod) and, when done
// for dependencies in the middle of a build, impossible to
// correct. So we stopped.
func LegacyGoMod(modPath string) []byte {
	return []byte(fmt.Sprintf("module %s\n", modfile.AutoQuote(modPath)))
}

func (r *codeRepo) modPrefix(rev string) string {
	return r.modPath + "@" + rev
}

func (r *codeRepo) retractedVersions() (func(string) bool, error) {
	versions, err := r.Versions("")
	if err != nil {
		return nil, err
	}

	for i, v := range versions {
		if strings.HasSuffix(v, "+incompatible") {
			versions = versions[:i]
			break
		}
	}
	if len(versions) == 0 {
		return func(string) bool { return false }, nil
	}

	var highest string
	for i := len(versions) - 1; i >= 0; i-- {
		v := versions[i]
		if semver.Prerelease(v) == "" {
			highest = v
			break
		}
	}
	if highest == "" {
		highest = versions[len(versions)-1]
	}

	data, err := r.GoMod(highest)
	if err != nil {
		return nil, err
	}
	f, err := modfile.ParseLax("go.mod", data, nil)
	if err != nil {
		return nil, err
	}
	retractions := make([]modfile.VersionInterval, len(f.Retract))
	for _, r := range f.Retract {
		retractions = append(retractions, r.VersionInterval)
	}

	return func(v string) bool {
		for _, r := range retractions {
			if semver.Compare(r.Low, v) <= 0 && semver.Compare(v, r.High) <= 0 {
				return true
			}
		}
		return false
	}, nil
}

func (r *codeRepo) Zip(dst io.Writer, version string) error {
	if version != module.CanonicalVersion(version) {
		return fmt.Errorf("version %s is not canonical", version)
	}

	if module.IsPseudoVersion(version) {
		// findDir ignores the metadata encoded in a pseudo-version,
		// only using the revision at the end.
		// Invoke Stat to verify the metadata explicitly so we don't return
		// a bogus file for an invalid version.
		_, err := r.Stat(version)
		if err != nil {
			return err
		}
	}

	rev, subdir, _, err := r.findDir(version)
	if err != nil {
		return err
	}
	dl, err := r.code.ReadZip(rev, subdir, codehost.MaxZipFile)
	if err != nil {
		return err
	}
	defer dl.Close()
	subdir = strings.Trim(subdir, "/")

	// Spool to local file.
	f, err := os.CreateTemp("", "go-codehost-")
	if err != nil {
		dl.Close()
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	maxSize := int64(codehost.MaxZipFile)
	lr := &io.LimitedReader{R: dl, N: maxSize + 1}
	if _, err := io.Copy(f, lr); err != nil {
		dl.Close()
		return err
	}
	dl.Close()
	if lr.N <= 0 {
		return fmt.Errorf("downloaded zip file too large")
	}
	size := (maxSize + 1) - lr.N
	if _, err := f.Seek(0, 0); err != nil {
		return err
	}

	// Translate from zip file we have to zip file we want.
	zr, err := zip.NewReader(f, size)
	if err != nil {
		return err
	}

	var files []modzip.File
	if subdir != "" {
		subdir += "/"
	}
	haveLICENSE := false
	topPrefix := ""
	for _, zf := range zr.File {
		if topPrefix == "" {
			i := strings.Index(zf.Name, "/")
			if i < 0 {
				return fmt.Errorf("missing top-level directory prefix")
			}
			topPrefix = zf.Name[:i+1]
		}
		if !strings.HasPrefix(zf.Name, topPrefix) {
			return fmt.Errorf("zip file contains more than one top-level directory")
		}
		name := strings.TrimPrefix(zf.Name, topPrefix)
		if !strings.HasPrefix(name, subdir) {
			continue
		}
		name = strings.TrimPrefix(name, subdir)
		if name == "" || strings.HasSuffix(name, "/") {
			continue
		}
		files = append(files, zipFile{name: name, f: zf})
		if name == "LICENSE" {
			haveLICENSE = true
		}
	}

	if !haveLICENSE && subdir != "" {
		data, err := r.code.ReadFile(rev, "LICENSE", codehost.MaxLICENSE)
		if err == nil {
			files = append(files, dataFile{name: "LICENSE", data: data})
		}
	}

	return modzip.Create(dst, module.Version{Path: r.modPath, Version: version}, files)
}

type zipFile struct {
	name string
	f    *zip.File
}

func (f zipFile) Path() string                 { return f.name }
func (f zipFile) Lstat() (fs.FileInfo, error)  { return f.f.FileInfo(), nil }
func (f zipFile) Open() (io.ReadCloser, error) { return f.f.Open() }

type dataFile struct {
	name string
	data []byte
}

func (f dataFile) Path() string                { return f.name }
func (f dataFile) Lstat() (fs.FileInfo, error) { return dataFileInfo{f}, nil }
func (f dataFile) Open() (io.ReadCloser, error) {
	return io.NopCloser(bytes.NewReader(f.data)), nil
}

type dataFileInfo struct {
	f dataFile
}

func (fi dataFileInfo) Name() string       { return path.Base(fi.f.name) }
func (fi dataFileInfo) Size() int64        { return int64(len(fi.f.data)) }
func (fi dataFileInfo) Mode() fs.FileMode  { return 0644 }
func (fi dataFileInfo) ModTime() time.Time { return time.Time{} }
func (fi dataFileInfo) IsDir() bool        { return false }
func (fi dataFileInfo) Sys() interface{}   { return nil }

// hasPathPrefix reports whether the path s begins with the
// elements in prefix.
func hasPathPrefix(s, prefix string) bool {
	switch {
	default:
		return false
	case len(s) == len(prefix):
		return s == prefix
	case len(s) > len(prefix):
		if prefix != "" && prefix[len(prefix)-1] == '/' {
			return strings.HasPrefix(s, prefix)
		}
		return s[len(prefix)] == '/' && s[:len(prefix)] == prefix
	}
}
