// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

// A codeRepo implements modfetch.Repo using an underlying codehost.Repo.
type codeRepo struct {
	modPath string

	code     codehost.Repo
	codeRoot string
	codeDir  string

	path        string
	pathPrefix  string
	pathMajor   string
	pseudoMajor string
}

func newCodeRepo(code codehost.Repo, root, path string) (Repo, error) {
	if !hasPathPrefix(path, root) {
		return nil, fmt.Errorf("mismatched repo: found %s for %s", root, path)
	}
	pathPrefix, pathMajor, ok := module.SplitPathVersion(path)
	if !ok {
		return nil, fmt.Errorf("invalid module path %q", path)
	}
	pseudoMajor := "v0"
	if pathMajor != "" {
		pseudoMajor = pathMajor[1:]
	}

	// At this point we might have:
	//	codeRoot = github.com/rsc/foo
	//	path = github.com/rsc/foo/bar/v2
	//	pathPrefix = github.com/rsc/foo/bar
	//	pathMajor = /v2
	//	pseudoMajor = v2
	//
	// Compute codeDir = bar, the subdirectory within the repo
	// corresponding to the module root.
	codeDir := strings.Trim(strings.TrimPrefix(pathPrefix, root), "/")
	if strings.HasPrefix(path, "gopkg.in/") {
		// But gopkg.in is a special legacy case, in which pathPrefix does not start with codeRoot.
		// For example we might have:
		//	codeRoot = gopkg.in/yaml.v2
		//	pathPrefix = gopkg.in/yaml
		//	pathMajor = .v2
		//	pseudoMajor = v2
		//	codeDir = pathPrefix (because codeRoot is not a prefix of pathPrefix)
		// Clear codeDir - the module root is the repo root for gopkg.in repos.
		codeDir = ""
	}

	r := &codeRepo{
		modPath:     path,
		code:        code,
		codeRoot:    root,
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
		return nil, err
	}

	list := []string{}
	var incompatible []string
	for _, tag := range tags {
		if !strings.HasPrefix(tag, p) {
			continue
		}
		v := tag
		if r.codeDir != "" {
			v = v[len(r.codeDir)+1:]
		}
		if v == "" || v != module.CanonicalVersion(v) || IsPseudoVersion(v) {
			continue
		}
		if !module.MatchPathMajor(v, r.pathMajor) {
			if r.codeDir == "" && r.pathMajor == "" && semver.Major(v) > "v1" {
				incompatible = append(incompatible, v)
			}
			continue
		}
		list = append(list, v)
	}

	if len(incompatible) > 0 {
		// Check for later versions that were created not following semantic import versioning,
		// as indicated by the absence of a go.mod file. Those versions can be addressed
		// by referring to them with a +incompatible suffix, as in v17.0.0+incompatible.
		files, err := r.code.ReadFileRevs(incompatible, "go.mod", codehost.MaxGoMod)
		if err != nil {
			return nil, err
		}
		for _, rev := range incompatible {
			f := files[rev]
			if os.IsNotExist(f.Err) {
				list = append(list, rev+"+incompatible")
			}
		}
	}

	SortVersions(list)
	return list, nil
}

func (r *codeRepo) Stat(rev string) (*RevInfo, error) {
	if rev == "latest" {
		return r.Latest()
	}
	codeRev := r.revToRev(rev)
	if semver.IsValid(codeRev) && r.codeDir != "" {
		codeRev = r.codeDir + "/" + codeRev
	}
	info, err := r.code.Stat(codeRev)
	if err != nil {
		return nil, err
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

func (r *codeRepo) convert(info *codehost.RevInfo, statVers string) (*RevInfo, error) {
	info2 := &RevInfo{
		Name:  info.Name,
		Short: info.Short,
		Time:  info.Time,
	}

	// Determine version.
	if module.CanonicalVersion(statVers) == statVers && module.MatchPathMajor(statVers, r.pathMajor) {
		// The original call was repo.Stat(statVers), and requestedVersion is OK, so use it.
		info2.Version = statVers
	} else {
		// Otherwise derive a version from a code repo tag.
		// Tag must have a prefix matching codeDir.
		p := ""
		if r.codeDir != "" {
			p = r.codeDir + "/"
		}

		// If this is a plain tag (no dir/ prefix)
		// and the module path is unversioned,
		// and if the underlying file tree has no go.mod,
		// then allow using the tag with a +incompatible suffix.
		canUseIncompatible := false
		if r.codeDir == "" && r.pathMajor == "" {
			_, errGoMod := r.code.ReadFile(info.Name, "go.mod", codehost.MaxGoMod)
			if errGoMod != nil {
				canUseIncompatible = true
			}
		}

		tagToVersion := func(v string) string {
			if !strings.HasPrefix(v, p) {
				return ""
			}
			v = v[len(p):]
			if module.CanonicalVersion(v) != v || IsPseudoVersion(v) {
				return ""
			}
			if module.MatchPathMajor(v, r.pathMajor) {
				return v
			}
			if canUseIncompatible {
				return v + "+incompatible"
			}
			return ""
		}

		// If info.Version is OK, use it.
		if v := tagToVersion(info.Version); v != "" {
			info2.Version = v
		} else {
			// Otherwise look through all known tags for latest in semver ordering.
			for _, tag := range info.Tags {
				if v := tagToVersion(tag); v != "" && semver.Compare(info2.Version, v) < 0 {
					info2.Version = v
				}
			}
			// Otherwise make a pseudo-version.
			if info2.Version == "" {
				tag, _ := r.code.RecentTag(statVers, p)
				v = tagToVersion(tag)
				// TODO: Check that v is OK for r.pseudoMajor or else is OK for incompatible.
				info2.Version = PseudoVersion(r.pseudoMajor, v, info.Time, info.Short)
			}
		}
	}

	// Do not allow a successful stat of a pseudo-version for a subdirectory
	// unless the subdirectory actually does have a go.mod.
	if IsPseudoVersion(info2.Version) && r.codeDir != "" {
		_, _, _, err := r.findDir(info2.Version)
		if err != nil {
			// TODO: It would be nice to return an error like "not a module".
			// Right now we return "missing go.mod", which is a little confusing.
			return nil, err
		}
	}

	return info2, nil
}

func (r *codeRepo) revToRev(rev string) string {
	if semver.IsValid(rev) {
		if IsPseudoVersion(rev) {
			r, _ := PseudoVersionRev(rev)
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
		return "", fmt.Errorf("malformed semantic version %q", version)
	}
	return r.revToRev(version), nil
}

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
	found1 := err1 == nil && isMajor(mpath1, r.pathMajor)

	var file2 string
	if r.pathMajor != "" && !strings.HasPrefix(r.pathMajor, ".") {
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
		// Explicit go.mod with matching module path OK.
		return rev, r.codeDir, gomod1, nil
	}
	if err1 == nil {
		// Explicit go.mod with non-matching module path disallowed.
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

func isMajor(mpath, pathMajor string) bool {
	if mpath == "" {
		return false
	}
	if pathMajor == "" {
		// mpath must NOT have version suffix.
		i := len(mpath)
		for i > 0 && '0' <= mpath[i-1] && mpath[i-1] <= '9' {
			i--
		}
		if i < len(mpath) && i >= 2 && mpath[i-1] == 'v' && mpath[i-2] == '/' {
			// Found valid suffix.
			return false
		}
		return true
	}
	// Otherwise pathMajor is ".v1", ".v2" (gopkg.in), or "/v2", "/v3" etc.
	return strings.HasSuffix(mpath, pathMajor)
}

func (r *codeRepo) GoMod(version string) (data []byte, err error) {
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
			return r.legacyGoMod(rev, dir), nil
		}
		return nil, err
	}
	return data, nil
}

func (r *codeRepo) legacyGoMod(rev, dir string) []byte {
	// We used to try to build a go.mod reflecting pre-existing
	// package management metadata files, but the conversion
	// was inherently imperfect (because those files don't have
	// exactly the same semantics as go.mod) and, when done
	// for dependencies in the middle of a build, impossible to
	// correct. So we stopped.
	// Return a fake go.mod that simply declares the module path.
	return []byte(fmt.Sprintf("module %s\n", modfile.AutoQuote(r.modPath)))
}

func (r *codeRepo) modPrefix(rev string) string {
	return r.modPath + "@" + rev
}

func (r *codeRepo) Zip(version string, tmpdir string) (tmpfile string, err error) {
	rev, dir, _, err := r.findDir(version)
	if err != nil {
		return "", err
	}
	dl, actualDir, err := r.code.ReadZip(rev, dir, codehost.MaxZipFile)
	if err != nil {
		return "", err
	}
	if actualDir != "" && !hasPathPrefix(dir, actualDir) {
		return "", fmt.Errorf("internal error: downloading %v %v: dir=%q but actualDir=%q", r.path, rev, dir, actualDir)
	}
	subdir := strings.Trim(strings.TrimPrefix(dir, actualDir), "/")

	// Spool to local file.
	f, err := ioutil.TempFile(tmpdir, "go-codehost-")
	if err != nil {
		dl.Close()
		return "", err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	maxSize := int64(codehost.MaxZipFile)
	lr := &io.LimitedReader{R: dl, N: maxSize + 1}
	if _, err := io.Copy(f, lr); err != nil {
		dl.Close()
		return "", err
	}
	dl.Close()
	if lr.N <= 0 {
		return "", fmt.Errorf("downloaded zip file too large")
	}
	size := (maxSize + 1) - lr.N
	if _, err := f.Seek(0, 0); err != nil {
		return "", err
	}

	// Translate from zip file we have to zip file we want.
	zr, err := zip.NewReader(f, size)
	if err != nil {
		return "", err
	}
	f2, err := ioutil.TempFile(tmpdir, "go-codezip-")
	if err != nil {
		return "", err
	}

	zw := zip.NewWriter(f2)
	newName := f2.Name()
	defer func() {
		f2.Close()
		if err != nil {
			os.Remove(newName)
		}
	}()
	if subdir != "" {
		subdir += "/"
	}
	haveLICENSE := false
	topPrefix := ""
	haveGoMod := make(map[string]bool)
	for _, zf := range zr.File {
		if topPrefix == "" {
			i := strings.Index(zf.Name, "/")
			if i < 0 {
				return "", fmt.Errorf("missing top-level directory prefix")
			}
			topPrefix = zf.Name[:i+1]
		}
		if !strings.HasPrefix(zf.Name, topPrefix) {
			return "", fmt.Errorf("zip file contains more than one top-level directory")
		}
		dir, file := path.Split(zf.Name)
		if file == "go.mod" {
			haveGoMod[dir] = true
		}
	}
	root := topPrefix + subdir
	inSubmodule := func(name string) bool {
		for {
			dir, _ := path.Split(name)
			if len(dir) <= len(root) {
				return false
			}
			if haveGoMod[dir] {
				return true
			}
			name = dir[:len(dir)-1]
		}
	}
	for _, zf := range zr.File {
		if topPrefix == "" {
			i := strings.Index(zf.Name, "/")
			if i < 0 {
				return "", fmt.Errorf("missing top-level directory prefix")
			}
			topPrefix = zf.Name[:i+1]
		}
		if strings.HasSuffix(zf.Name, "/") { // drop directory dummy entries
			continue
		}
		if !strings.HasPrefix(zf.Name, topPrefix) {
			return "", fmt.Errorf("zip file contains more than one top-level directory")
		}
		name := strings.TrimPrefix(zf.Name, topPrefix)
		if !strings.HasPrefix(name, subdir) {
			continue
		}
		if name == ".hg_archival.txt" {
			// Inserted by hg archive.
			// Not correct to drop from other version control systems, but too bad.
			continue
		}
		name = strings.TrimPrefix(name, subdir)
		if isVendoredPackage(name) {
			continue
		}
		if inSubmodule(zf.Name) {
			continue
		}
		base := path.Base(name)
		if strings.ToLower(base) == "go.mod" && base != "go.mod" {
			return "", fmt.Errorf("zip file contains %s, want all lower-case go.mod", zf.Name)
		}
		if name == "LICENSE" {
			haveLICENSE = true
		}
		size := int64(zf.UncompressedSize)
		if size < 0 || maxSize < size {
			return "", fmt.Errorf("module source tree too big")
		}
		maxSize -= size

		rc, err := zf.Open()
		if err != nil {
			return "", err
		}
		w, err := zw.Create(r.modPrefix(version) + "/" + name)
		lr := &io.LimitedReader{R: rc, N: size + 1}
		if _, err := io.Copy(w, lr); err != nil {
			return "", err
		}
		if lr.N <= 0 {
			return "", fmt.Errorf("individual file too large")
		}
	}

	if !haveLICENSE && subdir != "" {
		data, err := r.code.ReadFile(rev, "LICENSE", codehost.MaxLICENSE)
		if err == nil {
			w, err := zw.Create(r.modPrefix(version) + "/LICENSE")
			if err != nil {
				return "", err
			}
			if _, err := w.Write(data); err != nil {
				return "", err
			}
		}
	}
	if err := zw.Close(); err != nil {
		return "", err
	}
	if err := f2.Close(); err != nil {
		return "", err
	}

	return f2.Name(), nil
}

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

func isVendoredPackage(name string) bool {
	var i int
	if strings.HasPrefix(name, "vendor/") {
		i += len("vendor/")
	} else if j := strings.Index(name, "/vendor/"); j >= 0 {
		i += len("/vendor/")
	} else {
		return false
	}
	return strings.Contains(name[i:], "/")
}
