// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/mvs"
	"cmd/go/internal/par"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// mvsReqs implements mvs.Reqs for module semantic versions,
// with any exclusions or replacements applied internally.
type mvsReqs struct {
	buildList []module.Version
	cache     par.Cache
	versions  sync.Map
}

// Reqs returns the current module requirement graph.
// Future calls to SetBuildList do not affect the operation
// of the returned Reqs.
func Reqs() mvs.Reqs {
	r := &mvsReqs{
		buildList: buildList,
	}
	return r
}

func (r *mvsReqs) Required(mod module.Version) ([]module.Version, error) {
	type cached struct {
		list []module.Version
		err  error
	}

	c := r.cache.Do(mod, func() interface{} {
		list, err := r.required(mod)
		if err != nil {
			return cached{nil, err}
		}
		for i, mv := range list {
			if index != nil {
				for index.exclude[mv] {
					mv1, err := r.next(mv)
					if err != nil {
						return cached{nil, err}
					}
					if mv1.Version == "none" {
						return cached{nil, fmt.Errorf("%s(%s) depends on excluded %s(%s) with no newer version available", mod.Path, mod.Version, mv.Path, mv.Version)}
					}
					mv = mv1
				}
			}
			list[i] = mv
		}

		return cached{list, nil}
	}).(cached)

	return c.list, c.err
}

func (r *mvsReqs) modFileToList(f *modfile.File) []module.Version {
	list := make([]module.Version, 0, len(f.Require))
	for _, r := range f.Require {
		list = append(list, r.Mod)
	}
	return list
}

// required returns a unique copy of the requirements of mod.
func (r *mvsReqs) required(mod module.Version) ([]module.Version, error) {
	if mod == Target {
		if modFile != nil && modFile.Go != nil {
			r.versions.LoadOrStore(mod, modFile.Go.Version)
		}
		return append([]module.Version(nil), r.buildList[1:]...), nil
	}

	if cfg.BuildMod == "vendor" {
		// For every module other than the target,
		// return the full list of modules from modules.txt.
		readVendorList()
		return append([]module.Version(nil), vendorList...), nil
	}

	origPath := mod.Path
	if repl := Replacement(mod); repl.Path != "" {
		if repl.Version == "" {
			// TODO: need to slip the new version into the tags list etc.
			dir := repl.Path
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(ModRoot(), dir)
			}
			gomod := filepath.Join(dir, "go.mod")
			data, err := lockedfile.Read(gomod)
			if err != nil {
				return nil, fmt.Errorf("parsing %s: %v", base.ShortPath(gomod), err)
			}
			f, err := modfile.ParseLax(gomod, data, nil)
			if err != nil {
				return nil, fmt.Errorf("parsing %s: %v", base.ShortPath(gomod), err)
			}
			if f.Go != nil {
				r.versions.LoadOrStore(mod, f.Go.Version)
			}
			return r.modFileToList(f), nil
		}
		mod = repl
	}

	if mod.Version == "none" {
		return nil, nil
	}

	if !semver.IsValid(mod.Version) {
		// Disallow the broader queries supported by fetch.Lookup.
		base.Fatalf("go: internal error: %s@%s: unexpected invalid semantic version", mod.Path, mod.Version)
	}

	data, err := modfetch.GoMod(mod.Path, mod.Version)
	if err != nil {
		return nil, err
	}
	f, err := modfile.ParseLax("go.mod", data, nil)
	if err != nil {
		return nil, module.VersionError(mod, fmt.Errorf("parsing go.mod: %v", err))
	}

	if f.Module == nil {
		return nil, module.VersionError(mod, errors.New("parsing go.mod: missing module line"))
	}
	if mpath := f.Module.Mod.Path; mpath != origPath && mpath != mod.Path {
		return nil, module.VersionError(mod, fmt.Errorf(`parsing go.mod:
	module declares its path as: %s
	        but was required as: %s`, mpath, origPath))
	}
	if f.Go != nil {
		r.versions.LoadOrStore(mod, f.Go.Version)
	}

	return r.modFileToList(f), nil
}

// Max returns the maximum of v1 and v2 according to semver.Compare.
//
// As a special case, the version "" is considered higher than all other
// versions. The main module (also known as the target) has no version and must
// be chosen over other versions of the same module in the module dependency
// graph.
func (*mvsReqs) Max(v1, v2 string) string {
	if v1 != "" && semver.Compare(v1, v2) == -1 {
		return v2
	}
	return v1
}

// Upgrade is a no-op, here to implement mvs.Reqs.
// The upgrade logic for go get -u is in ../modget/get.go.
func (*mvsReqs) Upgrade(m module.Version) (module.Version, error) {
	return m, nil
}

func versions(path string) ([]string, error) {
	// Note: modfetch.Lookup and repo.Versions are cached,
	// so there's no need for us to add extra caching here.
	var versions []string
	err := modfetch.TryProxies(func(proxy string) error {
		repo, err := modfetch.Lookup(proxy, path)
		if err == nil {
			versions, err = repo.Versions("")
		}
		return err
	})
	return versions, err
}

// Previous returns the tagged version of m.Path immediately prior to
// m.Version, or version "none" if no prior version is tagged.
func (*mvsReqs) Previous(m module.Version) (module.Version, error) {
	list, err := versions(m.Path)
	if err != nil {
		return module.Version{}, err
	}
	i := sort.Search(len(list), func(i int) bool { return semver.Compare(list[i], m.Version) >= 0 })
	if i > 0 {
		return module.Version{Path: m.Path, Version: list[i-1]}, nil
	}
	return module.Version{Path: m.Path, Version: "none"}, nil
}

// next returns the next version of m.Path after m.Version.
// It is only used by the exclusion processing in the Required method,
// not called directly by MVS.
func (*mvsReqs) next(m module.Version) (module.Version, error) {
	list, err := versions(m.Path)
	if err != nil {
		return module.Version{}, err
	}
	i := sort.Search(len(list), func(i int) bool { return semver.Compare(list[i], m.Version) > 0 })
	if i < len(list) {
		return module.Version{Path: m.Path, Version: list[i]}, nil
	}
	return module.Version{Path: m.Path, Version: "none"}, nil
}

// fetch downloads the given module (or its replacement)
// and returns its location.
//
// The isLocal return value reports whether the replacement,
// if any, is local to the filesystem.
func fetch(mod module.Version) (dir string, isLocal bool, err error) {
	if mod == Target {
		return ModRoot(), true, nil
	}
	if r := Replacement(mod); r.Path != "" {
		if r.Version == "" {
			dir = r.Path
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(ModRoot(), dir)
			}
			// Ensure that the replacement directory actually exists:
			// dirInModule does not report errors for missing modules,
			// so if we don't report the error now, later failures will be
			// very mysterious.
			if _, err := os.Stat(dir); err != nil {
				if os.IsNotExist(err) {
					// Semantically the module version itself “exists” — we just don't
					// have its source code. Remove the equivalence to os.ErrNotExist,
					// and make the message more concise while we're at it.
					err = fmt.Errorf("replacement directory %s does not exist", r.Path)
				} else {
					err = fmt.Errorf("replacement directory %s: %w", r.Path, err)
				}
				return dir, true, module.VersionError(mod, err)
			}
			return dir, true, nil
		}
		mod = r
	}

	dir, err = modfetch.Download(mod)
	return dir, false, err
}
