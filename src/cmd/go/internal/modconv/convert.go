// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/modfetch"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// ConvertLegacyConfig converts legacy config to modfile.
// The file argument is slash-delimited.
func ConvertLegacyConfig(f *modfile.File, file string, data []byte) error {
	i := strings.LastIndex(file, "/")
	j := -2
	if i >= 0 {
		j = strings.LastIndex(file[:i], "/")
	}
	convert := Converters[file[i+1:]]
	if convert == nil && j != -2 {
		convert = Converters[file[j+1:]]
	}
	if convert == nil {
		return fmt.Errorf("unknown legacy config file %s", file)
	}
	mf, err := convert(file, data)
	if err != nil {
		return fmt.Errorf("parsing %s: %v", file, err)
	}

	// Convert requirements block, which may use raw SHA1 hashes as versions,
	// to valid semver requirement list, respecting major versions.
	versions := make([]module.Version, len(mf.Require))
	replace := make(map[string]*modfile.Replace)

	for _, r := range mf.Replace {
		replace[r.New.Path] = r
		replace[r.Old.Path] = r
	}

	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	for i, r := range mf.Require {
		m := r.Mod
		if m.Path == "" {
			continue
		}
		if re, ok := replace[m.Path]; ok {
			m = re.New
		}
		sem <- token{}
		go func(i int, m module.Version) {
			defer func() { <-sem }()
			repo, info, err := modfetch.ImportRepoRev(m.Path, m.Version)
			if err != nil {
				fmt.Fprintf(os.Stderr, "go: converting %s: stat %s@%s: %v\n", base.ShortPath(file), m.Path, m.Version, err)
				return
			}

			path := repo.ModulePath()
			versions[i].Path = path
			versions[i].Version = info.Version
		}(i, m)
	}
	// Fill semaphore channel to wait for all tasks to finish.
	for n := cap(sem); n > 0; n-- {
		sem <- token{}
	}

	need := map[string]string{}
	for _, v := range versions {
		if v.Path == "" {
			continue
		}
		// Don't use semver.Max here; need to preserve +incompatible suffix.
		if needv, ok := need[v.Path]; !ok || semver.Compare(needv, v.Version) < 0 {
			need[v.Path] = v.Version
		}
	}
	paths := make([]string, 0, len(need))
	for path := range need {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		if re, ok := replace[path]; ok {
			err := f.AddReplace(re.Old.Path, re.Old.Version, path, need[path])
			if err != nil {
				return fmt.Errorf("add replace: %v", err)
			}
		}
		f.AddNewRequire(path, need[path], false)
	}

	f.Cleanup()
	return nil
}
