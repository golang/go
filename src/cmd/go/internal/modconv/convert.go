// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
	"cmd/go/internal/par"
	"cmd/go/internal/semver"
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
	var (
		work    par.Work
		mu      sync.Mutex
		need    = make(map[string]string)
		replace = make(map[string]*modfile.Replace)
	)

	for _, r := range mf.Replace {
		replace[r.New.Path] = r
		replace[r.Old.Path] = r
	}
	for _, r := range mf.Require {
		m := r.Mod
		if m.Path == "" {
			continue
		}
		if re, ok := replace[m.Path]; ok {
			work.Add(re.New)
			continue
		}
		work.Add(r.Mod)
	}

	work.Do(10, func(item interface{}) {
		r := item.(module.Version)
		repo, info, err := modfetch.ImportRepoRev(r.Path, r.Version)
		if err != nil {
			fmt.Fprintf(os.Stderr, "go: converting %s: stat %s@%s: %v\n", base.ShortPath(file), r.Path, r.Version, err)
			return
		}
		mu.Lock()
		path := repo.ModulePath()
		// Don't use semver.Max here; need to preserve +incompatible suffix.
		if v, ok := need[path]; !ok || semver.Compare(v, info.Version) < 0 {
			need[path] = info.Version
		}
		mu.Unlock()
	})

	var paths []string
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
