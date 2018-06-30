// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"cmd/go/internal/modconv"
	"cmd/go/internal/modfile"
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
	convert := modconv.Converters[file[i+1:]]
	if convert == nil && j != -2 {
		convert = modconv.Converters[file[j+1:]]
	}
	if convert == nil {
		return fmt.Errorf("unknown legacy config file %s", file)
	}
	require, err := convert(file, data)
	if err != nil {
		return fmt.Errorf("parsing %s: %v", file, err)
	}

	// Convert requirements block, which may use raw SHA1 hashes as versions,
	// to valid semver requirement list, respecting major versions.
	need := make(map[string]string)
	for _, r := range require {
		if r.Path == "" {
			continue
		}

		// TODO: Something better here.
		if strings.HasPrefix(r.Path, "github.com/") || strings.HasPrefix(r.Path, "golang.org/x/") {
			f := strings.Split(r.Path, "/")
			if len(f) > 3 {
				r.Path = strings.Join(f[:3], "/")
			}
		}

		repo, err := Lookup(r.Path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "vgo: lookup %s: %v\n", r.Path, err)
			continue
		}
		info, err := repo.Stat(r.Version)
		if err != nil {
			fmt.Fprintf(os.Stderr, "vgo: stat %s@%s: %v\n", r.Path, r.Version, err)
			continue
		}
		path := repo.ModulePath()
		need[path] = semver.Max(need[path], info.Version)
	}

	var paths []string
	for path := range need {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		f.AddRequire(path, need[path])
	}

	return nil
}
