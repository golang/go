// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"cmd/go/internal/fsys"
)

func ScanDir(dir string, tags map[string]bool) ([]string, []string, error) {
	infos, err := fsys.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	var files []string
	for _, info := range infos {
		name := info.Name()

		// If the directory entry is a symlink, stat it to obtain the info for the
		// link target instead of the link itself.
		if info.Mode()&fs.ModeSymlink != 0 {
			info, err = fsys.Stat(filepath.Join(dir, name))
			if err != nil {
				continue // Ignore broken symlinks.
			}
		}

		if info.Mode().IsRegular() && !strings.HasPrefix(name, "_") && !strings.HasPrefix(name, ".") && strings.HasSuffix(name, ".go") && MatchFile(name, tags) {
			files = append(files, filepath.Join(dir, name))
		}
	}
	return scanFiles(files, tags, false)
}

func ScanFiles(files []string, tags map[string]bool) ([]string, []string, error) {
	return scanFiles(files, tags, true)
}

func scanFiles(files []string, tags map[string]bool, explicitFiles bool) ([]string, []string, error) {
	imports := make(map[string]bool)
	testImports := make(map[string]bool)
	numFiles := 0
Files:
	for _, name := range files {
		r, err := fsys.Open(name)
		if err != nil {
			return nil, nil, err
		}
		var list []string
		data, err := ReadImports(r, false, &list)
		r.Close()
		if err != nil {
			return nil, nil, fmt.Errorf("reading %s: %v", name, err)
		}

		// import "C" is implicit requirement of cgo tag.
		// When listing files on the command line (explicitFiles=true)
		// we do not apply build tag filtering but we still do apply
		// cgo filtering, so no explicitFiles check here.
		// Why? Because we always have, and it's not worth breaking
		// that behavior now.
		for _, path := range list {
			if path == `"C"` && !tags["cgo"] && !tags["*"] {
				continue Files
			}
		}

		if !explicitFiles && !ShouldBuild(data, tags) {
			continue
		}
		numFiles++
		m := imports
		if strings.HasSuffix(name, "_test.go") {
			m = testImports
		}
		for _, p := range list {
			q, err := strconv.Unquote(p)
			if err != nil {
				continue
			}
			m[q] = true
		}
	}
	if numFiles == 0 {
		return nil, nil, ErrNoGo
	}
	return keys(imports), keys(testImports), nil
}

var ErrNoGo = fmt.Errorf("no Go source files")

func keys(m map[string]bool) []string {
	list := make([]string, 0, len(m))
	for k := range m {
		list = append(list, k)
	}
	sort.Strings(list)
	return list
}
