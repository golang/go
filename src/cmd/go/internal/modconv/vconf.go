// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"strings"

	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
)

func ParseVendorConf(file string, data []byte) (*modfile.File, error) {
	mf := new(modfile.File)
	for lineno, line := range strings.Split(string(data), "\n") {
		lineno++
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		f := strings.Fields(line)
		if len(f) >= 2 {
			v := module.Version{Path: f[0]}
			if len(f) >= 3 {
				vNew := module.Version{Path: repoPathToImportPath(f[2]), Version: f[1]}
				mf.Replace = append(mf.Replace, &modfile.Replace{Old: v, New: vNew})
				v.Version = "latest"
			} else {
				v.Version = f[1]
			}
			mf.Require = append(mf.Require, &modfile.Require{Mod: v})
		}
	}
	return mf, nil
}

func repoPathToImportPath(repo string) string {
	path := repo
	for _, prefix := range []string{"https://", "git://", "ssh://", "git+ssh://", "svn://", "svn+ssh://", "bzr://", "bzr+ssh://", "http://"} {
		if strings.HasPrefix(path, prefix) {
			path = strings.TrimPrefix(path, prefix)
			break
		}
	}
	return strings.TrimSuffix(path, ".git")
}
