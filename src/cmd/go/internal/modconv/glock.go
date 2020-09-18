// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
)

func ParseGLOCKFILE(file string, data []byte) (*modfile.File, error) {
	mf := new(modfile.File)
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) >= 2 && f[0] != "cmd" {
			mf.Require = append(mf.Require, &modfile.Require{Mod: module.Version{Path: f[0], Version: f[1]}})
		}
	}
	return mf, nil
}
