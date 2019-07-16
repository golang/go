// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"strings"

	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
)

func ParseGlideLock(file string, data []byte) (*modfile.File, error) {
	mf := new(modfile.File)
	imports := false
	name := ""
	for _, line := range strings.Split(string(data), "\n") {
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "imports:") {
			imports = true
		} else if line[0] != '-' && line[0] != ' ' && line[0] != '\t' {
			imports = false
		}
		if !imports {
			continue
		}
		if strings.HasPrefix(line, "- name:") {
			name = strings.TrimSpace(line[len("- name:"):])
		}
		if strings.HasPrefix(line, "  version:") {
			version := strings.TrimSpace(line[len("  version:"):])
			if name != "" && version != "" {
				mf.Require = append(mf.Require, &modfile.Require{Mod: module.Version{Path: name, Version: version}})
			}
		}
	}
	return mf, nil
}
