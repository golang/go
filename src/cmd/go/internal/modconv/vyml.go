// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"strings"

	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
)

func ParseVendorYML(file string, data []byte) (*modfile.File, error) {
	mf := new(modfile.File)
	vendors := false
	path := ""
	for _, line := range strings.Split(string(data), "\n") {
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "vendors:") {
			vendors = true
		} else if line[0] != '-' && line[0] != ' ' && line[0] != '\t' {
			vendors = false
		}
		if !vendors {
			continue
		}
		if strings.HasPrefix(line, "- path:") {
			path = strings.TrimSpace(line[len("- path:"):])
		}
		if strings.HasPrefix(line, "  rev:") {
			rev := strings.TrimSpace(line[len("  rev:"):])
			if path != "" && rev != "" {
				mf.Require = append(mf.Require, &modfile.Require{Mod: module.Version{Path: path, Version: rev}})
			}
		}
	}
	return mf, nil
}
