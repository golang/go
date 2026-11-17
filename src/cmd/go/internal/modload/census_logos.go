// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"path/filepath"

	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"

	"golang.org/x/mod/module"
)

// LogosModulePackageDirs finds the source directories owned by m for the
// module-kind census. The caller classifies their active files. Module
// ownership must be resolved here: vendor trees flatten module roots, and
// workspace module paths can overlap without sharing source directories.
func LogosModulePackageDirs(ctx context.Context, ld *Loader, m module.Version) (map[string]string, error) {
	prefix := m.Path
	if ld.MainModules.Contains(m.Path) {
		prefix = ld.MainModules.PathPrefix(m)
	}
	match := MatchInModule(ld, ctx, prefix+"/...", m, imports.Tags())
	if len(match.Errs) != 0 {
		return nil, errors.Join(match.Errs...)
	}

	dirs := make(map[string]string)
	if cfg.BuildMod == "vendor" && !ld.MainModules.Contains(m.Path) {
		root := VendorDir(ld)
		readVendorList(root)
		for _, name := range match.Pkgs {
			if vendorPkgModule[name] == m {
				dirs[name] = filepath.Join(root, filepath.FromSlash(name))
			}
		}
		return dirs, nil
	}

	root, isLocal, err := fetch(ld, ctx, m)
	if err != nil {
		return nil, err
	}
	for _, name := range match.Pkgs {
		dir, haveFiles, err := dirInModule(name, prefix, root, isLocal)
		if err != nil {
			return nil, err
		}
		if haveFiles {
			dirs[name] = dir
		}
	}
	return dirs, nil
}
