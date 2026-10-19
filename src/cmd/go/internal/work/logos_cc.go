// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
)

// refreshLogosCgoPolicy recomputes the global cgo switch from every main
// module root once flags are parsed and modload has loaded the workspace
// member list. With no main modules cfg falls back to the module above the
// working directory. It runs in BuildInit, before package loading decides
// cgo file visibility, so -C targets and mixed-policy workspaces apply the
// owning modules' policies instead of the startup directory's.
func refreshLogosCgoPolicy(ld *modload.Loader) {
	var roots []string
	if ld != nil && ld.MainModules != nil {
		for _, m := range ld.MainModules.Versions() {
			if root := ld.MainModules.ModRoot(m); root != "" {
				roots = append(roots, root)
			}
		}
	}
	cfg.RefreshLogosCgoPolicy(roots)
}

// logosCCFallback resolves the C compiler for a cgo build through the cfg
// zig fallback, which honors the startup CC snapshot. Plain envList callers
// keep their behavior for every other key.
func logosCCFallback(def string) string {
	return cfg.ZigCC(def)
}
