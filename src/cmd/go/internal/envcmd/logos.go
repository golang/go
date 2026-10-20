// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package envcmd

import (
	"cmd/go/internal/cfg"
)

// refreshLogosPolicy recomputes the global cgo switch and the arena
// experiment from the module above the working directory after the -C
// chdir and before MkEnv bakes CGO_ENABLED and GOEXPERIMENT into the
// process environment. Version output never depends on the policy, so it
// skips the filesystem work. Workspace members beyond the current
// directory resolve later in work BuildInit; per-package checks read the
// startup CGO_ENABLED snapshot, so the single effective value baked here
// cannot skew their precedence.
func refreshLogosPolicy() {
	if !cfg.Logos || cfg.CmdName == "version" {
		return
	}
	cfg.RefreshLogosCgoPolicy(nil)
	cfg.RefreshLogosArena(nil)
}
