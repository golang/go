// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
	"cmd/internal/objabi"
)

// LogosBuildGC returns the [build] gc key of the logos.toml of the main
// module containing the current directory, and whether the key is set. A file
// that does not parse reads as unset: checkLogosConfig reports its error on
// the package.
func LogosBuildGC(ld *modload.Loader) (mode string, set bool) {
	cf, err := logosConfig(ld)
	if err != nil || cf == nil {
		return "", false
	}
	return cf.Build.GC, cf.Build.GCSet
}

// logosGCGcflags returns the compile flags the collector mode adds to p.
// Batch and epsilon compile every package without write barriers except the
// runtime and the packages it imports (the objabi runtime list), which keep
// them. The mode is sound only because marking never runs concurrently with
// mutators: batch forces every collection to stop the world, and epsilon
// starts none on its own. The flag enters the compile action ID with the
// package's other gcflags.
func logosGCGcflags(p *Package) []string {
	switch cfg.LogosGC {
	case "batch", "epsilon":
		if !p.Standard || !objabi.LookupPkgSpecial(p.ImportPath).Runtime {
			return []string{"-wb=false"}
		}
	}
	return nil
}
