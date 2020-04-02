// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"cmd/go/internal/base"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var (
	vendorOnce      sync.Once
	vendorList      []module.Version          // modules that contribute packages to the build, in order of appearance
	vendorReplaced  []module.Version          // all replaced modules; may or may not also contribute packages
	vendorVersion   map[string]string         // module path → selected version (if known)
	vendorPkgModule map[string]module.Version // package → containing module
	vendorMeta      map[module.Version]vendorMetadata
)

type vendorMetadata struct {
	Explicit    bool
	Replacement module.Version
}

// readVendorList reads the list of vendored modules from vendor/modules.txt.
func readVendorList() {
	vendorOnce.Do(func() {
		vendorList = nil
		vendorPkgModule = make(map[string]module.Version)
		vendorVersion = make(map[string]string)
		vendorMeta = make(map[module.Version]vendorMetadata)
		data, err := ioutil.ReadFile(filepath.Join(ModRoot(), "vendor/modules.txt"))
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				base.Fatalf("go: %s", err)
			}
			return
		}

		var mod module.Version
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "# ") {
				f := strings.Fields(line)

				if len(f) < 3 {
					continue
				}
				if semver.IsValid(f[2]) {
					// A module, but we don't yet know whether it is in the build list or
					// only included to indicate a replacement.
					mod = module.Version{Path: f[1], Version: f[2]}
					f = f[3:]
				} else if f[2] == "=>" {
					// A wildcard replacement found in the main module's go.mod file.
					mod = module.Version{Path: f[1]}
					f = f[2:]
				} else {
					// Not a version or a wildcard replacement.
					// We don't know how to interpret this module line, so ignore it.
					mod = module.Version{}
					continue
				}

				if len(f) >= 2 && f[0] == "=>" {
					meta := vendorMeta[mod]
					if len(f) == 2 {
						// File replacement.
						meta.Replacement = module.Version{Path: f[1]}
						vendorReplaced = append(vendorReplaced, mod)
					} else if len(f) == 3 && semver.IsValid(f[2]) {
						// Path and version replacement.
						meta.Replacement = module.Version{Path: f[1], Version: f[2]}
						vendorReplaced = append(vendorReplaced, mod)
					} else {
						// We don't understand this replacement. Ignore it.
					}
					vendorMeta[mod] = meta
				}
				continue
			}

			// Not a module line. Must be a package within a module or a metadata
			// directive, either of which requires a preceding module line.
			if mod.Path == "" {
				continue
			}

			if strings.HasPrefix(line, "## ") {
				// Metadata. Take the union of annotations across multiple lines, if present.
				meta := vendorMeta[mod]
				for _, entry := range strings.Split(strings.TrimPrefix(line, "## "), ";") {
					entry = strings.TrimSpace(entry)
					if entry == "explicit" {
						meta.Explicit = true
					}
					// All other tokens are reserved for future use.
				}
				vendorMeta[mod] = meta
				continue
			}

			if f := strings.Fields(line); len(f) == 1 && module.CheckImportPath(f[0]) == nil {
				// A package within the current module.
				vendorPkgModule[f[0]] = mod

				// Since this module provides a package for the build, we know that it
				// is in the build list and is the selected version of its path.
				// If this information is new, record it.
				if v, ok := vendorVersion[mod.Path]; !ok || semver.Compare(v, mod.Version) < 0 {
					vendorList = append(vendorList, mod)
					vendorVersion[mod.Path] = mod.Version
				}
			}
		}
	})
}

// checkVendorConsistency verifies that the vendor/modules.txt file matches (if
// go 1.14) or at least does not contradict (go 1.13 or earlier) the
// requirements and replacements listed in the main module's go.mod file.
func checkVendorConsistency() {
	readVendorList()

	pre114 := false
	if modFile.Go == nil || semver.Compare("v"+modFile.Go.Version, "v1.14") < 0 {
		// Go versions before 1.14 did not include enough information in
		// vendor/modules.txt to check for consistency.
		// If we know that we're on an earlier version, relax the consistency check.
		pre114 = true
	}

	vendErrors := new(strings.Builder)
	vendErrorf := func(mod module.Version, format string, args ...interface{}) {
		detail := fmt.Sprintf(format, args...)
		if mod.Version == "" {
			fmt.Fprintf(vendErrors, "\n\t%s: %s", mod.Path, detail)
		} else {
			fmt.Fprintf(vendErrors, "\n\t%s@%s: %s", mod.Path, mod.Version, detail)
		}
	}

	for _, r := range modFile.Require {
		if !vendorMeta[r.Mod].Explicit {
			if pre114 {
				// Before 1.14, modules.txt did not indicate whether modules were listed
				// explicitly in the main module's go.mod file.
				// However, we can at least detect a version mismatch if packages were
				// vendored from a non-matching version.
				if vv, ok := vendorVersion[r.Mod.Path]; ok && vv != r.Mod.Version {
					vendErrorf(r.Mod, fmt.Sprintf("is explicitly required in go.mod, but vendor/modules.txt indicates %s@%s", r.Mod.Path, vv))
				}
			} else {
				vendErrorf(r.Mod, "is explicitly required in go.mod, but not marked as explicit in vendor/modules.txt")
			}
		}
	}

	describe := func(m module.Version) string {
		if m.Version == "" {
			return m.Path
		}
		return m.Path + "@" + m.Version
	}

	// We need to verify *all* replacements that occur in modfile: even if they
	// don't directly apply to any module in the vendor list, the replacement
	// go.mod file can affect the selected versions of other (transitive)
	// dependencies
	for _, r := range modFile.Replace {
		vr := vendorMeta[r.Old].Replacement
		if vr == (module.Version{}) {
			if pre114 && (r.Old.Version == "" || vendorVersion[r.Old.Path] != r.Old.Version) {
				// Before 1.14, modules.txt omitted wildcard replacements and
				// replacements for modules that did not have any packages to vendor.
			} else {
				vendErrorf(r.Old, "is replaced in go.mod, but not marked as replaced in vendor/modules.txt")
			}
		} else if vr != r.New {
			vendErrorf(r.Old, "is replaced by %s in go.mod, but marked as replaced by %s in vendor/modules.txt", describe(r.New), describe(vr))
		}
	}

	for _, mod := range vendorList {
		meta := vendorMeta[mod]
		if meta.Explicit {
			if _, inGoMod := index.require[mod]; !inGoMod {
				vendErrorf(mod, "is marked as explicit in vendor/modules.txt, but not explicitly required in go.mod")
			}
		}
	}

	for _, mod := range vendorReplaced {
		r := Replacement(mod)
		if r == (module.Version{}) {
			vendErrorf(mod, "is marked as replaced in vendor/modules.txt, but not replaced in go.mod")
			continue
		}
		if meta := vendorMeta[mod]; r != meta.Replacement {
			vendErrorf(mod, "is marked as replaced by %s in vendor/modules.txt, but replaced by %s in go.mod", describe(meta.Replacement), describe(r))
		}
	}

	if vendErrors.Len() > 0 {
		base.Fatalf("go: inconsistent vendoring in %s:%s\n\nrun 'go mod vendor' to sync, or use -mod=mod or -mod=readonly to ignore the vendor directory", modRoot, vendErrors)
	}
}
