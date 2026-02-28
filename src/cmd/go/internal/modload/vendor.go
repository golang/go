// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/gover"

	"golang.org/x/mod/modfile"
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
	GoVersion   string
}

// readVendorList reads the list of vendored modules from vendor/modules.txt.
func readVendorList(vendorDir string) {
	vendorOnce.Do(func() {
		vendorList = nil
		vendorPkgModule = make(map[string]module.Version)
		vendorVersion = make(map[string]string)
		vendorMeta = make(map[module.Version]vendorMetadata)
		vendorFile := filepath.Join(vendorDir, "modules.txt")
		data, err := os.ReadFile(vendorFile)
		if err != nil {
			if !errors.Is(err, fs.ErrNotExist) {
				base.Fatalf("go: %s", err)
			}
			return
		}

		var mod module.Version
		for line := range strings.SplitSeq(string(data), "\n") {
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

			if annotations, ok := strings.CutPrefix(line, "## "); ok {
				// Metadata. Take the union of annotations across multiple lines, if present.
				meta := vendorMeta[mod]
				for entry := range strings.SplitSeq(annotations, ";") {
					entry = strings.TrimSpace(entry)
					if entry == "explicit" {
						meta.Explicit = true
					}
					if goVersion, ok := strings.CutPrefix(entry, "go "); ok {
						meta.GoVersion = goVersion
						rawGoVersion.Store(mod, meta.GoVersion)
						if gover.Compare(goVersion, gover.Local()) > 0 {
							base.Fatal(&gover.TooNewError{What: mod.Path + " in " + base.ShortPath(vendorFile), GoVersion: goVersion})
						}
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
				if v, ok := vendorVersion[mod.Path]; !ok || gover.ModCompare(mod.Path, v, mod.Version) < 0 {
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
func checkVendorConsistency(loaderstate *State, indexes []*modFileIndex, modFiles []*modfile.File, modRoots []string) {
	// readVendorList only needs the main module to get the directory
	// the vendor directory is in.
	readVendorList(VendorDir(loaderstate))

	if len(modFiles) < 1 {
		// We should never get here if there are zero modfiles. Either
		// we're in single module mode and there's a single module, or
		// we're in workspace mode, and we fail earlier reporting that
		// "no modules were found in the current workspace".
		panic("checkVendorConsistency called with zero modfiles")
	}

	pre114 := false
	if !loaderstate.inWorkspaceMode() { // workspace mode was added after Go 1.14
		if len(indexes) != 1 {
			panic(fmt.Errorf("not in workspace mode but number of indexes is %v, not 1", len(indexes)))
		}
		index := indexes[0]
		if gover.Compare(index.goVersion, "1.14") < 0 {
			// Go versions before 1.14 did not include enough information in
			// vendor/modules.txt to check for consistency.
			// If we know that we're on an earlier version, relax the consistency check.
			pre114 = true
		}
	}

	vendErrors := new(strings.Builder)
	vendErrorf := func(mod module.Version, format string, args ...any) {
		detail := fmt.Sprintf(format, args...)
		if mod.Version == "" {
			fmt.Fprintf(vendErrors, "\n\t%s: %s", mod.Path, detail)
		} else {
			fmt.Fprintf(vendErrors, "\n\t%s@%s: %s", mod.Path, mod.Version, detail)
		}
	}

	// Iterate over the Require directives in their original (not indexed) order
	// so that the errors match the original file.
	for _, modFile := range modFiles {
		for _, r := range modFile.Require {
			if !vendorMeta[r.Mod].Explicit {
				if pre114 {
					// Before 1.14, modules.txt did not indicate whether modules were listed
					// explicitly in the main module's go.mod file.
					// However, we can at least detect a version mismatch if packages were
					// vendored from a non-matching version.
					if vv, ok := vendorVersion[r.Mod.Path]; ok && vv != r.Mod.Version {
						vendErrorf(r.Mod, "is explicitly required in go.mod, but vendor/modules.txt indicates %s@%s", r.Mod.Path, vv)
					}
				} else {
					vendErrorf(r.Mod, "is explicitly required in go.mod, but not marked as explicit in vendor/modules.txt")
				}
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
	seenrep := make(map[module.Version]bool)
	checkReplace := func(replaces []*modfile.Replace) {
		for _, r := range replaces {
			if seenrep[r.Old] {
				continue // Don't print the same error more than once
			}
			seenrep[r.Old] = true
			rNew, modRoot, replacementSource := replacementFrom(loaderstate, r.Old)
			rNewCanonical := canonicalizeReplacePath(loaderstate, rNew, modRoot)
			vr := vendorMeta[r.Old].Replacement
			if vr == (module.Version{}) {
				if rNewCanonical == (module.Version{}) {
					// r.Old is not actually replaced. It might be a main module.
					// Don't return an error.
				} else if pre114 && (r.Old.Version == "" || vendorVersion[r.Old.Path] != r.Old.Version) {
					// Before 1.14, modules.txt omitted wildcard replacements and
					// replacements for modules that did not have any packages to vendor.
				} else {
					vendErrorf(r.Old, "is replaced in %s, but not marked as replaced in vendor/modules.txt", base.ShortPath(replacementSource))
				}
			} else if vr != rNewCanonical {
				vendErrorf(r.Old, "is replaced by %s in %s, but marked as replaced by %s in vendor/modules.txt", describe(rNew), base.ShortPath(replacementSource), describe(vr))
			}
		}
	}
	for _, modFile := range modFiles {
		checkReplace(modFile.Replace)
	}
	if loaderstate.MainModules.workFile != nil {
		checkReplace(loaderstate.MainModules.workFile.Replace)
	}

	for _, mod := range vendorList {
		meta := vendorMeta[mod]
		if meta.Explicit {
			// in workspace mode, check that it's required by at least one of the main modules
			var foundRequire bool
			for _, index := range indexes {
				if _, inGoMod := index.require[mod]; inGoMod {
					foundRequire = true
				}
			}
			if !foundRequire {
				article := ""
				if loaderstate.inWorkspaceMode() {
					article = "a "
				}
				vendErrorf(mod, "is marked as explicit in vendor/modules.txt, but not explicitly required in %vgo.mod", article)
			}

		}
	}

	for _, mod := range vendorReplaced {
		r := Replacement(loaderstate, mod)
		replacementSource := "go.mod"
		if loaderstate.inWorkspaceMode() {
			replacementSource = "the workspace"
		}
		if r == (module.Version{}) {
			vendErrorf(mod, "is marked as replaced in vendor/modules.txt, but not replaced in %s", replacementSource)
			continue
		}
		// If both replacements exist, we've already reported that they're different above.
	}

	if vendErrors.Len() > 0 {
		subcmd := "mod"
		if loaderstate.inWorkspaceMode() {
			subcmd = "work"
		}
		base.Fatalf("go: inconsistent vendoring in %s:%s\n\n\tTo ignore the vendor directory, use -mod=readonly or -mod=mod.\n\tTo sync the vendor directory, run:\n\t\tgo %s vendor", filepath.Dir(VendorDir(loaderstate)), vendErrors, subcmd)
	}
}
