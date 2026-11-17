// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/load"
	"cmd/go/internal/modinfo"
	"cmd/internal/logosowned"
)

func variantStatePath(out string) string {
	return out + ".state"
}

// variantPackage carries only source inventory from an actual helper load.
// Reconstructed packages are emission/provenance records, never build roots.
type variantPackage struct {
	ImportPath, Name, Dir, ForTest            string
	Standard, Goroot, TestBuilt               bool
	Module                                    *modinfo.ModulePublic
	GoFiles, CgoFiles, CompiledGoFiles        []string
	IgnoredGoFiles, TestGoFiles, XTestGoFiles []string
}

type variantState struct {
	Packages     []variantPackage
	EjectModules []string
}

func writeVariantState(plan *ejectPlan, staging, out string) {
	var state variantState
	for _, path := range slices.Sorted(maps.Keys(plan.provenance)) {
		for _, p := range plan.provenance[path] {
			if p.Internal.TestmainGo != nil {
				continue
			}
			state.Packages = append(state.Packages, variantPackage{
				ImportPath: p.ImportPath, Name: p.Name, Dir: p.Dir, ForTest: p.ForTest,
				Standard: p.Standard, Goroot: p.Goroot, TestBuilt: plan.testBuilt[p], Module: p.Module,
				GoFiles: p.GoFiles, CgoFiles: p.CgoFiles, CompiledGoFiles: p.CompiledGoFiles,
				IgnoredGoFiles: p.IgnoredGoFiles, TestGoFiles: p.TestGoFiles, XTestGoFiles: p.XTestGoFiles,
			})
		}
	}
	for _, m := range plan.deps {
		state.EjectModules = append(state.EjectModules, m.path)
	}
	dir := variantStatePath(out)
	if err := os.Mkdir(dir, 0777); err != nil {
		base.Fatalf("logos eject: make variant state: %v", err)
	}
	data, err := json.Marshal(state)
	if err != nil {
		base.Fatalf("logos eject: encode variant inventory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "packages.json"), data, 0666); err != nil {
		base.Fatalf("logos eject: write variant inventory: %v", err)
	}
	// Both directories belong to this helper's temporary output. Moving the
	// staging tree preserves owned-request sidecars and avoids another copy.
	if err := os.Rename(staging, filepath.Join(dir, "staging")); err != nil {
		base.Fatalf("logos eject: retain variant staging: %v", err)
	}
}

func absorbVariantState(plan *ejectPlan, out, staging string, ordinal int) {
	dir := variantStatePath(out)
	data, err := os.ReadFile(filepath.Join(dir, "packages.json"))
	if err != nil {
		base.Fatalf("logos eject: read variant inventory: %v", err)
	}
	var state variantState
	if err := json.Unmarshal(data, &state); err != nil {
		base.Fatalf("logos eject: decode variant inventory %s: %v", dir, err)
	}
	for _, inventory := range state.Packages {
		p := &load.Package{PackagePublic: load.PackagePublic{
			ImportPath: inventory.ImportPath, Name: inventory.Name, Dir: inventory.Dir, ForTest: inventory.ForTest,
			Standard: inventory.Standard, Goroot: inventory.Goroot, Module: inventory.Module,
			GoFiles: inventory.GoFiles, CgoFiles: inventory.CgoFiles, CompiledGoFiles: inventory.CompiledGoFiles,
			IgnoredGoFiles: inventory.IgnoredGoFiles, TestGoFiles: inventory.TestGoFiles, XTestGoFiles: inventory.XTestGoFiles,
		}}
		plan.provenance[p.ImportPath] = append(plan.provenance[p.ImportPath], p)
		plan.testBuilt[p] = inventory.TestBuilt
		plan.addPackage(p)
	}
	for _, path := range state.EjectModules {
		module := plan.depByPath[path]
		if module == nil {
			base.Fatalf("logos eject: variant inventory has no packages for module %s", path)
		}
		module.required = true
	}
	resortEjectPlan(plan)
	srcRoot := filepath.Join(dir, "staging")
	err = filepath.WalkDir(srcRoot, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("non-regular variant output %s", path)
		}
		rel, err := filepath.Rel(srcRoot, path)
		if err != nil {
			return err
		}
		dst := filepath.Join(staging, rel)
		if entry.Name() == logosowned.FileName {
			// Sidecars from each context stay separate until the existing
			// owned-request merger combines their declaration conditions.
			dst = filepath.Join(staging, "_variants", strconv.Itoa(ordinal), rel)
		} else if !strings.HasSuffix(entry.Name(), ".go") {
			return nil
		} else if _, err := os.Stat(dst); err == nil {
			return nil
		} else if !os.IsNotExist(err) {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
			return err
		}
		return os.WriteFile(dst, data, 0666)
	})
	if err != nil {
		base.Fatalf("logos eject: merge variant staging %s: %v", srcRoot, err)
	}
}
