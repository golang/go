// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
)

func logosBuiltPackage(a1 *Action) bool {
	if a1 == nil || a1.Package == nil || a1.Package.ImportPath == "" {
		return false
	}
	if a1.built == "" || strings.HasPrefix(a1.built, "DO NOT USE") {
		return false
	}
	return true
}

func logosSynthImports(a *Action) []*Action {
	if a.Package == nil || cfg.BuildN {
		return nil
	}
	all := a.Package.IsLogos()
	byPath := make(map[string]*Action)
	seenIdx := map[*Action]bool{a: true}
	var index func(a1 *Action)
	index = func(a1 *Action) {
		if seenIdx[a1] {
			return
		}
		seenIdx[a1] = true
		if logosBuiltPackage(a1) {
			if _, ok := byPath[a1.Package.ImportPath]; !ok {
				byPath[a1.Package.ImportPath] = a1
			}
		}
		for _, a2 := range a1.Deps {
			index(a2)
		}
	}
	for _, a1 := range a.Deps {
		index(a1)
	}
	have := make(map[string]bool)
	have[a.Package.ImportPath] = true
	for _, a1 := range a.Deps {
		if logosBuiltPackage(a1) {
			have[a1.Package.ImportPath] = true
		}
	}
	var extra []*Action
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 == nil || p1.ImportPath == "" || p1.Standard {
			continue
		}
		if have[p1.ImportPath] {
			continue
		}
		real, ok := byPath[p1.ImportPath]
		if !ok || real.Package == nil || real.Package.Standard {
			continue
		}
		have[p1.ImportPath] = true
		extra = append(extra, real)
	}
	seen := map[*Action]bool{a: true}
	var walk func(a1 *Action)
	walk = func(a1 *Action) {
		if seen[a1] {
			return
		}
		seen[a1] = true
		if p1 := a1.Package; p1 != nil && p1.ImportPath != "" && !have[p1.ImportPath] && (all || p1.IsLogos()) {
			if real, ok := byPath[p1.ImportPath]; ok && real.Package != nil && (all || real.Package.IsLogos()) {
				have[p1.ImportPath] = true
				extra = append(extra, real)
			}
		}
		for _, a2 := range a1.Deps {
			walk(a2)
		}
	}
	for _, a1 := range a.Deps {
		walk(a1)
	}
	slices.SortFunc(extra, func(x, y *Action) int {
		return strings.Compare(x.Package.ImportPath, y.Package.ImportPath)
	})
	return extra
}

func logosSynthImportCfg(w io.Writer, a *Action) {
	for _, a1 := range logosSynthImports(a) {
		fmt.Fprintf(w, "packagefile %s=%s\n", a1.Package.ImportPath, a1.built)
	}
}

func logosSynthActionID(h io.Writer, a *Action) {
	for _, a1 := range logosSynthImports(a) {
		fmt.Fprintf(h, "synthimport %s %s\n", a1.Package.ImportPath, contentID(a1.buildID))
	}
}
