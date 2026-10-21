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

// The loGo/s compiler writes code that names packages its root does not
// import: a copy of an imported comptime generic, a lowered const for, and
// a spelled type each qualify the packages their source mentions. Those
// packages sit behind the root's imports, possibly behind plain Go ones.
// Which of them a lowering reaches depends on type information, so the go
// command cannot predict it from the archives. The root's import config and
// content ID take every transitive built package instead.

// logosBuiltPackage reports whether a1 builds a package that an import
// config can name.
func logosBuiltPackage(a1 *Action) bool {
	p1 := a1.Package
	return p1 != nil && p1.ImportPath != "" && a1.built != ""
}

// logosSynthImports returns the built packages that the compiler may import
// for the code it synthesizes in the loGo/s compile root a: every built
// package transitively reachable through a.Deps, including through actions
// that carry no package, except the packages a imports directly, which the
// ordinary import config already names. The result is sorted by import path
// and names each package once. A plain Go root and a dry run (-n) get nil.
//
// Every consumer of the set calls this function, so the build import config,
// the lint import config, and the content ID cannot disagree. A change in
// any transitive content ID changes the root's action ID, which can
// recompile a loGo/s root that an exact dependency interface would not.
func logosSynthImports(a *Action) []*Action {
	if a.Package == nil || !a.Package.IsLogos() || cfg.BuildN {
		return nil
	}
	have := make(map[string]bool)
	for _, a1 := range a.Deps {
		if logosBuiltPackage(a1) {
			have[a1.Package.ImportPath] = true
		}
	}
	var extra []*Action
	seen := map[*Action]bool{a: true}
	var walk func(a1 *Action)
	walk = func(a1 *Action) {
		if seen[a1] {
			return
		}
		seen[a1] = true
		if logosBuiltPackage(a1) && !have[a1.Package.ImportPath] {
			have[a1.Package.ImportPath] = true
			extra = append(extra, a1)
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

// logosSynthImportCfg writes the import config lines for the synthesized
// imports of a.
func logosSynthImportCfg(w io.Writer, a *Action) {
	for _, a1 := range logosSynthImports(a) {
		fmt.Fprintf(w, "packagefile %s=%s\n", a1.Package.ImportPath, a1.built)
	}
}

// logosSynthActionID writes the content IDs of the synthesized imports of a
// into the action ID hash h.
func logosSynthActionID(h io.Writer, a *Action) {
	for _, a1 := range logosSynthImports(a) {
		fmt.Fprintf(h, "synthimport %s %s\n", a1.Package.ImportPath, contentID(a1.buildID))
	}
}
