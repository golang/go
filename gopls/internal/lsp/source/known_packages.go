// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/parser"
	"go/token"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
)

// KnownPackagePaths returns a new list of package paths of all known
// packages in the package graph that could potentially be imported by
// the given file. The list is ordered lexicographically, except that
// all dot-free paths (standard packages) appear before dotful ones.
//
// It is part of the gopls.list_known_packages command.
func KnownPackagePaths(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]PackagePath, error) {
	// This algorithm is expressed in terms of Metadata, not Packages,
	// so it doesn't cause or wait for type checking.

	current, err := NarrowestMetadataForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, err // e.g. context cancelled
	}

	// Parse the file's imports so we can compute which
	// PackagePaths are imported by this specific file.
	src, err := fh.Content()
	if err != nil {
		return nil, err
	}
	file, err := parser.ParseFile(token.NewFileSet(), fh.URI().Filename(), src, parser.ImportsOnly)
	if err != nil {
		return nil, err
	}
	imported := make(map[PackagePath]bool)
	for _, imp := range file.Imports {
		if id := current.DepsByImpPath[UnquoteImportPath(imp)]; id != "" {
			if m := snapshot.Metadata(id); m != nil {
				imported[m.PkgPath] = true
			}
		}
	}

	// Now find candidates among all known packages.
	knownPkgs, err := snapshot.AllMetadata(ctx)
	if err != nil {
		return nil, err
	}
	seen := make(map[PackagePath]bool)
	for _, knownPkg := range knownPkgs {
		// package main cannot be imported
		if knownPkg.Name == "main" {
			continue
		}
		// test packages cannot be imported
		if knownPkg.ForTest != "" {
			continue
		}
		// No need to import what the file already imports.
		// This check is based on PackagePath, not PackageID,
		// so that all test variants are filtered out too.
		if imported[knownPkg.PkgPath] {
			continue
		}
		// make sure internal packages are importable by the file
		if !IsValidImport(current.PkgPath, knownPkg.PkgPath) {
			continue
		}
		// naive check on cyclical imports
		if isDirectlyCyclical(current, knownPkg) {
			continue
		}
		// AllMetadata may have multiple variants of a pkg.
		seen[knownPkg.PkgPath] = true
	}

	// Augment the set by invoking the goimports algorithm.
	if err := snapshot.RunProcessEnvFunc(ctx, func(ctx context.Context, o *imports.Options) error {
		ctx, cancel := context.WithTimeout(ctx, time.Millisecond*80)
		defer cancel()
		var seenMu sync.Mutex
		wrapped := func(ifix imports.ImportFix) {
			seenMu.Lock()
			defer seenMu.Unlock()
			// TODO(adonovan): what if the actual package path has a vendor/ prefix?
			seen[PackagePath(ifix.StmtInfo.ImportPath)] = true
		}
		return imports.GetAllCandidates(ctx, wrapped, "", fh.URI().Filename(), string(current.Name), o.Env)
	}); err != nil {
		// If goimports failed, proceed with just the candidates from the metadata.
		event.Error(ctx, "imports.GetAllCandidates", err)
	}

	// Sort lexicographically, but with std before non-std packages.
	paths := make([]PackagePath, 0, len(seen))
	for path := range seen {
		paths = append(paths, path)
	}
	sort.Slice(paths, func(i, j int) bool {
		importI, importJ := paths[i], paths[j]
		iHasDot := strings.Contains(string(importI), ".")
		jHasDot := strings.Contains(string(importJ), ".")
		if iHasDot != jHasDot {
			return jHasDot // dot-free paths (standard packages) compare less
		}
		return importI < importJ
	})

	return paths, nil
}

// isDirectlyCyclical checks if imported directly imports pkg.
// It does not (yet) offer a full cyclical check because showing a user
// a list of importable packages already generates a very large list
// and having a few false positives in there could be worth the
// performance snappiness.
//
// TODO(adonovan): ensure that metadata graph is always cyclic!
// Many algorithms will get confused or even stuck in the
// presence of cycles. Then replace this function by 'false'.
func isDirectlyCyclical(pkg, imported *Metadata) bool {
	_, ok := imported.DepsByPkgPath[pkg.PkgPath]
	return ok
}
