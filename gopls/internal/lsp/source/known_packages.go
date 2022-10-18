// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
)

// KnownPackages returns a list of all known packages
// in the package graph that could potentially be imported
// by the given file.
func KnownPackages(ctx context.Context, snapshot Snapshot, fh VersionedFileHandle) ([]string, error) {
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, fmt.Errorf("GetParsedFile: %w", err)
	}
	alreadyImported := map[string]struct{}{}
	for _, imp := range pgf.File.Imports {
		alreadyImported[imp.Path.Value] = struct{}{}
	}
	// TODO(adonovan): this whole algorithm could be more
	// simply expressed in terms of Metadata, not Packages.
	pkgs, err := snapshot.CachedImportPaths(ctx)
	if err != nil {
		return nil, err
	}
	var (
		seen  = make(map[string]struct{})
		paths []string
	)
	for path, knownPkg := range pkgs {
		gofiles := knownPkg.CompiledGoFiles()
		if len(gofiles) == 0 || gofiles[0].File.Name == nil {
			continue
		}
		pkgName := gofiles[0].File.Name.Name
		// package main cannot be imported
		if pkgName == "main" {
			continue
		}
		// test packages cannot be imported
		if knownPkg.ForTest() != "" {
			continue
		}
		// no need to import what the file already imports
		if _, ok := alreadyImported[path]; ok {
			continue
		}
		// snapshot.KnownPackages could have multiple versions of a pkg
		if _, ok := seen[path]; ok {
			continue
		}
		seen[path] = struct{}{}
		// make sure internal packages are importable by the file
		if !IsValidImport(pkg.PkgPath(), path) {
			continue
		}
		// naive check on cyclical imports
		if isDirectlyCyclical(pkg, knownPkg) {
			continue
		}
		paths = append(paths, path)
		seen[path] = struct{}{}
	}
	err = snapshot.RunProcessEnvFunc(ctx, func(o *imports.Options) error {
		var mu sync.Mutex
		ctx, cancel := context.WithTimeout(ctx, time.Millisecond*80)
		defer cancel()
		return imports.GetAllCandidates(ctx, func(ifix imports.ImportFix) {
			mu.Lock()
			defer mu.Unlock()
			if _, ok := seen[ifix.StmtInfo.ImportPath]; ok {
				return
			}
			paths = append(paths, ifix.StmtInfo.ImportPath)
		}, "", pgf.URI.Filename(), pkg.GetTypes().Name(), o.Env)
	})
	if err != nil {
		// if an error occurred, we still have a decent list we can
		// show to the user through snapshot.CachedImportPaths
		event.Error(ctx, "imports.GetAllCandidates", err)
	}
	sort.Slice(paths, func(i, j int) bool {
		importI, importJ := paths[i], paths[j]
		iHasDot := strings.Contains(importI, ".")
		jHasDot := strings.Contains(importJ, ".")
		if iHasDot && !jHasDot {
			return false
		}
		if jHasDot && !iHasDot {
			return true
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
func isDirectlyCyclical(pkg, imported Package) bool {
	for _, imp := range imported.Imports() {
		if imp.PkgPath() == pkg.PkgPath() {
			return true
		}
	}
	return false
}
