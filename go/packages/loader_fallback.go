// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(matloob): Delete this file once Go 1.12 is released.

// This file provides backwards compatibility support for
// loading for versions of Go earlier than 1.10.4. This support is meant to
// assist with migration to the Package API until there's
// widespread adoption of these newer Go versions.
// This support will be removed once Go 1.12 is released
// in Q1 2019.

// The support is incomplete. These are some of the missing
// features:
// - the Tests option has no behavior, and test packages are
//   never returned.
// - Package.OtherFiles are always missing even if the package
//   contains non-go sources.

package packages

import (
	"fmt"
	"go/build"
	legacy "golang.org/x/tools/go/loader"
	"golang.org/x/tools/imports"
	"strings"
)

func loaderFallback(dir string, env []string, patterns []string) ([]*Package, error) {
	cfg := legacy.Config{}
	cfg.Cwd = dir
	cfg.AllowErrors = true
	cfg.FromArgs(patterns, false) // test packages are not supported

	// Set build ctx
	buildCtx := build.Default
	for _, ev := range env {
		sp := strings.Split(ev, "=")
		if len(sp) != 2 {
			continue
		}
		evar, val := sp[0], sp[1]
		switch evar {
		case "GOPATH":
			buildCtx.GOPATH = val
		case "GOROOT":
			buildCtx.GOROOT = val
		case "GOARCH":
			buildCtx.GOARCH = val
		case "GOOS":
			buildCtx.GOOS = val
		}
	}
	cfg.Build = &buildCtx

	lprog, err := cfg.Load()
	if err != nil {
		if err.Error() == "no initial packages were loaded" {
			return nil, fmt.Errorf("packages not found") // Return same error as golist-based code
		}
		return nil, fmt.Errorf("failed to load packages with legacy loader: %v", err)
	}

	allpkgs := make(map[string]*loaderPackage)

	initial := make(map[*legacy.PackageInfo]bool)
	for _, lpkg := range lprog.InitialPackages() {
		initial[lpkg] = true
	}
	for _, lpkg := range lprog.AllPackages {
		id := lpkg.Pkg.Path()

		var goFiles []string
		for _, f := range lpkg.Files {
			goFiles = append(goFiles, lprog.Fset.File(f.Pos()).Name())
		}

		pkgimports := make(map[string]string)
		for _, imppkg := range lpkg.Pkg.Imports() {
			// TODO(matloob): Is the import path of a package always VendorlessPath(path)?
			pkgimports[imports.VendorlessPath(imppkg.Path())] = imppkg.Path()
		}

		allpkgs[id] = &loaderPackage{
			Package: &Package{
				ID:         id,
				Name:       lpkg.Pkg.Name(),
				GoFiles:    goFiles,
				Fset:       lprog.Fset,
				Syntax:     lpkg.Files,
				Errors:     lpkg.Errors,
				Types:      lpkg.Pkg,
				TypesInfo:  &lpkg.Info,
				IllTyped:   !lpkg.TransitivelyErrorFree,
				OtherFiles: nil, // Never set for the fallback, because we can't extract from loader.
			},
			imports: pkgimports,
		}
	}

	// Do a second pass to populate imports.
	for _, pkg := range allpkgs {
		pkg.Imports = make(map[string]*Package)
		for imppath, impid := range pkg.imports {
			target, ok := allpkgs[impid]
			if !ok {
				// return nil, fmt.Errorf("could not load package: %v", impid)
				continue
			}
			pkg.Imports[imppath] = target.Package
		}
	}

	// Grab the initial set of packages.
	var packages []*Package
	for _, lpkg := range lprog.InitialPackages() {
		packages = append(packages, allpkgs[lpkg.Pkg.Path()].Package)
	}

	return packages, nil
}
