// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"context"
	"debug/buildinfo"
	"fmt"
	"go/build"
	"os/exec"

	"cmd/go/internal/load"
	"cmd/go/internal/modload"
)

// loadVersioned loads a package at a specific version.
func loadVersioned(ctx context.Context, pkgPath, version string) (*build.Package, error) {
	loaderState := modload.NewState()
	loaderState.ForceUseModules = true
	loaderState.RootMode = modload.NoRoot
	modload.Init(loaderState)

	var opts load.PackageOpts
	args := []string{
		fmt.Sprintf("%s@%s", pkgPath, version),
	}
	pkgs, err := load.PackagesAndErrorsOutsideModule(loaderState, ctx, opts, args)
	if err != nil {
		return nil, err
	}
	if len(pkgs) != 1 {
		return nil, fmt.Errorf("incorrect number of packages: want 1, got %d", len(pkgs))
	}
	return pkgs[0].Internal.Build, nil
}

// inferVersion checks if the argument matches a command on $PATH and returns its module path and version.
func inferVersion(arg string) (pkgPath, version string, ok bool) {
	path, err := exec.LookPath(arg)
	if err != nil {
		return "", "", false
	}
	bi, err := buildinfo.ReadFile(path)
	if err != nil {
		return "", "", false
	}
	if bi.Main.Path == "" || bi.Main.Version == "" {
		return "", "", false
	}
	// bi.Path is the package path for the main package.
	return bi.Path, bi.Main.Version, true
}
