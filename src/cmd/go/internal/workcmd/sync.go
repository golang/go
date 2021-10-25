// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go work sync

package workcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/imports"
	"cmd/go/internal/modload"
	"context"

	"golang.org/x/mod/module"
)

var _ = modload.TODOWorkspaces("Add more documentation below. Though this is" +
	"enough for those trying workspaces out, there should be more through" +
	"documentation if the proposal is accepted and released.")

var cmdSync = &base.Command{
	UsageLine: "go work sync [moddirs]",
	Short:     "sync workspace build list to modules",
	Long:      `go work sync`,
	Run:       runSync,
}

func init() {
	base.AddModCommonFlags(&cmdSync.Flag)
	base.AddWorkfileFlag(&cmdSync.Flag)
}

func runSync(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	modload.ForceUseModules = true

	workGraph := modload.LoadModGraph(ctx, "")
	_ = workGraph
	mustSelectFor := map[module.Version][]module.Version{}

	mms := modload.MainModules

	opts := modload.PackageOpts{
		Tags:                     imports.AnyTags(),
		VendorModulesInGOROOTSrc: true,
		ResolveMissingImports:    false,
		LoadTests:                true,
		AllowErrors:              true,
		SilencePackageErrors:     true,
		SilenceUnmatchedWarnings: true,
	}
	for _, m := range mms.Versions() {
		opts.MainModule = m
		_, pkgs := modload.LoadPackages(ctx, opts, "all")
		opts.MainModule = module.Version{} // reset

		var (
			mustSelect   []module.Version
			inMustSelect = map[module.Version]bool{}
		)
		for _, pkg := range pkgs {
			if r := modload.PackageModule(pkg); r.Version != "" && !inMustSelect[r] {
				// r has a known version, so force that version.
				mustSelect = append(mustSelect, r)
				inMustSelect[r] = true
			}
		}
		module.Sort(mustSelect) // ensure determinism
		mustSelectFor[m] = mustSelect
	}

	for _, m := range mms.Versions() {
		// Use EnterModule to reset the global state in modload to be in
		// single-module mode using the modroot of m.
		modload.EnterModule(ctx, mms.ModRoot(m))

		// Edit the build list in the same way that 'go get' would if we
		// requested the relevant module versions explicitly.
		changed, err := modload.EditBuildList(ctx, nil, mustSelectFor[m])
		if err != nil {
			base.Errorf("go: %v", err)
		}
		if !changed {
			continue
		}

		modload.LoadPackages(ctx, modload.PackageOpts{
			Tags:                     imports.AnyTags(),
			VendorModulesInGOROOTSrc: true,
			ResolveMissingImports:    false,
			LoadTests:                true,
			AllowErrors:              true,
			SilencePackageErrors:     true,
			Tidy:                     true,
			SilenceUnmatchedWarnings: true,
		}, "all")
		modload.WriteGoMod(ctx)
	}
}
