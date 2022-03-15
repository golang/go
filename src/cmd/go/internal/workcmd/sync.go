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

var cmdSync = &base.Command{
	UsageLine: "go work sync",
	Short:     "sync workspace build list to modules",
	Long: `Sync syncs the workspace's build list back to the
workspace's modules

The workspace's build list is the set of versions of all the
(transitive) dependency modules used to do builds in the workspace. go
work sync generates that build list using the Minimal Version Selection
algorithm, and then syncs those versions back to each of modules
specified in the workspace (with use directives).

The syncing is done by sequentially upgrading each of the dependency
modules specified in a workspace module to the version in the build list
if the dependency module's version is not already the same as the build
list's version. Note that Minimal Version Selection guarantees that the
build list's version of each module is always the same or higher than
that in each workspace module.

See the workspaces reference at https://go.dev/ref/mod#workspaces
for more information.
`,
	Run: runSync,
}

func init() {
	base.AddModCommonFlags(&cmdSync.Flag)
}

func runSync(ctx context.Context, cmd *base.Command, args []string) {
	modload.ForceUseModules = true
	modload.InitWorkfile()
	if modload.WorkFilePath() == "" {
		base.Fatalf("go: no go.work file found\n\t(run 'go work init' first or specify path using GOWORK environment variable)")
	}

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

	workFilePath := modload.WorkFilePath() // save go.work path because EnterModule clobbers it.

	for _, m := range mms.Versions() {
		if mms.ModRoot(m) == "" && m.Path == "command-line-arguments" {
			// This is not a real module.
			// TODO(#49228): Remove this special case once the special
			// command-line-arguments module is gone.
			continue
		}

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
			Tidy:                     true,
			VendorModulesInGOROOTSrc: true,
			ResolveMissingImports:    false,
			LoadTests:                true,
			AllowErrors:              true,
			SilenceMissingStdImports: true,
			SilencePackageErrors:     true,
		}, "all")
		modload.WriteGoMod(ctx)
	}

	wf, err := modload.ReadWorkFile(workFilePath)
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	modload.UpdateWorkFile(wf)
	if err := modload.WriteWorkFile(workFilePath, wf); err != nil {
		base.Fatalf("go: %v", err)
	}
}
