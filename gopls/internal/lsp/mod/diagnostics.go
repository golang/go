// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.Diagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	return collectDiagnostics(ctx, snapshot, ModDiagnostics)
}

func UpgradeDiagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.UpgradeDiagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	return collectDiagnostics(ctx, snapshot, ModUpgradeDiagnostics)
}

func collectDiagnostics(ctx context.Context, snapshot source.Snapshot, diagFn func(context.Context, source.Snapshot, source.FileHandle) ([]*source.Diagnostic, error)) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	reports := make(map[source.VersionedFileIdentity][]*source.Diagnostic)
	for _, uri := range snapshot.ModFiles() {
		fh, err := snapshot.GetVersionedFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		reports[fh.VersionedFileIdentity()] = []*source.Diagnostic{}
		diagnostics, err := diagFn(ctx, snapshot, fh)
		if err != nil {
			return nil, err
		}
		for _, d := range diagnostics {
			fh, err := snapshot.GetVersionedFile(ctx, d.URI)
			if err != nil {
				return nil, err
			}
			reports[fh.VersionedFileIdentity()] = append(reports[fh.VersionedFileIdentity()], d)
		}
	}
	return reports, nil
}

// ModDiagnostics returns diagnostics from diagnosing the packages in the workspace and
// from tidying the go.mod file.
func ModDiagnostics(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) (diagnostics []*source.Diagnostic, err error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		if pm == nil || len(pm.ParseErrors) == 0 {
			return nil, err
		}
		return pm.ParseErrors, nil
	}

	// Packages in the workspace can contribute diagnostics to go.mod files.
	// TODO(rfindley): Try to avoid calling DiagnosePackage on all packages in the workspace here,
	// for every go.mod file. If gc_details is enabled, it looks like this could lead to extra
	// go command invocations (as gc details is not memoized).
	wspkgs, err := snapshot.ActivePackages(ctx)
	if err != nil && !source.IsNonFatalGoModError(err) {
		event.Error(ctx, fmt.Sprintf("workspace packages: diagnosing %s", pm.URI), err)
	}
	if err == nil {
		for _, pkg := range wspkgs {
			pkgDiagnostics, err := snapshot.DiagnosePackage(ctx, pkg)
			if err != nil {
				return nil, err
			}
			diagnostics = append(diagnostics, pkgDiagnostics[fh.URI()]...)
		}
	}

	tidied, err := snapshot.ModTidy(ctx, pm)
	if err != nil && !source.IsNonFatalGoModError(err) {
		event.Error(ctx, fmt.Sprintf("tidy: diagnosing %s", pm.URI), err)
	}
	if err == nil {
		for _, d := range tidied.Diagnostics {
			if d.URI != fh.URI() {
				continue
			}
			diagnostics = append(diagnostics, d)
		}
	}
	return diagnostics, nil
}

// ModUpgradeDiagnostics adds upgrade quick fixes for individual modules if the upgrades
// are recorded in the view.
func ModUpgradeDiagnostics(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) (upgradeDiagnostics []*source.Diagnostic, err error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		if pm == nil || len(pm.ParseErrors) == 0 {
			return nil, err
		}
		return nil, nil
	}

	upgrades := snapshot.View().ModuleUpgrades(fh.URI())
	for _, req := range pm.File.Require {
		ver, ok := upgrades[req.Mod.Path]
		if !ok || req.Mod.Version == ver {
			continue
		}
		rng, err := source.LineToRange(pm.Mapper, fh.URI(), req.Syntax.Start, req.Syntax.End)
		if err != nil {
			return nil, err
		}
		// Upgrade to the exact version we offer the user, not the most recent.
		title := fmt.Sprintf("Upgrade to %v", ver)
		cmd, err := command.NewUpgradeDependencyCommand(title, command.DependencyArgs{
			URI:        protocol.URIFromSpanURI(fh.URI()),
			AddRequire: false,
			GoCmdArgs:  []string{req.Mod.Path + "@" + ver},
		})
		if err != nil {
			return nil, err
		}
		upgradeDiagnostics = append(upgradeDiagnostics, &source.Diagnostic{
			URI:            fh.URI(),
			Range:          rng,
			Severity:       protocol.SeverityInformation,
			Source:         source.UpgradeNotification,
			Message:        fmt.Sprintf("%v can be upgraded", req.Mod.Path),
			SuggestedFixes: []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)},
		})
	}

	return upgradeDiagnostics, nil
}
