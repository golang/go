// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.Diagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	reports := map[source.VersionedFileIdentity][]*source.Diagnostic{}
	for _, uri := range snapshot.ModFiles() {
		fh, err := snapshot.GetVersionedFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		reports[fh.VersionedFileIdentity()] = []*source.Diagnostic{}
		diagnostics, err := DiagnosticsForMod(ctx, snapshot, fh)
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

func DiagnosticsForMod(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) ([]*source.Diagnostic, error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		if pm == nil || len(pm.ParseErrors) == 0 {
			return nil, err
		}
		return pm.ParseErrors, nil
	}

	var diagnostics []*source.Diagnostic

	// Add upgrade quick fixes for individual modules if we know about them.
	upgrades := snapshot.View().ModuleUpgrades()
	for _, req := range pm.File.Require {
		ver, ok := upgrades[req.Mod.Path]
		if !ok || req.Mod.Version == ver {
			continue
		}
		rng, err := lineToRange(pm.Mapper, fh.URI(), req.Syntax.Start, req.Syntax.End)
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
		diagnostics = append(diagnostics, &source.Diagnostic{
			URI:            fh.URI(),
			Range:          rng,
			Severity:       protocol.SeverityInformation,
			Source:         source.UpgradeNotification,
			Message:        fmt.Sprintf("%v can be upgraded", req.Mod.Path),
			SuggestedFixes: []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)},
		})
	}

	// Packages in the workspace can contribute diagnostics to go.mod files.
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
