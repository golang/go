// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/vuln/osv"
)

// Diagnostics returns diagnostics for the modules in the workspace.
//
// It waits for completion of type-checking of all active packages.
func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.Diagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	return collectDiagnostics(ctx, snapshot, ModDiagnostics)
}

// UpgradeDiagnostics returns upgrade diagnostics for the modules in the
// workspace with known upgrades.
func UpgradeDiagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.UpgradeDiagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	return collectDiagnostics(ctx, snapshot, ModUpgradeDiagnostics)
}

// VulnerabilityDiagnostics returns vulnerability diagnostics for the active modules in the
// workspace with known vulnerabilites.
func VulnerabilityDiagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.VersionedFileIdentity][]*source.Diagnostic, error) {
	ctx, done := event.Start(ctx, "mod.VulnerabilityDiagnostics", tag.Snapshot.Of(snapshot.ID()))
	defer done()

	return collectDiagnostics(ctx, snapshot, ModVulnerabilityDiagnostics)
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

// ModDiagnostics waits for completion of type-checking of all active
// packages, then returns diagnostics from diagnosing the packages in
// the workspace and from tidying the go.mod file.
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
		// Don't return an error if there are parse error diagnostics to be shown, but also do not
		// continue since we won't be able to show the upgrade diagnostics.
		if pm != nil && len(pm.ParseErrors) != 0 {
			return nil, nil
		}
		return nil, err
	}

	upgrades := snapshot.View().ModuleUpgrades(fh.URI())
	for _, req := range pm.File.Require {
		ver, ok := upgrades[req.Mod.Path]
		if !ok || req.Mod.Version == ver {
			continue
		}
		rng, err := pm.Mapper.OffsetRange(req.Syntax.Start.Byte, req.Syntax.End.Byte)
		if err != nil {
			return nil, err
		}
		// Upgrade to the exact version we offer the user, not the most recent.
		title := fmt.Sprintf("%s%v", upgradeCodeActionPrefix, ver)
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

const upgradeCodeActionPrefix = "Upgrade to "

// ModVulnerabilityDiagnostics adds diagnostics for vulnerabilities in individual modules
// if the vulnerability is recorded in the view.
func ModVulnerabilityDiagnostics(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) (vulnDiagnostics []*source.Diagnostic, err error) {
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		// Don't return an error if there are parse error diagnostics to be shown, but also do not
		// continue since we won't be able to show the vulnerability diagnostics.
		if pm != nil && len(pm.ParseErrors) != 0 {
			return nil, nil
		}
		return nil, err
	}

	vs := snapshot.View().Vulnerabilities(fh.URI())
	// TODO(suzmue): should we just store the vulnerabilities like this?
	type modVuln struct {
		mod  *govulncheck.Module
		vuln *govulncheck.Vuln
	}
	vulnsByModule := make(map[string][]modVuln)
	for _, vuln := range vs {
		for _, mod := range vuln.Modules {
			vulnsByModule[mod.Path] = append(vulnsByModule[mod.Path], modVuln{mod, vuln})
		}
	}

	for _, req := range pm.File.Require {
		vulns := vulnsByModule[req.Mod.Path]
		if len(vulns) == 0 {
			continue
		}
		rng, err := pm.Mapper.OffsetRange(req.Syntax.Start.Byte, req.Syntax.End.Byte)
		if err != nil {
			return nil, err
		}
		// Map affecting vulns to 'warning' level diagnostics,
		// others to 'info' level diagnostics.
		// Fixes will include only the upgrades for warning level diagnostics.
		var fixes []source.SuggestedFix
		var warning, info []string
		for _, mv := range vulns {
			mod, vuln := mv.mod, mv.vuln
			// Only show the diagnostic if the vulnerability was calculated
			// for the module at the current version.
			// TODO(hyangah): this prevents from surfacing vulnerable modules
			// since module version selection is affected by dependency module
			// requirements and replace/exclude/go.work use. Drop this check
			// and annotate only the module path.
			if semver.IsValid(mod.FoundVersion) && semver.Compare(req.Mod.Version, mod.FoundVersion) != 0 {
				continue
			}
			if !vuln.IsCalled() {
				info = append(info, vuln.OSV.ID)
			} else {
				warning = append(warning, vuln.OSV.ID)
			}
			// Upgrade to the exact version we offer the user, not the most recent.
			if fixedVersion := mod.FixedVersion; semver.IsValid(fixedVersion) && semver.Compare(req.Mod.Version, fixedVersion) < 0 {
				cmd, err := getUpgradeCodeAction(fh, req, fixedVersion)
				if err != nil {
					return nil, err
				}
				// Add an upgrade for module@latest.
				// TODO(suzmue): verify if latest is the same as fixedVersion.
				latest, err := getUpgradeCodeAction(fh, req, "latest")
				if err != nil {
					return nil, err
				}

				fixes = []source.SuggestedFix{
					source.SuggestedFixFromCommand(cmd, protocol.QuickFix),
					source.SuggestedFixFromCommand(latest, protocol.QuickFix),
				}
			}
		}

		if len(warning) == 0 && len(info) == 0 {
			return nil, nil
		}
		severity := protocol.SeverityInformation
		if len(warning) > 0 {
			severity = protocol.SeverityWarning
		}

		sort.Strings(warning)
		sort.Strings(info)

		var b strings.Builder
		if len(warning) == 1 {
			fmt.Fprintf(&b, "%v has a vulnerability used in the code: %v.", req.Mod.Path, warning[0])
		} else {
			fmt.Fprintf(&b, "%v has vulnerabilities used in the code: %v.", req.Mod.Path, strings.Join(warning, ", "))
		}
		if len(warning) == 0 {
			if len(info) == 1 {
				fmt.Fprintf(&b, "%v has a vulnerability %v that is not used in the code.", req.Mod.Path, info[0])
			} else {
				fmt.Fprintf(&b, "%v has known vulnerabilities %v that are not used in the code.", req.Mod.Path, strings.Join(info, ", "))
			}
		}

		vulnDiagnostics = append(vulnDiagnostics, &source.Diagnostic{
			URI:            fh.URI(),
			Range:          rng,
			Severity:       severity,
			Source:         source.Vulncheck,
			Message:        b.String(),
			SuggestedFixes: fixes,
		})
	}

	return vulnDiagnostics, nil
}

func formatMessage(v *govulncheck.Vuln) string {
	details := []byte(v.OSV.Details)
	// Remove any new lines that are not preceded or followed by a new line.
	for i, r := range details {
		if r == '\n' && i > 0 && details[i-1] != '\n' && i+1 < len(details) && details[i+1] != '\n' {
			details[i] = ' '
		}
	}
	return strings.TrimSpace(strings.Replace(string(details), "\n\n", "\n\n  ", -1))
}

// href returns a URL embedded in the entry if any.
// If no suitable URL is found, it returns a default entry in
// pkg.go.dev/vuln.
func href(vuln *osv.Entry) string {
	for _, affected := range vuln.Affected {
		if url := affected.DatabaseSpecific.URL; url != "" {
			return url
		}
	}
	for _, r := range vuln.References {
		if r.Type == "WEB" {
			return r.URL
		}
	}
	return fmt.Sprintf("https://pkg.go.dev/vuln/%s", vuln.ID)
}

func getUpgradeCodeAction(fh source.FileHandle, req *modfile.Require, version string) (protocol.Command, error) {
	cmd, err := command.NewUpgradeDependencyCommand(upgradeTitle(version), command.DependencyArgs{
		URI:        protocol.URIFromSpanURI(fh.URI()),
		AddRequire: false,
		GoCmdArgs:  []string{req.Mod.Path + "@" + version},
	})
	if err != nil {
		return protocol.Command{}, err
	}
	return cmd, nil
}

func upgradeTitle(fixedVersion string) string {
	title := fmt.Sprintf("%s%v", upgradeCodeActionPrefix, fixedVersion)
	return title
}

// SelectUpgradeCodeActions takes a list of upgrade code actions for a
// required module and returns a more selective list of upgrade code actions,
// where the code actions have been deduped.
func SelectUpgradeCodeActions(actions []protocol.CodeAction) []protocol.CodeAction {
	// TODO(suzmue): we can further limit the code actions to only return the most
	// recent version that will fix all the vulnerabilities.

	set := make(map[string]protocol.CodeAction)
	for _, action := range actions {
		set[action.Command.Title] = action
	}
	var result []protocol.CodeAction
	for _, action := range set {
		result = append(result, action)
	}
	// Sort results by version number, latest first.
	// There should be no duplicates at this point.
	sort.Slice(result, func(i, j int) bool {
		vi, vj := getUpgradeVersion(result[i]), getUpgradeVersion(result[j])
		return vi == "latest" || (vj != "latest" && semver.Compare(vi, vj) > 0)
	})
	return result
}

func getUpgradeVersion(p protocol.CodeAction) string {
	return strings.TrimPrefix(p.Title, upgradeCodeActionPrefix)
}
