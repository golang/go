// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mod

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
)

func Hover(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, position protocol.Position) (*protocol.Hover, error) {
	var found bool
	for _, uri := range snapshot.ModFiles() {
		if fh.URI() == uri {
			found = true
			break
		}
	}

	// We only provide hover information for the view's go.mod files.
	if !found {
		return nil, nil
	}

	ctx, done := event.Start(ctx, "mod.Hover")
	defer done()

	// Get the position of the cursor.
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, fmt.Errorf("getting modfile handle: %w", err)
	}
	offset, err := pm.Mapper.PositionOffset(position)
	if err != nil {
		return nil, fmt.Errorf("computing cursor position: %w", err)
	}

	// If the cursor position is on a module statement
	if hover, ok := hoverOnModuleStatement(ctx, pm, offset, snapshot, fh); ok {
		return hover, nil
	}
	return hoverOnRequireStatement(ctx, pm, offset, snapshot, fh)
}

func hoverOnRequireStatement(ctx context.Context, pm *source.ParsedModule, offset int, snapshot source.Snapshot, fh source.FileHandle) (*protocol.Hover, error) {
	// Confirm that the cursor is at the position of a require statement.
	var req *modfile.Require
	var startOffset, endOffset int
	for _, r := range pm.File.Require {
		dep := []byte(r.Mod.Path)
		s, e := r.Syntax.Start.Byte, r.Syntax.End.Byte
		i := bytes.Index(pm.Mapper.Content[s:e], dep)
		if i == -1 {
			continue
		}
		// Shift the start position to the location of the
		// dependency within the require statement.
		startOffset, endOffset = s+i, e
		if startOffset <= offset && offset <= endOffset {
			req = r
			break
		}
	}
	// TODO(hyangah): find position for info about vulnerabilities in Go

	// The cursor position is not on a require statement.
	if req == nil {
		return nil, nil
	}

	// Get the vulnerability info.
	fromGovulncheck := true
	vs := snapshot.View().Vulnerabilities(fh.URI())[fh.URI()]
	if vs == nil && snapshot.View().Options().Vulncheck == source.ModeVulncheckImports {
		var err error
		vs, err = snapshot.ModVuln(ctx, fh.URI())
		if err != nil {
			return nil, err
		}
		fromGovulncheck = false
	}
	affecting, nonaffecting := lookupVulns(vs, req.Mod.Path, req.Mod.Version)

	// Get the `go mod why` results for the given file.
	why, err := snapshot.ModWhy(ctx, fh)
	if err != nil {
		return nil, err
	}
	explanation, ok := why[req.Mod.Path]
	if !ok {
		return nil, nil
	}

	// Get the range to highlight for the hover.
	// TODO(hyangah): adjust the hover range to include the version number
	// to match the diagnostics' range.
	rng, err := pm.Mapper.OffsetRange(startOffset, endOffset)
	if err != nil {
		return nil, err
	}
	options := snapshot.View().Options()
	isPrivate := snapshot.View().IsGoPrivatePath(req.Mod.Path)
	header := formatHeader(req.Mod.Path, options)
	explanation = formatExplanation(explanation, req, options, isPrivate)
	vulns := formatVulnerabilities(req.Mod.Path, affecting, nonaffecting, options, fromGovulncheck)

	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  options.PreferredContentFormat,
			Value: header + vulns + explanation,
		},
		Range: rng,
	}, nil
}

func hoverOnModuleStatement(ctx context.Context, pm *source.ParsedModule, offset int, snapshot source.Snapshot, fh source.FileHandle) (*protocol.Hover, bool) {
	module := pm.File.Module
	if module == nil {
		return nil, false // no module stmt
	}
	if offset < module.Syntax.Start.Byte || offset > module.Syntax.End.Byte {
		return nil, false // cursor not in module stmt
	}

	rng, err := pm.Mapper.OffsetRange(module.Syntax.Start.Byte, module.Syntax.End.Byte)
	if err != nil {
		return nil, false
	}
	fromGovulncheck := true
	vs := snapshot.View().Vulnerabilities(fh.URI())[fh.URI()]

	if vs == nil && snapshot.View().Options().Vulncheck == source.ModeVulncheckImports {
		vs, err = snapshot.ModVuln(ctx, fh.URI())
		if err != nil {
			return nil, false
		}
		fromGovulncheck = false
	}
	modpath := "stdlib"
	goVersion := snapshot.View().GoVersionString()
	affecting, nonaffecting := lookupVulns(vs, modpath, goVersion)
	options := snapshot.View().Options()
	vulns := formatVulnerabilities(modpath, affecting, nonaffecting, options, fromGovulncheck)

	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  options.PreferredContentFormat,
			Value: vulns,
		},
		Range: rng,
	}, true
}

func formatHeader(modpath string, options *source.Options) string {
	var b strings.Builder
	// Write the heading as an H3.
	b.WriteString("#### " + modpath)
	if options.PreferredContentFormat == protocol.Markdown {
		b.WriteString("\n\n")
	} else {
		b.WriteRune('\n')
	}
	return b.String()
}

func lookupVulns(vulns *govulncheck.Result, modpath, version string) (affecting, nonaffecting []*govulncheck.Vuln) {
	if vulns == nil {
		return nil, nil
	}
	for _, vuln := range vulns.Vulns {
		for _, mod := range vuln.Modules {
			if mod.Path != modpath {
				continue
			}
			// It is possible that the source code was changed since the last
			// govulncheck run and information in the `vulns` info is stale.
			// For example, imagine that a user is in the middle of updating
			// problematic modules detected by the govulncheck run by applying
			// quick fixes. Stale diagnostics can be confusing and prevent the
			// user from quickly locating the next module to fix.
			// Ideally we should rerun the analysis with the updated module
			// dependencies or any other code changes, but we are not yet
			// in the position of automatically triggering the analysis
			// (govulncheck can take a while). We also don't know exactly what
			// part of source code was changed since `vulns` was computed.
			// As a heuristic, we assume that a user upgrades the affecting
			// module to the version with the fix or the latest one, and if the
			// version in the require statement is equal to or higher than the
			// fixed version, skip the vulnerability information in the hover.
			// Eventually, the user has to rerun govulncheck.
			if mod.FixedVersion != "" && semver.IsValid(version) && semver.Compare(mod.FixedVersion, version) <= 0 {
				continue
			}
			if vuln.IsCalled() {
				affecting = append(affecting, vuln)
			} else {
				nonaffecting = append(nonaffecting, vuln)
			}
		}
	}
	sort.Slice(nonaffecting, func(i, j int) bool { return nonaffecting[i].OSV.ID < nonaffecting[j].OSV.ID })
	sort.Slice(affecting, func(i, j int) bool { return affecting[i].OSV.ID < affecting[j].OSV.ID })
	return affecting, nonaffecting
}

func formatVulnerabilities(modPath string, affecting, nonaffecting []*govulncheck.Vuln, options *source.Options, fromGovulncheck bool) string {
	if len(affecting) == 0 && len(nonaffecting) == 0 {
		return ""
	}

	// TODO(hyangah): can we use go templates to generate hover messages?
	// Then, we can use a different template for markdown case.
	useMarkdown := options.PreferredContentFormat == protocol.Markdown

	var b strings.Builder

	if len(affecting) > 0 {
		// TODO(hyangah): make the message more eyecatching (icon/codicon/color)
		if len(affecting) == 1 {
			b.WriteString(fmt.Sprintf("\n**WARNING:** Found %d reachable vulnerability.\n", len(affecting)))
		} else {
			b.WriteString(fmt.Sprintf("\n**WARNING:** Found %d reachable vulnerabilities.\n", len(affecting)))
		}
	}
	for _, v := range affecting {
		fix := fixedVersionInfo(v, modPath)
		pkgs := vulnerablePkgsInfo(v, modPath, useMarkdown)

		if useMarkdown {
			fmt.Fprintf(&b, "- [**%v**](%v) %v%v%v\n", v.OSV.ID, href(v.OSV), formatMessage(v), pkgs, fix)
		} else {
			fmt.Fprintf(&b, "  - [%v] %v (%v) %v%v\n", v.OSV.ID, formatMessage(v), href(v.OSV), pkgs, fix)
		}
	}
	if len(nonaffecting) > 0 {
		if fromGovulncheck {
			fmt.Fprintf(&b, "\n**Note:** The project imports packages with known vulnerabilities, but does not call the vulnerable code.\n")
		} else {
			fmt.Fprintf(&b, "\n**Note:** The project imports packages with known vulnerabilities. Use `govulncheck` to check if the project uses vulnerable symbols.\n")
		}
	}
	for _, v := range nonaffecting {
		fix := fixedVersionInfo(v, modPath)
		pkgs := vulnerablePkgsInfo(v, modPath, useMarkdown)
		if useMarkdown {
			fmt.Fprintf(&b, "- [%v](%v) %v%v%v\n", v.OSV.ID, href(v.OSV), formatMessage(v), pkgs, fix)
		} else {
			fmt.Fprintf(&b, "  - [%v] %v (%v) %v%v\n", v.OSV.ID, formatMessage(v), href(v.OSV), pkgs, fix)
		}
	}
	b.WriteString("\n")
	return b.String()
}

func vulnerablePkgsInfo(v *govulncheck.Vuln, modPath string, useMarkdown bool) string {
	var b bytes.Buffer
	for _, m := range v.Modules {
		if m.Path != modPath {
			continue
		}
		if c := len(m.Packages); c == 1 {
			b.WriteString("\n  Vulnerable package is:")
		} else if c > 1 {
			b.WriteString("\n  Vulnerable packages are:")
		}
		for _, pkg := range m.Packages {
			if useMarkdown {
				b.WriteString("\n  * `")
			} else {
				b.WriteString("\n    ")
			}
			b.WriteString(pkg.Path)
			if useMarkdown {
				b.WriteString("`")
			}
		}
	}
	if b.Len() == 0 {
		return ""
	}
	return b.String()
}
func fixedVersionInfo(v *govulncheck.Vuln, modPath string) string {
	fix := "\n\n  **No fix is available.**"
	for _, m := range v.Modules {
		if m.Path != modPath {
			continue
		}
		if m.FixedVersion != "" {
			fix = "\n\n  Fixed in " + m.FixedVersion + "."
		}
		break
	}
	return fix
}

func formatExplanation(text string, req *modfile.Require, options *source.Options, isPrivate bool) string {
	text = strings.TrimSuffix(text, "\n")
	splt := strings.Split(text, "\n")
	length := len(splt)

	var b strings.Builder

	// If the explanation is 2 lines, then it is of the form:
	// # golang.org/x/text/encoding
	// (main module does not need package golang.org/x/text/encoding)
	if length == 2 {
		b.WriteString(splt[1])
		return b.String()
	}

	imp := splt[length-1] // import path
	reference := imp
	// See golang/go#36998: don't link to modules matching GOPRIVATE.
	if !isPrivate && options.PreferredContentFormat == protocol.Markdown {
		target := imp
		if strings.ToLower(options.LinkTarget) == "pkg.go.dev" {
			target = strings.Replace(target, req.Mod.Path, req.Mod.String(), 1)
		}
		reference = fmt.Sprintf("[%s](%s)", imp, source.BuildLink(options.LinkTarget, target, ""))
	}
	b.WriteString("This module is necessary because " + reference + " is imported in")

	// If the explanation is 3 lines, then it is of the form:
	// # golang.org/x/tools
	// modtest
	// golang.org/x/tools/go/packages
	if length == 3 {
		msg := fmt.Sprintf(" `%s`.", splt[1])
		b.WriteString(msg)
		return b.String()
	}

	// If the explanation is more than 3 lines, then it is of the form:
	// # golang.org/x/text/language
	// rsc.io/quote
	// rsc.io/sampler
	// golang.org/x/text/language
	b.WriteString(":\n```text")
	dash := ""
	for _, imp := range splt[1 : length-1] {
		dash += "-"
		b.WriteString("\n" + dash + " " + imp)
	}
	b.WriteString("\n```")
	return b.String()
}
