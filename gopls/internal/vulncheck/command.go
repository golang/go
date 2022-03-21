// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

// Package vulncheck provides an analysis command
// that runs vulnerability analysis using data from
// golang.org/x/exp/vulncheck.
// This package requires go1.18 or newer.
package vulncheck

import (
	"context"
	"fmt"

	"golang.org/x/exp/vulncheck"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/vuln/client"
)

// CallStack models a trace of function calls starting
// with a client function or method and ending with a
// call to a vulnerable symbol.
type CallStack []StackEntry

// StackEntry models an element of a call stack.
type StackEntry struct {
	// See golang.org/x/exp/vulncheck.StackEntry.

	// User-friendly representation of function/method names.
	// e.g. package.funcName, package.(recvType).methodName, ...
	Name string
	URI  span.URI
	Pos  protocol.Position // Start position. (0-based. Column is always 0)
}

// Vuln models an osv.Entry and representative call stacks.
type Vuln struct {
	// ID is the vulnerability ID (osv.Entry.ID).
	// https://ossf.github.io/osv-schema/#id-modified-fields
	ID string `json:"id,omitempty"`
	// Details is the description of the vulnerability (osv.Entry.Details).
	// https://ossf.github.io/osv-schema/#summary-details-fields
	Details string `json:"details,omitempty"`
	// Aliases are alternative IDs of the vulnerability.
	// https://ossf.github.io/osv-schema/#aliases-field
	Aliases []string `json:"aliases,omitempty"`

	// Symbol is the name of the detected vulnerable function or method.
	Symbol string `json:"symbol,omitempty"`
	// PkgPath is the package path of the detected Symbol.
	PkgPath string `json:"pkg_path,omitempty"`
	// ModPath is the module path corresponding to PkgPath.
	// TODO: don't we need explicit module version?
	// TODO: how do we specify standard library's vulnerability?
	ModPath string `json:"mod_path,omitempty"`

	// URL is the URL for more info about the information.
	// Either the database specific URL or the one of the URLs
	// included in osv.Entry.References.
	URL string `json:"url,omitempty"`

	// Current is the current module version.
	CurrentVersion string `json:"current_version,omitempty"`

	// Fixed is the minimum module version that contains the fix.
	FixedVersion string `json:"fixed_version,omitempty"`

	// Example call stacks.
	CallStacks []CallStack `json:"call_stacks,omitempty"`

	// TODO: import graph & module graph.
}

// cmd is an in-process govulncheck command runner
// that uses the provided client.Client.
type cmd struct {
	Client client.Client
}

// Run runs the govulncheck after loading packages using the provided packages.Config.
func (c *cmd) Run(ctx context.Context, cfg *packages.Config, patterns ...string) (_ []Vuln, err error) {
	// TODO: how&where can we ensure cfg is the right config for the given patterns?

	// vulncheck.Source may panic if the packages are incomplete. (e.g. broken code or failed dependency fetch)
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("cannot run vulncheck: %v", r)
		}
	}()
	return c.run(ctx, cfg, patterns)
}

func (c *cmd) run(ctx context.Context, packagesCfg *packages.Config, patterns []string) ([]Vuln, error) {
	packagesCfg.Mode |= packages.NeedModule | packages.NeedName | packages.NeedFiles |
		packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedTypes |
		packages.NeedTypesSizes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedDeps

	loadedPkgs, err := packages.Load(packagesCfg, patterns...)
	if err != nil {
		return nil, err
	}
	pkgs := vulncheck.Convert(loadedPkgs)
	res, err := vulncheck.Source(ctx, pkgs, &vulncheck.Config{
		Client:      c.Client,
		ImportsOnly: false,
	})
	cs := vulncheck.CallStacks(res)

	return toVulns(loadedPkgs, cs)

	// TODO: add import graphs.
}

func packageModule(p *packages.Package) *packages.Module {
	m := p.Module
	if m == nil {
		return nil
	}
	if r := m.Replace; r != nil {
		return r
	}
	return m
}

func toVulns(pkgs []*packages.Package, callstacks map[*vulncheck.Vuln][]vulncheck.CallStack) ([]Vuln, error) {
	// Build a map from module paths to versions.
	moduleVersions := map[string]string{}
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		if m := packageModule(p); m != nil {
			moduleVersions[m.Path] = m.Version
		}
	})

	var vulns []Vuln
	for v, trace := range callstacks {
		vuln := Vuln{
			ID:             v.OSV.ID,
			Details:        v.OSV.Details,
			Aliases:        v.OSV.Aliases,
			Symbol:         v.Symbol,
			PkgPath:        v.PkgPath,
			ModPath:        v.ModPath,
			URL:            href(v.OSV),
			CurrentVersion: moduleVersions[v.ModPath],
			FixedVersion:   fixedVersion(v.OSV),
			CallStacks:     toCallStacks(trace),
		}
		vulns = append(vulns, vuln)
	}
	return vulns, nil
}
