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
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/vuln/client"
)

type Vuln = command.Vuln
type CallStack = command.CallStack
type StackEntry = command.StackEntry

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
