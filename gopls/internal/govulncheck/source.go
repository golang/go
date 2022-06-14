// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/vuln/client"
	"golang.org/x/vuln/vulncheck"
)

// A PackageError contains errors from loading a set of packages.
type PackageError struct {
	Errors []packages.Error
}

func (e *PackageError) Error() string {
	var b strings.Builder
	fmt.Fprintln(&b, "Packages contain errors:")
	for _, e := range e.Errors {
		fmt.Println(&b, e)
	}
	return b.String()
}

// LoadPackages loads the packages matching patterns using cfg, after setting
// the cfg mode flags that vulncheck needs for analysis.
// If the packages contain errors, a PackageError is returned containing a list of the errors,
// along with the packages themselves.
func LoadPackages(cfg *packages.Config, patterns ...string) ([]*vulncheck.Package, error) {
	cfg.Mode |= packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
		packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
		packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedDeps |
		packages.NeedModule

	pkgs, err := packages.Load(cfg, patterns...)
	vpkgs := vulncheck.Convert(pkgs)
	if err != nil {
		return nil, err
	}
	var perrs []packages.Error
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		perrs = append(perrs, p.Errors...)
	})
	if len(perrs) > 0 {
		err = &PackageError{perrs}
	}
	return vpkgs, err
}

// Source calls vulncheck.Source on the Go source in pkgs. It returns the result
// with Vulns trimmed to those that are actually called.
//
// This function is being used by the Go IDE team.
func Source(ctx context.Context, pkgs []*vulncheck.Package, c client.Client) (*vulncheck.Result, error) {
	r, err := vulncheck.Source(ctx, pkgs, &vulncheck.Config{Client: c})
	if err != nil {
		return nil, err
	}
	// Keep only the vulns that are called.
	var vulns []*vulncheck.Vuln
	for _, v := range r.Vulns {
		if v.CallSink != 0 {
			vulns = append(vulns, v)
		}
	}
	r.Vulns = vulns
	return r, nil
}

// CallInfo is information about calls to vulnerable functions.
type CallInfo struct {
	// CallStacks contains all call stacks to vulnerable functions.
	CallStacks map[*vulncheck.Vuln][]vulncheck.CallStack

	// VulnGroups contains vulnerabilities grouped by ID and package.
	VulnGroups [][]*vulncheck.Vuln

	// ModuleVersions is a map of module paths to versions.
	ModuleVersions map[string]string

	// TopPackages contains the top-level packages in the call info.
	TopPackages map[string]bool
}

// GetCallInfo computes call stacks and related information from a vulncheck.Result.
// It also makes a set of top-level packages from pkgs.
func GetCallInfo(r *vulncheck.Result, pkgs []*vulncheck.Package) *CallInfo {
	pset := map[string]bool{}
	for _, p := range pkgs {
		pset[p.PkgPath] = true
	}
	return &CallInfo{
		CallStacks:     vulncheck.CallStacks(r),
		VulnGroups:     groupByIDAndPackage(r.Vulns),
		ModuleVersions: moduleVersionMap(r.Modules),
		TopPackages:    pset,
	}
}

func groupByIDAndPackage(vs []*vulncheck.Vuln) [][]*vulncheck.Vuln {
	groups := map[[2]string][]*vulncheck.Vuln{}
	for _, v := range vs {
		key := [2]string{v.OSV.ID, v.PkgPath}
		groups[key] = append(groups[key], v)
	}

	var res [][]*vulncheck.Vuln
	for _, g := range groups {
		res = append(res, g)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i][0].PkgPath < res[j][0].PkgPath
	})
	return res
}

// moduleVersionMap builds a map from module paths to versions.
func moduleVersionMap(mods []*vulncheck.Module) map[string]string {
	moduleVersions := map[string]string{}
	for _, m := range mods {
		v := m.Version
		if m.Replace != nil {
			v = m.Replace.Version
		}
		moduleVersions[m.Path] = v
	}
	return moduleVersions
}
