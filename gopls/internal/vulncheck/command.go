// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"context"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/vuln/client"
	"golang.org/x/vuln/vulncheck"
)

func init() {
	Govulncheck = govulncheck
}

func govulncheck(ctx context.Context, cfg *packages.Config, args command.VulncheckArgs) (res command.VulncheckResult, _ error) {
	if args.Pattern == "" {
		args.Pattern = "."
	}

	dbClient, err := client.NewClient(findGOVULNDB(cfg), client.Options{HTTPCache: defaultCache()})
	if err != nil {
		return res, err
	}

	c := cmd{Client: dbClient}
	vulns, err := c.Run(ctx, cfg, args.Pattern)
	if err != nil {
		return res, err
	}

	res.Vuln = vulns
	return res, err
}

func findGOVULNDB(cfg *packages.Config) []string {
	for _, kv := range cfg.Env {
		if strings.HasPrefix(kv, "GOVULNDB=") {
			return strings.Split(kv[len("GOVULNDB="):], ",")
		}
	}
	if GOVULNDB := os.Getenv("GOVULNDB"); GOVULNDB != "" {
		return strings.Split(GOVULNDB, ",")
	}
	return []string{"https://vuln.go.dev"}
}

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
	cfg.Mode |= packages.NeedModule | packages.NeedName | packages.NeedFiles |
		packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedTypes |
		packages.NeedTypesSizes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedDeps

	log.Println("loading packages...")

	loadedPkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		log.Printf("package load failed: %v", err)
		return nil, err
	}
	log.Printf("loaded %d packages\n", len(loadedPkgs))

	pkgs := vulncheck.Convert(loadedPkgs)
	r, err := vulncheck.Source(ctx, pkgs, &vulncheck.Config{
		Client: c.Client,
	})
	if err != nil {
		return nil, err
	}

	// Skip vulns that are in the import graph but have no calls to them.
	var vulns []*vulncheck.Vuln
	for _, v := range r.Vulns {
		if v.CallSink != 0 {
			vulns = append(vulns, v)
		}
	}

	callStacks := vulncheck.CallStacks(r)
	// Create set of top-level packages, used to find representative symbols
	topPackages := map[string]bool{}
	for _, p := range pkgs {
		topPackages[p.PkgPath] = true
	}
	vulnGroups := groupByIDAndPackage(vulns)
	moduleVersions := moduleVersionMap(r.Modules)

	return toVulns(callStacks, moduleVersions, topPackages, vulnGroups)
	// TODO: add import graphs.
}

func toVulns(callStacks map[*vulncheck.Vuln][]vulncheck.CallStack, moduleVersions map[string]string, topPackages map[string]bool, vulnGroups [][]*vulncheck.Vuln) ([]Vuln, error) {
	var vulns []Vuln

	for _, vg := range vulnGroups {
		v0 := vg[0]
		vuln := Vuln{
			ID:             v0.OSV.ID,
			PkgPath:        v0.PkgPath,
			CurrentVersion: moduleVersions[v0.ModPath],
			FixedVersion:   latestFixed(v0.OSV.Affected),
			Details:        v0.OSV.Details,

			Aliases: v0.OSV.Aliases,
			Symbol:  v0.Symbol,
			ModPath: v0.ModPath,
			URL:     href(v0.OSV),
		}

		// Keep first call stack for each vuln.
		for _, v := range vg {
			if css := callStacks[v]; len(css) > 0 {
				vuln.CallStacks = append(vuln.CallStacks, toCallStack(css[0]))
				vuln.CallStackSummaries = append(vuln.CallStackSummaries, summarizeCallStack(css[0], topPackages, v.PkgPath))
			}
		}
		vulns = append(vulns, vuln)
	}
	return vulns, nil
}
