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
	gvc "golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/vuln/client"
)

func init() {
	Govulncheck = govulncheck
}

func govulncheck(ctx context.Context, cfg *packages.Config, args command.VulncheckArgs) (res command.VulncheckResult, _ error) {
	if args.Pattern == "" {
		args.Pattern = "."
	}

	dbClient, err := client.NewClient(findGOVULNDB(cfg), client.Options{HTTPCache: gvc.DefaultCache()})
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
	loadedPkgs, err := gvc.LoadPackages(cfg, patterns...)
	if err != nil {
		log.Printf("package load failed: %v", err)
		return nil, err
	}
	log.Printf("loaded %d packages\n", len(loadedPkgs))

	r, err := gvc.Source(ctx, loadedPkgs, c.Client)
	if err != nil {
		return nil, err
	}
	callInfo := gvc.GetCallInfo(r, loadedPkgs)
	return toVulns(callInfo)
	// TODO: add import graphs.
}

func toVulns(ci *gvc.CallInfo) ([]Vuln, error) {
	var vulns []Vuln

	for _, vg := range ci.VulnGroups {
		v0 := vg[0]
		lf := gvc.LatestFixed(v0.OSV.Affected)
		if lf != "" && lf[0] != 'v' {
			lf = "v" + lf
		}
		vuln := Vuln{
			ID:             v0.OSV.ID,
			PkgPath:        v0.PkgPath,
			CurrentVersion: ci.ModuleVersions[v0.ModPath],
			FixedVersion:   lf,
			Details:        v0.OSV.Details,

			Aliases: v0.OSV.Aliases,
			Symbol:  v0.Symbol,
			ModPath: v0.ModPath,
			URL:     href(v0.OSV),
		}

		// Keep first call stack for each vuln.
		for _, v := range vg {
			if css := ci.CallStacks[v]; len(css) > 0 {
				vuln.CallStacks = append(vuln.CallStacks, toCallStack(css[0]))
				vuln.CallStackSummaries = append(vuln.CallStackSummaries, gvc.SummarizeCallStack(css[0], ci.TopPackages, v.PkgPath))
			}
		}
		vulns = append(vulns, vuln)
	}
	return vulns, nil
}
