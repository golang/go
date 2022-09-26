// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
	gvc "golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/vuln/client"
	gvcapi "golang.org/x/vuln/exp/govulncheck"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

func init() {
	Govulncheck = govulncheck
}

func govulncheck(ctx context.Context, cfg *packages.Config, patterns string) (res command.VulncheckResult, _ error) {
	if patterns == "" {
		patterns = "."
	}

	dbClient, err := client.NewClient(findGOVULNDB(cfg), client.Options{HTTPCache: gvc.DefaultCache()})
	if err != nil {
		return res, err
	}

	c := cmd{Client: dbClient}
	vulns, err := c.Run(ctx, cfg, patterns)
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
	logger := log.New(log.Default().Writer(), "", 0)
	cfg.Mode |= packages.NeedModule | packages.NeedName | packages.NeedFiles |
		packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedTypes |
		packages.NeedTypesSizes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedDeps

	logger.Println("loading packages...")
	loadedPkgs, err := gvc.LoadPackages(cfg, patterns...)
	if err != nil {
		logger.Printf("%v", err)
		return nil, fmt.Errorf("package load failed")
	}

	logger.Printf("analyzing %d packages...\n", len(loadedPkgs))

	r, err := vulncheck.Source(ctx, loadedPkgs, &vulncheck.Config{Client: c.Client, SourceGoVersion: goVersion()})
	if err != nil {
		return nil, err
	}

	logger.Printf("selecting affecting vulnerabilities from %d findings...\n", len(r.Vulns))
	unaffectedMods := filterUnaffected(r.Vulns)
	r.Vulns = filterCalled(r)

	logger.Printf("found %d vulnerabilities.\n", len(r.Vulns))
	callInfo := gvc.GetCallInfo(r, loadedPkgs)
	return toVulns(callInfo, unaffectedMods)
	// TODO: add import graphs.
}

// filterCalled returns vulnerabilities where the symbols are actually called.
func filterCalled(r *vulncheck.Result) []*vulncheck.Vuln {
	var vulns []*vulncheck.Vuln
	for _, v := range r.Vulns {
		if v.CallSink != 0 {
			vulns = append(vulns, v)
		}
	}
	return vulns
}

// filterUnaffected returns vulnerabilities where no symbols are called,
// grouped by module.
func filterUnaffected(vulns []*vulncheck.Vuln) map[string][]*osv.Entry {
	// It is possible that the same vuln.OSV.ID has vuln.CallSink != 0
	// for one symbol, but vuln.CallSink == 0 for a different one, so
	// we need to filter out ones that have been called.
	called := map[string]bool{}
	for _, vuln := range vulns {
		if vuln.CallSink != 0 {
			called[vuln.OSV.ID] = true
		}
	}

	modToIDs := map[string]map[string]*osv.Entry{}
	for _, vuln := range vulns {
		if !called[vuln.OSV.ID] {
			if _, ok := modToIDs[vuln.ModPath]; !ok {
				modToIDs[vuln.ModPath] = map[string]*osv.Entry{}
			}
			// keep only one vuln.OSV instance for the same ID.
			modToIDs[vuln.ModPath][vuln.OSV.ID] = vuln.OSV
		}
	}
	output := map[string][]*osv.Entry{}
	for m, vulnSet := range modToIDs {
		var vulns []*osv.Entry
		for _, vuln := range vulnSet {
			vulns = append(vulns, vuln)
		}
		sort.Slice(vulns, func(i, j int) bool { return vulns[i].ID < vulns[j].ID })
		output[m] = vulns
	}
	return output
}

func fixed(v *osv.Entry) string {
	lf := gvc.LatestFixed(v.Affected)
	if lf != "" && lf[0] != 'v' {
		lf = "v" + lf
	}
	return lf
}

func toVulns(ci *gvc.CallInfo, unaffectedMods map[string][]*osv.Entry) ([]Vuln, error) {
	var vulns []Vuln

	for _, vg := range ci.VulnGroups {
		v0 := vg[0]
		vuln := Vuln{
			ID:             v0.OSV.ID,
			PkgPath:        v0.PkgPath,
			CurrentVersion: ci.ModuleVersions[v0.ModPath],
			FixedVersion:   fixed(v0.OSV),
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
				// TODO(hyangah):  https://go-review.googlesource.com/c/vuln/+/425183 added position info
				// in the summary but we don't need the info. Allow SummarizeCallStack to skip it optionally.
				sum := trimPosPrefix(gvc.SummarizeCallStack(css[0], ci.TopPackages, v.PkgPath))
				vuln.CallStackSummaries = append(vuln.CallStackSummaries, sum)
			}
		}
		vulns = append(vulns, vuln)
	}
	for m, vg := range unaffectedMods {
		for _, v0 := range vg {
			vuln := Vuln{
				ID:             v0.ID,
				Details:        v0.Details,
				Aliases:        v0.Aliases,
				ModPath:        m,
				URL:            href(v0),
				CurrentVersion: "",
				FixedVersion:   fixed(v0),
			}
			vulns = append(vulns, vuln)
		}
	}
	return vulns, nil
}

func trimPosPrefix(summary string) string {
	_, after, found := strings.Cut(summary, ": ")
	if !found {
		return summary
	}
	return after
}

// GoVersionForVulnTest is an internal environment variable used in gopls
// testing to examine govulncheck behavior with a go version different
// than what `go version` returns in the system.
const GoVersionForVulnTest = "_GOPLS_TEST_VULNCHECK_GOVERSION"

func init() {
	Main = func(cfg packages.Config, patterns ...string) {
		// never return
		err := gvcapi.Main(gvcapi.Config{
			AnalysisType:     "source",
			OutputType:       "summary",
			Patterns:         patterns,
			SourceLoadConfig: &cfg,
			GoVersion:        os.Getenv(GoVersionForVulnTest),
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}
