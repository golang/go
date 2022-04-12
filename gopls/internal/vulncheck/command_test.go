// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/vuln/client"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

func TestCmd_Run(t *testing.T) {
	runTest(t, workspace1, proxy1, func(ctx context.Context, snapshot source.Snapshot) {
		cmd := &cmd{Client: testClient1}
		cfg := packagesCfg(ctx, snapshot)
		result, err := cmd.Run(ctx, cfg, "./...")
		if err != nil {
			t.Fatal(err)
		}
		// Check that we find the right number of vulnerabilities.
		// There should be three entries as there are three vulnerable
		// symbols in the two import-reachable OSVs.
		var got []report
		for _, v := range result {
			got = append(got, toReport(v))
		}

		var want = []report{
			{
				Vuln: Vuln{
					ID:             "GO-2022-01",
					Symbol:         "VulnData.Vuln1",
					PkgPath:        "golang.org/amod/avuln",
					ModPath:        "golang.org/amod",
					URL:            "https://pkg.go.dev/vuln/GO-2022-01",
					CurrentVersion: "v1.1.3",
					FixedVersion:   "v1.0.4",
				},
				CallStacksStr: []string{
					"golang.org/cmod/c.I.t0 called from golang.org/entry/x.X [approx.] (x.go:8)\n" +
						"golang.org/amod/avuln.VulnData.Vuln1 (avuln.go:3)\n",
				},
			},
			{
				Vuln: Vuln{
					ID:             "GO-2022-01",
					Symbol:         "VulnData.Vuln2",
					PkgPath:        "golang.org/amod/avuln",
					ModPath:        "golang.org/amod",
					URL:            "https://pkg.go.dev/vuln/GO-2022-01",
					CurrentVersion: "v1.1.3",
					FixedVersion:   "v1.0.4",
				},
				CallStacksStr: []string{
					"C1 called from golang.org/entry/x.X (x.go:8)\n" +
						"Vuln2 called from golang.org/cmod/c.C1 (c.go:13)\n" +
						"golang.org/amod/avuln.VulnData.Vuln2 (avuln.go:4)\n",
				},
			},
			{
				Vuln: Vuln{
					ID:             "GO-2022-02",
					Symbol:         "Vuln",
					PkgPath:        "golang.org/bmod/bvuln",
					ModPath:        "golang.org/bmod",
					URL:            "https://pkg.go.dev/vuln/GO-2022-02",
					CurrentVersion: "v0.5.0",
				},
				CallStacksStr: []string{
					"t0 called from golang.org/entry/y.Y [approx.] (y.go:5)\n" +
						"golang.org/bmod/bvuln.Vuln (bvuln.go:2)\n",
					"Y called from golang.org/entry/x.CallY (x.go:12)\n" +
						"t0 called from golang.org/entry/y.Y [approx.] (y.go:5)\n" +
						"golang.org/bmod/bvuln.Vuln (bvuln.go:2)\n",
				},
			},
		}
		// sort reports for stability before comparison.
		for _, rpts := range [][]report{got, want} {
			sort.Slice(rpts, func(i, j int) bool {
				a, b := got[i], got[j]
				if b.ID != b.ID {
					return a.ID < b.ID
				}
				if a.PkgPath != b.PkgPath {
					return a.PkgPath < b.PkgPath
				}
				return a.Symbol < b.Symbol
			})
		}
		if diff := cmp.Diff(want, got, cmpopts.IgnoreFields(report{}, "Vuln.CallStacks")); diff != "" {
			t.Error(diff)
		}

	})
}

type report struct {
	Vuln
	// Trace is stringified Vuln.CallStacks
	CallStacksStr []string
}

func toReport(v Vuln) report {
	var r = report{Vuln: v}
	for _, s := range v.CallStacks {
		r.CallStacksStr = append(r.CallStacksStr, CallStackString(s))
	}
	return r
}

func CallStackString(callstack CallStack) string {
	var b bytes.Buffer
	for _, entry := range callstack {
		fname := filepath.Base(entry.URI.SpanURI().Filename())
		fmt.Fprintf(&b, "%v (%v:%d)\n", entry.Name, fname, entry.Pos.Line)
	}
	return b.String()
}

const workspace1 = `
-- go.mod --
module golang.org/entry

require (
	golang.org/cmod v1.1.3
)
go 1.18
-- x/x.go --
package x

import 	(
   "golang.org/cmod/c"
   "golang.org/entry/y"
)

func X() {
	c.C1().Vuln1() // vuln use: X -> Vuln1
}

func CallY() {
	y.Y()  // vuln use: CallY -> y.Y -> bvuln.Vuln 
}

-- y/y.go --
package y

import "golang.org/cmod/c"

func Y() {
	c.C2()() // vuln use: Y -> bvuln.Vuln
}
`

const proxy1 = `
-- golang.org/cmod@v1.1.3/go.mod --
module golang.org/cmod

go 1.12
-- golang.org/cmod@v1.1.3/c/c.go --
package c

import (
	"golang.org/amod/avuln"
	"golang.org/bmod/bvuln"
)

type I interface {
	Vuln1()
}

func C1() I {
	v := avuln.VulnData{}
	v.Vuln2() // vuln use
	return v
}

func C2() func() {
	return bvuln.Vuln
}
-- golang.org/amod@v1.1.3/go.mod --
module golang.org/amod

go 1.14
-- golang.org/amod@v1.1.3/avuln/avuln.go --
package avuln

type VulnData struct {}
func (v VulnData) Vuln1() {}
func (v VulnData) Vuln2() {}
-- golang.org/bmod@v0.5.0/go.mod --
module golang.org/bmod

go 1.14
-- golang.org/bmod@v0.5.0/bvuln/bvuln.go --
package bvuln

func Vuln() {
	// something evil
}
`

// testClient contains the following test vulnerabilities
//
//	golang.org/amod/avuln.{VulnData.Vuln1, vulnData.Vuln2}
//	golang.org/bmod/bvuln.{Vuln}
var testClient1 = &mockClient{
	ret: map[string][]*osv.Entry{
		"golang.org/amod": {
			{
				ID: "GO-2022-01",
				References: []osv.Reference{
					{
						Type: "href",
						URL:  "pkg.go.dev/vuln/GO-2022-01",
					},
				},
				Affected: []osv.Affected{{
					Package:           osv.Package{Name: "golang.org/amod/avuln"},
					Ranges:            osv.Affects{{Type: osv.TypeSemver, Events: []osv.RangeEvent{{Introduced: "1.0.0"}, {Fixed: "1.0.4"}, {Introduced: "1.1.2"}}}},
					EcosystemSpecific: osv.EcosystemSpecific{Symbols: []string{"VulnData.Vuln1", "VulnData.Vuln2"}},
				}},
			},
		},
		"golang.org/bmod": {
			{
				ID: "GO-2022-02",
				Affected: []osv.Affected{{
					Package:           osv.Package{Name: "golang.org/bmod/bvuln"},
					Ranges:            osv.Affects{{Type: osv.TypeSemver}},
					EcosystemSpecific: osv.EcosystemSpecific{Symbols: []string{"Vuln"}},
				}},
			},
		},
	},
}

var goldenReport1 = []string{`
{
	ID: "GO-2022-01",
	Symbol: "VulnData.Vuln1",
	PkgPath: "golang.org/amod/avuln",
	ModPath: "golang.org/amod",
	URL: "https://pkg.go.dev/vuln/GO-2022-01",
	CurrentVersion "v1.1.3",
	FixedVersion "v1.0.4",
	"call_stacks": [
	 "golang.org/cmod/c.I.t0 called from golang.org/entry/x.X [approx.] (x.go:8)\ngolang.org/amod/avuln.VulnData.Vuln1 (avuln.go:3)\n\n"
	]
}
`,
	`
{
	"id": "GO-2022-02",
	"symbol": "Vuln",
	"pkg_path": "golang.org/bmod/bvuln",
	"mod_path": "golang.org/bmod",
	"url": "https://pkg.go.dev/vuln/GO-2022-02",
	"current_version": "v0.5.0",
	"call_stacks": [
	 "t0 called from golang.org/entry/y.Y [approx.] (y.go:5)\ngolang.org/bmod/bvuln.Vuln (bvuln.go:2)\n\n",
	 "Y called from golang.org/entry/x.CallY (x.go:12)\nt0 called from golang.org/entry/y.Y [approx.] (y.go:5)\ngolang.org/bmod/bvuln.Vuln (bvuln.go:2)\n\n"
	]
}
`,
	`
{
	"id": "GO-2022-01",
	"symbol": "VulnData.Vuln2",
	"pkg_path": "golang.org/amod/avuln",
	"mod_path": "golang.org/amod",
	"url": "https://pkg.go.dev/vuln/GO-2022-01",
	"current_version": "v1.1.3",
	FixedVersion: "v1.0.4",
	"call_stacks": [
	 "C1 called from golang.org/entry/x.X (x.go:8)\nVuln2 called from golang.org/cmod/c.C1 (c.go:13)\ngolang.org/amod/avuln.VulnData.Vuln2 (avuln.go:4)\n\n"
	]
}
`,
}

type mockClient struct {
	client.Client
	ret map[string][]*osv.Entry
}

func (mc *mockClient) GetByModule(ctx context.Context, a string) ([]*osv.Entry, error) {
	return mc.ret[a], nil
}

func runTest(t *testing.T, workspaceData, proxyData string, test func(context.Context, source.Snapshot)) {
	ws, err := fake.NewSandbox(&fake.SandboxConfig{
		Files:      fake.UnpackTxt(workspaceData),
		ProxyFiles: fake.UnpackTxt(proxyData),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()

	ctx := tests.Context(t)

	// get the module cache populated and the go.sum file at the root auto-generated.
	dir := ws.Workdir.RootURI().SpanURI().Filename()
	if err := ws.RunGoCommand(ctx, dir, "list", []string{"-mod=mod", "..."}, true); err != nil {
		t.Fatal(err)
	}

	cache := cache.New(nil)
	session := cache.NewSession(ctx)
	options := source.DefaultOptions().Clone()
	tests.DefaultOptions(options)
	session.SetOptions(options)
	envs := []string{}
	for k, v := range ws.GoEnv() {
		envs = append(envs, k+"="+v)
	}
	options.SetEnvSlice(envs)
	name := ws.RootDir()
	folder := ws.Workdir.RootURI().SpanURI()
	view, snapshot, release, err := session.NewView(ctx, name, folder, options)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	defer view.Shutdown(ctx)

	test(ctx, snapshot)
}

func sortStrs(s []string) []string {
	sort.Strings(s)
	return s
}

func pkgPaths(pkgs []*vulncheck.Package) []string {
	var r []string
	for _, p := range pkgs {
		r = append(r, p.PkgPath)
	}
	return sortStrs(r)
}

// TODO: expose this as a method of Snapshot.
func packagesCfg(ctx context.Context, snapshot source.Snapshot) *packages.Config {
	view := snapshot.View()
	viewBuildFlags := view.Options().BuildFlags
	var viewEnv []string
	if e := view.Options().EnvSlice(); e != nil {
		viewEnv = append(os.Environ(), e...)
	}
	return &packages.Config{
		// Mode will be set by cmd.Run.
		Context:    ctx,
		Tests:      true,
		BuildFlags: viewBuildFlags,
		Env:        viewEnv,
		Dir:        view.Folder().Filename(),
	}
}
