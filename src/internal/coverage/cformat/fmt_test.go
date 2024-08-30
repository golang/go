// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cformat_test

import (
	"internal/coverage"
	"internal/coverage/cformat"
	"slices"
	"strings"
	"testing"
)

func TestBasics(t *testing.T) {
	fm := cformat.NewFormatter(coverage.CtrModeAtomic)

	mku := func(stl, enl, nx uint32) coverage.CoverableUnit {
		return coverage.CoverableUnit{
			StLine:  stl,
			EnLine:  enl,
			NxStmts: nx,
		}
	}
	fn1units := []coverage.CoverableUnit{
		mku(10, 11, 2),
		mku(15, 11, 1),
	}
	fn2units := []coverage.CoverableUnit{
		mku(20, 25, 3),
		mku(30, 31, 2),
		mku(33, 40, 7),
	}
	fn3units := []coverage.CoverableUnit{
		mku(99, 100, 1),
	}
	fm.SetPackage("my/pack1")
	for k, u := range fn1units {
		fm.AddUnit("p.go", "f1", false, u, uint32(k))
	}
	for k, u := range fn2units {
		fm.AddUnit("q.go", "f2", false, u, 0)
		fm.AddUnit("q.go", "f2", false, u, uint32(k))
	}
	fm.SetPackage("my/pack2")
	for _, u := range fn3units {
		fm.AddUnit("lit.go", "f3", true, u, 0)
	}

	var b1, b2, b3, b4 strings.Builder
	if err := fm.EmitTextual(&b1); err != nil {
		t.Fatalf("EmitTextual returned %v", err)
	}
	wantText := strings.TrimSpace(`
mode: atomic
p.go:10.0,11.0 2 0
p.go:15.0,11.0 1 1
q.go:20.0,25.0 3 0
q.go:30.0,31.0 2 1
q.go:33.0,40.0 7 2
lit.go:99.0,100.0 1 0`)
	gotText := strings.TrimSpace(b1.String())
	if wantText != gotText {
		t.Errorf("emit text: got:\n%s\nwant:\n%s\n", gotText, wantText)
	}

	// Percent output with no aggregation.
	noCoverPkg := ""
	if err := fm.EmitPercent(&b2, nil, noCoverPkg, false, false); err != nil {
		t.Fatalf("EmitPercent returned %v", err)
	}
	wantPercent := strings.Fields(`
       	my/pack1		coverage: 66.7% of statements
        my/pack2		coverage: 0.0% of statements
`)
	gotPercent := strings.Fields(b2.String())
	if !slices.Equal(wantPercent, gotPercent) {
		t.Errorf("emit percent: got:\n%+v\nwant:\n%+v\n",
			gotPercent, wantPercent)
	}

	// Percent mode with aggregation.
	withCoverPkg := " in ./..."
	if err := fm.EmitPercent(&b3, nil, withCoverPkg, false, true); err != nil {
		t.Fatalf("EmitPercent returned %v", err)
	}
	wantPercent = strings.Fields(`
		coverage: 62.5% of statements in ./...
`)
	gotPercent = strings.Fields(b3.String())
	if !slices.Equal(wantPercent, gotPercent) {
		t.Errorf("emit percent: got:\n%+v\nwant:\n%+v\n",
			gotPercent, wantPercent)
	}

	if err := fm.EmitFuncs(&b4); err != nil {
		t.Fatalf("EmitFuncs returned %v", err)
	}
	wantFuncs := strings.TrimSpace(`
p.go:10:	f1		33.3%
q.go:20:	f2		75.0%
total		(statements)	62.5%`)
	gotFuncs := strings.TrimSpace(b4.String())
	if wantFuncs != gotFuncs {
		t.Errorf("emit funcs: got:\n%s\nwant:\n%s\n", gotFuncs, wantFuncs)
	}
	if false {
		t.Logf("text is %s\n", b1.String())
		t.Logf("perc is %s\n", b2.String())
		t.Logf("perc2 is %s\n", b3.String())
		t.Logf("funcs is %s\n", b4.String())
	}

	// Percent output with specific packages selected.
	{
		var b strings.Builder
		selpkgs := []string{"foo/bar", "my/pack1"}
		if err := fm.EmitPercent(&b, selpkgs, noCoverPkg, false, false); err != nil {
			t.Fatalf("EmitPercent returned %v", err)
		}
		wantPercent := strings.Fields(`
       	my/pack1		coverage: 66.7% of statements
`)
		gotPercent := strings.Fields(b.String())
		if !slices.Equal(wantPercent, gotPercent) {
			t.Errorf("emit percent: got:\n%+v\nwant:\n%+v\n",
				gotPercent, wantPercent)
		}
	}

}

func TestEmptyPackages(t *testing.T) {

	fm := cformat.NewFormatter(coverage.CtrModeAtomic)
	fm.SetPackage("my/pack1")
	fm.SetPackage("my/pack2")

	// No aggregation.
	{
		var b strings.Builder
		noCoverPkg := ""
		if err := fm.EmitPercent(&b, nil, noCoverPkg, true, false); err != nil {
			t.Fatalf("EmitPercent returned %v", err)
		}
		wantPercent := strings.Fields(`
       	my/pack1 coverage:	[no statements]
        my/pack2 coverage:	[no statements]
`)
		gotPercent := strings.Fields(b.String())
		if !slices.Equal(wantPercent, gotPercent) {
			t.Errorf("emit percent: got:\n%+v\nwant:\n%+v\n",
				gotPercent, wantPercent)
		}
	}

	// With aggregation.
	{
		var b strings.Builder
		noCoverPkg := ""
		if err := fm.EmitPercent(&b, nil, noCoverPkg, true, true); err != nil {
			t.Fatalf("EmitPercent returned %v", err)
		}
		wantPercent := strings.Fields(`
       	coverage:	[no statements]
`)
		gotPercent := strings.Fields(b.String())
		if !slices.Equal(wantPercent, gotPercent) {
			t.Errorf("emit percent: got:\n%+v\nwant:\n%+v\n",
				gotPercent, wantPercent)
		}
	}
}
