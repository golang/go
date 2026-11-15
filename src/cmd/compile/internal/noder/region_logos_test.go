// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"testing"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"internal/buildcfg"
)

func enableRegionBlockExperiment(t *testing.T) {
	t.Helper()
	old := buildcfg.Experiment.LogosArena
	buildcfg.Experiment.LogosArena = true
	t.Cleanup(func() { buildcfg.Experiment.LogosArena = old })
}

func blockOpenCall() *ir.CallExpr {
	fun := ir.NewNameAt(src.NoXPos, types.NewPkg("logos/arena", "arena").Lookup("Open"), types.NewSignature(nil, nil, nil))
	fun.Class = ir.PFUNC
	return ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, fun, nil)
}

func blockPlainStmt() ir.Node {
	x := ir.NewNameAt(src.NoXPos, types.NewPkg("main", "main").Lookup("x"), nil)
	x.Class = ir.PAUTO
	return ir.NewAssignStmt(src.NoXPos, x, x)
}

func TestLogosKeepRegionBlockOff(t *testing.T) {
	old := buildcfg.Experiment.LogosArena
	buildcfg.Experiment.LogosArena = false
	t.Cleanup(func() { buildcfg.Experiment.LogosArena = old })
	if logosKeepRegionBlock([]ir.Node{blockOpenCall()}) {
		t.Fatalf("logosKeepRegionBlock = true with experiment off")
	}
}

func TestLogosKeepRegionBlock(t *testing.T) {
	enableRegionBlockExperiment(t)

	if !logosKeepRegionBlock([]ir.Node{blockOpenCall()}) {
		t.Fatalf("logosKeepRegionBlock(direct Open call) = false, want true")
	}
	oas := ir.NewAssignStmt(src.NoXPos, blockPlainStmt().(*ir.AssignStmt).X, blockOpenCall())
	if !logosKeepRegionBlock([]ir.Node{blockPlainStmt(), oas}) {
		t.Fatalf("logosKeepRegionBlock(Open in assignment) = false, want true")
	}
	nested := ir.NewBlockStmt(src.NoXPos, []ir.Node{blockOpenCall()})
	if logosKeepRegionBlock([]ir.Node{nested}) {
		t.Fatalf("logosKeepRegionBlock(Open only in nested block) = true, want false")
	}
	if logosKeepRegionBlock([]ir.Node{blockPlainStmt()}) {
		t.Fatalf("logosKeepRegionBlock(no Open) = true, want false")
	}
	if logosKeepRegionBlock(nil) {
		t.Fatalf("logosKeepRegionBlock(nil) = true, want false")
	}
	inner := testRegionFunc([]ir.Node{blockOpenCall()})
	cl := &ir.ClosureExpr{Func: inner}
	if logosKeepRegionBlock([]ir.Node{cl}) {
		t.Fatalf("logosKeepRegionBlock(Open only in closure body) = true, want false")
	}
}

func testRegionFunc(body []ir.Node) *ir.Func {
	fn := ir.NewFunc(src.NoXPos, src.NoXPos, types.NewPkg("main", "main").Lookup("lit"), types.NewSignature(nil, nil, nil))
	fn.Body = body
	return fn
}
