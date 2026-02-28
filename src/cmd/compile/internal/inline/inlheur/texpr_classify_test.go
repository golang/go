// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"cmd/internal/sys"
	"go/constant"
	"testing"
)

var pos src.XPos
var local *types.Pkg
var f *ir.Func

func init() {
	types.PtrSize = 8
	types.RegSize = 8
	types.MaxWidth = 1 << 50
	base.Ctxt = &obj.Link{Arch: &obj.LinkArch{Arch: &sys.Arch{Alignment: 1, CanMergeLoads: true}}}

	typecheck.InitUniverse()
	local = types.NewPkg("", "")
	fsym := &types.Sym{
		Pkg:  types.NewPkg("my/import/path", "path"),
		Name: "function",
	}
	f = ir.NewFunc(src.NoXPos, src.NoXPos, fsym, nil)
}

type state struct {
	ntab map[string]*ir.Name
}

func mkstate() *state {
	return &state{
		ntab: make(map[string]*ir.Name),
	}
}

func bin(x ir.Node, op ir.Op, y ir.Node) ir.Node {
	return ir.NewBinaryExpr(pos, op, x, y)
}

func conv(x ir.Node, t *types.Type) ir.Node {
	return ir.NewConvExpr(pos, ir.OCONV, t, x)
}

func logical(x ir.Node, op ir.Op, y ir.Node) ir.Node {
	return ir.NewLogicalExpr(pos, op, x, y)
}

func un(op ir.Op, x ir.Node) ir.Node {
	return ir.NewUnaryExpr(pos, op, x)
}

func liti(i int64) ir.Node {
	return ir.NewBasicLit(pos, types.Types[types.TINT64], constant.MakeInt64(i))
}

func lits(s string) ir.Node {
	return ir.NewBasicLit(pos, types.Types[types.TSTRING], constant.MakeString(s))
}

func (s *state) nm(name string, t *types.Type) *ir.Name {
	if n, ok := s.ntab[name]; ok {
		if n.Type() != t {
			panic("bad")
		}
		return n
	}
	sym := local.Lookup(name)
	nn := ir.NewNameAt(pos, sym, t)
	s.ntab[name] = nn
	return nn
}

func (s *state) nmi64(name string) *ir.Name {
	return s.nm(name, types.Types[types.TINT64])
}

func (s *state) nms(name string) *ir.Name {
	return s.nm(name, types.Types[types.TSTRING])
}

func TestClassifyIntegerCompare(t *testing.T) {

	// (n < 10 || n > 100) && (n >= 12 || n <= 99 || n != 101)
	s := mkstate()
	nn := s.nmi64("n")
	nlt10 := bin(nn, ir.OLT, liti(10))         // n < 10
	ngt100 := bin(nn, ir.OGT, liti(100))       // n > 100
	nge12 := bin(nn, ir.OGE, liti(12))         // n >= 12
	nle99 := bin(nn, ir.OLE, liti(99))         // n < 10
	nne101 := bin(nn, ir.ONE, liti(101))       // n != 101
	noror1 := logical(nlt10, ir.OOROR, ngt100) // n < 10 || n > 100
	noror2 := logical(nge12, ir.OOROR, nle99)  // n >= 12 || n <= 99
	noror3 := logical(noror2, ir.OOROR, nne101)
	nandand := typecheck.Expr(logical(noror1, ir.OANDAND, noror3))

	wantv := true
	v := ShouldFoldIfNameConstant(nandand, []*ir.Name{nn})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", nandand, wantv, v)
	}
}

func TestClassifyStringCompare(t *testing.T) {

	// s != "foo" && s < "ooblek" && s > "plarkish"
	s := mkstate()
	nn := s.nms("s")
	snefoo := bin(nn, ir.ONE, lits("foo"))     // s != "foo"
	sltoob := bin(nn, ir.OLT, lits("ooblek"))  // s < "ooblek"
	sgtpk := bin(nn, ir.OGT, lits("plarkish")) // s > "plarkish"
	nandand := logical(snefoo, ir.OANDAND, sltoob)
	top := typecheck.Expr(logical(nandand, ir.OANDAND, sgtpk))

	wantv := true
	v := ShouldFoldIfNameConstant(top, []*ir.Name{nn})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", top, wantv, v)
	}
}

func TestClassifyIntegerArith(t *testing.T) {
	// n+1 ^ n-3 * n/2 + n<<9 + n>>2 - n&^7

	s := mkstate()
	nn := s.nmi64("n")
	np1 := bin(nn, ir.OADD, liti(1))     // n+1
	nm3 := bin(nn, ir.OSUB, liti(3))     // n-3
	nd2 := bin(nn, ir.ODIV, liti(2))     // n/2
	nls9 := bin(nn, ir.OLSH, liti(9))    // n<<9
	nrs2 := bin(nn, ir.ORSH, liti(2))    // n>>2
	nan7 := bin(nn, ir.OANDNOT, liti(7)) // n&^7
	c1xor := bin(np1, ir.OXOR, nm3)
	c2mul := bin(c1xor, ir.OMUL, nd2)
	c3add := bin(c2mul, ir.OADD, nls9)
	c4add := bin(c3add, ir.OADD, nrs2)
	c5sub := bin(c4add, ir.OSUB, nan7)
	top := typecheck.Expr(c5sub)

	wantv := true
	v := ShouldFoldIfNameConstant(top, []*ir.Name{nn})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", top, wantv, v)
	}
}

func TestClassifyAssortedShifts(t *testing.T) {

	s := mkstate()
	nn := s.nmi64("n")
	badcases := []ir.Node{
		bin(liti(3), ir.OLSH, nn), // 3<<n
		bin(liti(7), ir.ORSH, nn), // 7>>n
	}
	for _, bc := range badcases {
		wantv := false
		v := ShouldFoldIfNameConstant(typecheck.Expr(bc), []*ir.Name{nn})
		if v != wantv {
			t.Errorf("wanted shouldfold(%v) %v, got %v", bc, wantv, v)
		}
	}
}

func TestClassifyFloat(t *testing.T) {
	// float32(n) + float32(10)
	s := mkstate()
	nn := s.nm("n", types.Types[types.TUINT32])
	f1 := conv(nn, types.Types[types.TFLOAT32])
	f2 := conv(liti(10), types.Types[types.TFLOAT32])
	add := bin(f1, ir.OADD, f2)

	wantv := false
	v := ShouldFoldIfNameConstant(typecheck.Expr(add), []*ir.Name{nn})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", add, wantv, v)
	}
}

func TestMultipleNamesAllUsed(t *testing.T) {
	// n != 101 && m < 2
	s := mkstate()
	nn := s.nmi64("n")
	nm := s.nmi64("m")
	nne101 := bin(nn, ir.ONE, liti(101)) // n != 101
	mlt2 := bin(nm, ir.OLT, liti(2))     // m < 2
	nandand := typecheck.Expr(logical(nne101, ir.OANDAND, mlt2))

	// all names used
	wantv := true
	v := ShouldFoldIfNameConstant(nandand, []*ir.Name{nn, nm})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", nandand, wantv, v)
	}

	// not all names used
	wantv = false
	v = ShouldFoldIfNameConstant(nne101, []*ir.Name{nn, nm})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", nne101, wantv, v)
	}

	// other names used.
	np := s.nmi64("p")
	pne0 := bin(np, ir.ONE, liti(101)) // p != 0
	noror := logical(nandand, ir.OOROR, pne0)
	wantv = false
	v = ShouldFoldIfNameConstant(noror, []*ir.Name{nn, nm})
	if v != wantv {
		t.Errorf("wanted shouldfold(%v) %v, got %v", noror, wantv, v)
	}
}
