// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"
)

func newSolver(t *testing.T) *Solver {
	var s Solver
	s.SetTrace(t.Output())
	return &s
}

func TestSolver(t *testing.T) {
	t.Run("constant assignment", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		s.Assign(v1, Int(10))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{v1: Int(10)})
	})

	t.Run("simple dependency", func(t *testing.T) {
		testBinaryOp(t, 10, &BinExpr{
			Op: OpTimes,
			X:  Variable("v1"),
			Y:  Int(2),
		}, Int(20))
	})

	t.Run("division", func(t *testing.T) {
		testBinaryOp(t, 16, &BinExpr{
			Op: OpDiv,
			X:  Variable("v1"),
			Y:  Int(4),
		}, Int(4))
	})

	t.Run("cycle detection", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")
		s.Assign(v1, Variable(v2))
		s.Assign(v2, Variable(v1))

		err := solverError(t, s)
		if !strings.Contains(err.Error(), "cyclic requirements") {
			t.Fatalf("expected cycle error, got %v", err)
		}
	})

	t.Run("multiple assignment conflicting values", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		s.Assign(v1, Int(10))
		s.Assign(v1, Int(20))

		err := solverError(t, s)
		if !strings.Contains(err.Error(), "no solutions") {
			t.Fatalf("expected no solutions error, got %v", err)
		}
	})

	t.Run("multiple assignment same value", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		s.Assign(v1, Int(10))
		s.Assign(v1, Int(10))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{v1: Int(10)})
	})

	t.Run("swidth times int", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")
		s.Assign(v1, mkWidth(1, 2))
		s.Assign(v2, &BinExpr{
			Op: OpTimes,
			X:  v1,
			Y:  Int(4),
		})

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			v1: mkWidth(1, 2),
			v2: mkWidth(2, 1),
		})
	})

	t.Run("int times swidth", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")
		s.Assign(v1, mkWidth(1, 2))
		s.Assign(v2, &BinExpr{
			Op: OpTimes,
			X:  Int(4),
			Y:  v1,
		})

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			v1: mkWidth(1, 2),
			v2: mkWidth(2, 1),
		})
	})

	t.Run("swidth div int", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")
		s.Assign(v1, mkWidth(1, 2))
		s.Assign(v2, &BinExpr{
			Op: OpDiv,
			X:  v1,
			Y:  Int(2),
		})

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			v1: mkWidth(1, 2),
			v2: mkWidth(1, 4),
		})
	})
}

func testBinaryOp(t *testing.T, v1Val Int, v2Expr Expr, expectedV2Val any) {
	t.Helper()
	s := newSolver(t)
	v1 := Variable("v1")
	v2 := Variable("v2")
	s.Assign(v1, v1Val)
	s.Assign(v2, v2Expr)

	sol := uniqueSolution(t, s)
	bCheck(t, sol, map[Variable]any{v1: v1Val, v2: expectedV2Val})
}

func TestComparisons(t *testing.T) {
	t.Run("greater than", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")
		s.Assign(v1, Int(20))
		s.Assign(v2, Int(10))

		// Add comparison v1 > v2 and assert it must be true
		s.Assert(&BinExpr{Op: OpGreaterThan, X: v1, Y: v2})

		uniqueSolution(t, s)
	})

	t.Run("less than", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")
		s.Assign(v1, Int(10))
		s.Assign(v2, Int(20))

		s.Assert(&BinExpr{Op: OpLessThan, X: v1, Y: v2})

		uniqueSolution(t, s)
	})
}

func TestSolveShape(t *testing.T) {
	t.Run("scalar Int32", func(t *testing.T) {
		s := newSolver(t)
		s.Assign("x", mustParseExpr(t, "Int32"))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"x": Basic{"int", 32},
		})
	})

	t.Run("vector Int32x4", func(t *testing.T) {
		s := newSolver(t)
		s.Assign("x", mustParseExpr(t, "Int32x4"))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"x": Vector{Basic{"int", 32}, Int(128)},
		})
	})

	t.Run("scalar symbolic {xB}{xN}", func(t *testing.T) {
		s := newSolver(t)
		s.Assign("x", mustParseExpr(t, "{xB}{xN}"))
		s.Assign("xB", &Literal{"int"})
		s.Assign("xN", Int(32))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"x":  Basic{"int", 32},
			"xB": "int",
			"xN": Int(32),
		})
	})

	vectorElem := MakeField[Vector]("Elem")
	basicBase := MakeField[Basic]("Base")
	basicBits := MakeField[Basic]("Bits")
	vectorWidth := MakeField[Vector]("Width")
	assignVector := func(s *Solver, v Variable, e Expr) {
		x := s.Assign(v, e)
		s.Assign(v+"B", basicBase.Apply(vectorElem.Apply(x)))
		xN := s.Assign(v+"N", basicBits.Apply(vectorElem.Apply(x)))
		xW := s.Assign(v+"W", vectorWidth.Apply(x))
		s.Assign(v+"L", &BinExpr{Op: OpDiv, X: xW, Y: xN})
	}

	t.Run("derived scalable vector with lane count", func(t *testing.T) {
		s := newSolver(t)
		assignVector(s, "x", mustParseExpr(t, "Int32s"))
		s.Assign("y", mustParseExpr(t, "{xB}{xN*2}x{xL/2}"))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"y":  Vector{Basic{"int", 64}, mkWidth(1, 1)},
			"xW": mkWidth(1, 1),
			"xL": mkWidth(1, 32),
		})
	})

	t.Run("derived scalable vector with width", func(t *testing.T) {
		s := newSolver(t)
		assignVector(s, "x", mustParseExpr(t, "Int32s"))
		s.Assign("y", mustParseExpr(t, "{xB}{xN*2}w{xW}"))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"y":  Vector{Basic{"int", 64}, mkWidth(1, 1)},
			"xW": mkWidth(1, 1),
		})
	})

	t.Run("width rounding", func(t *testing.T) {
		s := newSolver(t)
		assignVector(s, "x", mustParseExpr(t, "Int64x2"))
		s.Assign("y", mustParseExpr(t, "{xB}{xN/2}x{xL}"))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"y": Vector{Basic{"int", 32}, Int(128)},
		})
	})

	t.Run("domain limits", func(t *testing.T) {
		s := newSolver(t)
		s.Declare("x", []any{1, 2})
		s.Declare("y", []any{1, 2})
		s.Assign("y", mustParseExpr(t, "x*2"))

		sol := uniqueSolution(t, s)
		bCheck(t, sol, map[Variable]any{
			"x": Int(1), "y": Int(2),
		})
	})

	t.Run("non-scalable width", func(t *testing.T) {
		s := newSolver(t)
		assignVector(s, "x", mustParseExpr(t, "Int32s"))
		s.Assign("y", mustParseExpr(t, "Int16x{xL}"))

		err := solverError(t, s)
		if !strings.Contains(err.Error(), "invalid width") {
			t.Fatalf("expected invalid width error, got: %v", err)
		}
	})
}

func TestEnumerator(t *testing.T) {
	t.Run("simple enumeration", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")

		s.Declare(v1, []any{1, 2, 3, 4})

		// Assert v2 = v1 * 2
		s.Assign(v2, &BinExpr{Op: OpTimes, X: v1, Y: Int(2)})

		sols := allSolutions(s)

		if len(sols) != 4 {
			t.Errorf("expected 4 solutions, got %d", len(sols))
		}

		// Verify that each solution maps v2 to v1*2
		for _, sol := range sols {
			m := bmap(sol)
			v1Val := m[v1].(Int)
			v2Val := m[v2].(Int)
			if v2Val != v1Val*2 {
				t.Errorf("solution %v violates v2 = v1*2", m)
			}
		}
	})

	t.Run("comparison constraints enumeration", func(t *testing.T) {
		s := newSolver(t)
		v1 := Variable("v1")
		v2 := Variable("v2")

		s.Declare(v1, []any{1, 2, 3, 4, 5})

		// v2 = v1 * 2
		s.Assign(v2, &BinExpr{Op: OpTimes, X: v1, Y: Int(2)})

		// v1 > 2
		s.Assert(&BinExpr{Op: OpGreaterThan, X: v1, Y: Int(2)})

		// v2 < 10
		s.Assert(&BinExpr{Op: OpLessThan, X: v2, Y: Int(10)})

		sols := allSolutions(s)

		// Solutions should be v1=3, v1=4. So 2 solutions
		if len(sols) != 2 {
			t.Errorf("expected 2 solutions, got %d", len(sols))
		}
	})
}

func mustParseExpr(t *testing.T, x string) Expr {
	t.Helper()
	e, err := ParseExpr(x)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func allSolutions(s *Solver) []*Bindings {
	var sols []*Bindings
	for b, err := range s.Solve() {
		if err != nil {
			panic(err) // tests expect valid execution when enumerating
		}
		sols = append(sols, b)
	}
	return sols
}

func uniqueSolution(t *testing.T, s *Solver) *Bindings {
	t.Helper()
	var sols []*Bindings
	for b, err := range s.Solve() {
		if err != nil {
			t.Fatalf("solve failed: %v", err)
		}
		sols = append(sols, b)
		if len(sols) >= 20 {
			// Stop before we go too deep
			t.Fatalf("expected exactly one solution, got >= 20")
		}
	}
	if len(sols) != 1 {
		t.Errorf("expected exactly one solution, got %d", len(sols))
		for _, sol := range sols {
			t.Errorf("  %s", sol)
		}
		t.FailNow()
	}
	return sols[0]
}

func solverError(t *testing.T, s *Solver) error {
	t.Helper()
	for soln, err := range s.Solve() {
		if err != nil {
			return err
		}
		t.Fatalf("expected solver error, but got solution:\n%s", soln)
	}
	return fmt.Errorf("no solutions")
}

// bmap converts a Bindings to a map.
func bmap(b *Bindings) map[Variable]any {
	return maps.Collect(b.All())
}

// bCheck fails t if got[v] != want[v] for any keys in want.
func bCheck(t *testing.T, got *Bindings, want map[Variable]any) {
	t.Helper()
	var keys []Variable
	for k := range want {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	var mismatches []string
	for _, k := range keys {
		wantVal := want[k]
		gotVal := got.Get(k)
		if gotVal != wantVal {
			mismatches = append(mismatches, fmt.Sprintf("  %s: got %v (%T), want %v (%T)", k, gotVal, gotVal, wantVal, wantVal))
		}
	}
	if len(mismatches) > 0 {
		t.Fatalf("solution mismatch:\n%s", strings.Join(mismatches, "\n"))
	}
}
