// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"testing"
)

const (
	SetOrder             = "SetOrder"
	SetOrder_Fail        = "SetOrder_Fail"
	SetOrderOrEqual      = "SetOrderOrEqual"
	SetOrderOrEqual_Fail = "SetOrderOrEqual_Fail"
	Ordered              = "Ordered"
	Ordered_Fail         = "Ordered_Fail"
	OrderedOrEqual       = "OrderedOrEqual"
	OrderedOrEqual_Fail  = "OrderedOrEqual_Fail"
	SetEqual             = "SetEqual"
	SetEqual_Fail        = "SetEqual_Fail"
	Equal                = "Equal"
	Equal_Fail           = "Equal_Fail"
	SetNonEqual          = "SetNonEqual"
	SetNonEqual_Fail     = "SetNonEqual_Fail"
	NonEqual             = "NonEqual"
	NonEqual_Fail        = "NonEqual_Fail"
	Checkpoint           = "Checkpoint"
	Undo                 = "Undo"
)

type posetTestOp struct {
	typ  string
	a, b int
}

func vconst(i int) int {
	if i < -128 || i >= 128 {
		panic("invalid const")
	}
	return 1000 + 128 + i
}

func vconst2(i int) int {
	if i < -128 || i >= 128 {
		panic("invalid const")
	}
	return 1000 + 256 + i
}

func testPosetOps(t *testing.T, unsigned bool, ops []posetTestOp) {
	var v [1512]*Value
	for i := range v {
		v[i] = new(Value)
		v[i].ID = ID(i)
		if i >= 1000 && i < 1256 {
			v[i].Op = OpConst64
			v[i].AuxInt = int64(i - 1000 - 128)
		}
		if i >= 1256 && i < 1512 {
			v[i].Op = OpConst64
			v[i].AuxInt = int64(i - 1000 - 256)
		}
	}

	po := newPoset()
	po.SetUnsigned(unsigned)
	for idx, op := range ops {
		t.Logf("op%d%v", idx, op)
		switch op.typ {
		case SetOrder:
			if !po.SetOrder(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case SetOrder_Fail:
			if po.SetOrder(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case SetOrderOrEqual:
			if !po.SetOrderOrEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case SetOrderOrEqual_Fail:
			if po.SetOrderOrEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case Ordered:
			if !po.Ordered(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case Ordered_Fail:
			if po.Ordered(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case OrderedOrEqual:
			if !po.OrderedOrEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case OrderedOrEqual_Fail:
			if po.OrderedOrEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case SetEqual:
			if !po.SetEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case SetEqual_Fail:
			if po.SetEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case Equal:
			if !po.Equal(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case Equal_Fail:
			if po.Equal(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case SetNonEqual:
			if !po.SetNonEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case SetNonEqual_Fail:
			if po.SetNonEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case NonEqual:
			if !po.NonEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v failed", idx, op)
			}
		case NonEqual_Fail:
			if po.NonEqual(v[op.a], v[op.b]) {
				t.Errorf("FAILED: op%d%v passed", idx, op)
			}
		case Checkpoint:
			po.Checkpoint()
		case Undo:
			t.Log("Undo stack", po.undo)
			po.Undo()
		default:
			panic("unimplemented")
		}

		if false {
			po.DotDump(fmt.Sprintf("op%d.dot", idx), fmt.Sprintf("Last op: %v", op))
		}

		po.CheckIntegrity()
	}

	// Check that the poset is completely empty
	if err := po.CheckEmpty(); err != nil {
		t.Error(err)
	}
}

func TestPoset(t *testing.T) {
	testPosetOps(t, false, []posetTestOp{
		{Ordered_Fail, 123, 124},

		// Dag #0: 100<101
		{Checkpoint, 0, 0},
		{SetOrder, 100, 101},
		{Ordered, 100, 101},
		{Ordered_Fail, 101, 100},
		{SetOrder_Fail, 101, 100},
		{SetOrder, 100, 101}, // repeat
		{NonEqual, 100, 101},
		{NonEqual, 101, 100},
		{SetEqual_Fail, 100, 101},

		// Dag #1: 4<=7<12
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 4, 7},
		{OrderedOrEqual, 4, 7},
		{SetOrder, 7, 12},
		{Ordered, 7, 12},
		{Ordered, 4, 12},
		{Ordered_Fail, 12, 4},
		{NonEqual, 4, 12},
		{NonEqual, 12, 4},
		{NonEqual_Fail, 4, 100},
		{OrderedOrEqual, 4, 12},
		{OrderedOrEqual_Fail, 12, 4},
		{OrderedOrEqual, 4, 7},
		{OrderedOrEqual_Fail, 7, 4},

		// Dag #1: 1<4<=7<12
		{Checkpoint, 0, 0},
		{SetOrder, 1, 4},
		{Ordered, 1, 4},
		{Ordered, 1, 12},
		{Ordered_Fail, 12, 1},

		// Dag #1: 1<4<=7<12, 6<7
		{Checkpoint, 0, 0},
		{SetOrder, 6, 7},
		{Ordered, 6, 7},
		{Ordered, 6, 12},
		{SetOrder_Fail, 7, 4},
		{SetOrder_Fail, 7, 6},
		{SetOrder_Fail, 7, 1},

		// Dag #1: 1<4<=7<12, 1<6<7
		{Checkpoint, 0, 0},
		{Ordered_Fail, 1, 6},
		{SetOrder, 1, 6},
		{Ordered, 1, 6},
		{SetOrder_Fail, 6, 1},

		// Dag #1: 1<4<=7<12, 1<4<6<7
		{Checkpoint, 0, 0},
		{Ordered_Fail, 4, 6},
		{Ordered_Fail, 4, 7},
		{SetOrder, 4, 6},
		{Ordered, 4, 6},
		{OrderedOrEqual, 4, 6},
		{Ordered, 4, 7},
		{OrderedOrEqual, 4, 7},
		{SetOrder_Fail, 6, 4},
		{Ordered_Fail, 7, 6},
		{Ordered_Fail, 7, 4},
		{OrderedOrEqual_Fail, 7, 6},
		{OrderedOrEqual_Fail, 7, 4},

		// Merge: 1<4<6, 4<=7<12, 6<101
		{Checkpoint, 0, 0},
		{Ordered_Fail, 6, 101},
		{SetOrder, 6, 101},
		{Ordered, 6, 101},
		{Ordered, 1, 101},

		// Merge: 1<4<6, 4<=7<12, 6<100<101
		{Checkpoint, 0, 0},
		{Ordered_Fail, 6, 100},
		{SetOrder, 6, 100},
		{Ordered, 1, 100},

		// Undo: 1<4<6<7<12, 6<101
		{Ordered, 100, 101},
		{Undo, 0, 0},
		{Ordered, 100, 101},
		{Ordered_Fail, 6, 100},
		{Ordered, 6, 101},
		{Ordered, 1, 101},

		// Undo: 1<4<6<7<12, 100<101
		{Undo, 0, 0},
		{Ordered_Fail, 1, 100},
		{Ordered_Fail, 1, 101},
		{Ordered_Fail, 6, 100},
		{Ordered_Fail, 6, 101},

		// Merge: 1<4<6<7<12, 6<100<101
		{Checkpoint, 0, 0},
		{Ordered, 100, 101},
		{SetOrder, 6, 100},
		{Ordered, 6, 100},
		{Ordered, 6, 101},
		{Ordered, 1, 101},

		// Undo 2 times: 1<4<7<12, 1<6<7
		{Undo, 0, 0},
		{Undo, 0, 0},
		{Ordered, 1, 6},
		{Ordered, 4, 12},
		{Ordered_Fail, 4, 6},
		{SetOrder_Fail, 6, 1},

		// Undo 2 times: 1<4<7<12
		{Undo, 0, 0},
		{Undo, 0, 0},
		{Ordered, 1, 12},
		{Ordered, 7, 12},
		{Ordered_Fail, 1, 6},
		{Ordered_Fail, 6, 7},
		{Ordered, 100, 101},
		{Ordered_Fail, 1, 101},

		// Undo: 4<7<12
		{Undo, 0, 0},
		{Ordered_Fail, 1, 12},
		{Ordered_Fail, 1, 4},
		{Ordered, 4, 12},
		{Ordered, 100, 101},

		// Undo: 100<101
		{Undo, 0, 0},
		{Ordered_Fail, 4, 7},
		{Ordered_Fail, 7, 12},
		{Ordered, 100, 101},

		// Recreated DAG #1 from scratch, reusing same nodes.
		// This also stresses that Undo has done its job correctly.
		// DAG: 1<2<(5|6), 101<102<(105|106<107)
		{Checkpoint, 0, 0},
		{SetOrder, 101, 102},
		{SetOrder, 102, 105},
		{SetOrder, 102, 106},
		{SetOrder, 106, 107},
		{SetOrder, 1, 2},
		{SetOrder, 2, 5},
		{SetOrder, 2, 6},
		{SetEqual_Fail, 1, 6},
		{SetEqual_Fail, 107, 102},

		// Now Set 2 == 102
		// New DAG: (1|101)<2==102<(5|6|105|106<107)
		{Checkpoint, 0, 0},
		{SetEqual, 2, 102},
		{Equal, 2, 102},
		{SetEqual, 2, 102},         // trivially pass
		{SetNonEqual_Fail, 2, 102}, // trivially fail
		{Ordered, 1, 107},
		{Ordered, 101, 6},
		{Ordered, 101, 105},
		{Ordered, 2, 106},
		{Ordered, 102, 6},

		// Undo SetEqual
		{Undo, 0, 0},
		{Equal_Fail, 2, 102},
		{Ordered_Fail, 2, 102},
		{Ordered_Fail, 1, 107},
		{Ordered_Fail, 101, 6},
		{Checkpoint, 0, 0},
		{SetEqual, 2, 100},
		{Ordered, 1, 107},
		{Ordered, 100, 6},

		// SetEqual with new node
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetEqual, 2, 400},
		{SetEqual, 401, 2},
		{Equal, 400, 401},
		{Ordered, 1, 400},
		{Ordered, 400, 6},
		{Ordered, 1, 401},
		{Ordered, 401, 6},
		{Ordered_Fail, 2, 401},

		// SetEqual unseen nodes and then connect
		{Checkpoint, 0, 0},
		{SetEqual, 500, 501},
		{SetEqual, 102, 501},
		{Equal, 500, 102},
		{Ordered, 501, 106},
		{Ordered, 100, 500},
		{SetEqual, 500, 501},
		{Ordered_Fail, 500, 501},
		{Ordered_Fail, 102, 501},

		// SetNonEqual relations
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetNonEqual, 600, 601},
		{NonEqual, 600, 601},
		{SetNonEqual, 601, 602},
		{NonEqual, 601, 602},
		{NonEqual_Fail, 600, 602}, // non-transitive
		{SetEqual_Fail, 601, 602},

		// Undo back to beginning, leave the poset empty
		{Undo, 0, 0},
		{Undo, 0, 0},
		{Undo, 0, 0},
		{Undo, 0, 0},
	})
}

func TestPosetStrict(t *testing.T) {

	testPosetOps(t, false, []posetTestOp{
		{Checkpoint, 0, 0},
		// Build: 20!=30, 10<20<=30<40. The 20<=30 will become 20<30.
		{SetNonEqual, 20, 30},
		{SetOrder, 10, 20},
		{SetOrderOrEqual, 20, 30}, // this is affected by 20!=30
		{SetOrder, 30, 40},

		{Ordered, 10, 30},
		{Ordered, 20, 30},
		{Ordered, 10, 40},
		{OrderedOrEqual, 10, 30},
		{OrderedOrEqual, 20, 30},
		{OrderedOrEqual, 10, 40},

		{Undo, 0, 0},

		// Now do the opposite: first build the DAG and then learn non-equality
		{Checkpoint, 0, 0},
		{SetOrder, 10, 20},
		{SetOrderOrEqual, 20, 30}, // this is affected by 20!=30
		{SetOrder, 30, 40},

		{Ordered, 10, 30},
		{Ordered_Fail, 20, 30},
		{Ordered, 10, 40},
		{OrderedOrEqual, 10, 30},
		{OrderedOrEqual, 20, 30},
		{OrderedOrEqual, 10, 40},

		{Checkpoint, 0, 0},
		{SetNonEqual, 20, 30},
		{Ordered, 10, 30},
		{Ordered, 20, 30},
		{Ordered, 10, 40},
		{OrderedOrEqual, 10, 30},
		{OrderedOrEqual, 20, 30},
		{OrderedOrEqual, 10, 40},
		{Undo, 0, 0},

		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 30, 35},
		{OrderedOrEqual, 20, 35},
		{Ordered_Fail, 20, 35},
		{SetNonEqual, 20, 35},
		{Ordered, 20, 35},
		{Undo, 0, 0},

		// Learn <= and >=
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 50, 60},
		{SetOrderOrEqual, 60, 50},
		{OrderedOrEqual, 50, 60},
		{OrderedOrEqual, 60, 50},
		{Ordered_Fail, 50, 60},
		{Ordered_Fail, 60, 50},
		{Equal, 50, 60},
		{Equal, 60, 50},
		{NonEqual_Fail, 50, 60},
		{NonEqual_Fail, 60, 50},
		{Undo, 0, 0},

		{Undo, 0, 0},
	})
}

func TestPosetCollapse(t *testing.T) {
	testPosetOps(t, false, []posetTestOp{
		{Checkpoint, 0, 0},
		// Create a complex graph of <= relations among nodes between 10 and 25.
		{SetOrderOrEqual, 10, 15},
		{SetOrderOrEqual, 15, 20},
		{SetOrderOrEqual, 20, vconst(20)},
		{SetOrderOrEqual, vconst(20), 25},
		{SetOrderOrEqual, 10, 12},
		{SetOrderOrEqual, 12, 16},
		{SetOrderOrEqual, 16, vconst(20)},
		{SetOrderOrEqual, 10, 17},
		{SetOrderOrEqual, 17, 25},
		{SetOrderOrEqual, 15, 18},
		{SetOrderOrEqual, 18, vconst(20)},
		{SetOrderOrEqual, 15, 19},
		{SetOrderOrEqual, 19, 25},

		// These are other paths not part of the main collapsing path
		{SetOrderOrEqual, 10, 11},
		{SetOrderOrEqual, 11, 26},
		{SetOrderOrEqual, 13, 25},
		{SetOrderOrEqual, 100, 25},
		{SetOrderOrEqual, 101, 15},
		{SetOrderOrEqual, 102, 10},
		{SetOrderOrEqual, 25, 103},
		{SetOrderOrEqual, 20, 104},

		{Checkpoint, 0, 0},
		// Collapse everything by setting 10 >= 25: this should make everything equal
		{SetOrderOrEqual, 25, 10},

		// Check that all nodes are pairwise equal now
		{Equal, 10, 12},
		{Equal, 10, 15},
		{Equal, 10, 16},
		{Equal, 10, 17},
		{Equal, 10, 18},
		{Equal, 10, 19},
		{Equal, 10, vconst(20)},
		{Equal, 10, vconst2(20)},
		{Equal, 10, 25},

		{Equal, 12, 15},
		{Equal, 12, 16},
		{Equal, 12, 17},
		{Equal, 12, 18},
		{Equal, 12, 19},
		{Equal, 12, vconst(20)},
		{Equal, 12, vconst2(20)},
		{Equal, 12, 25},

		{Equal, 15, 16},
		{Equal, 15, 17},
		{Equal, 15, 18},
		{Equal, 15, 19},
		{Equal, 15, vconst(20)},
		{Equal, 15, vconst2(20)},
		{Equal, 15, 25},

		{Equal, 16, 17},
		{Equal, 16, 18},
		{Equal, 16, 19},
		{Equal, 16, vconst(20)},
		{Equal, 16, vconst2(20)},
		{Equal, 16, 25},

		{Equal, 17, 18},
		{Equal, 17, 19},
		{Equal, 17, vconst(20)},
		{Equal, 17, vconst2(20)},
		{Equal, 17, 25},

		{Equal, 18, 19},
		{Equal, 18, vconst(20)},
		{Equal, 18, vconst2(20)},
		{Equal, 18, 25},

		{Equal, 19, vconst(20)},
		{Equal, 19, vconst2(20)},
		{Equal, 19, 25},

		{Equal, vconst(20), vconst2(20)},
		{Equal, vconst(20), 25},

		{Equal, vconst2(20), 25},

		// ... but not 11/26/100/101/102, which were on a different path
		{Equal_Fail, 10, 11},
		{Equal_Fail, 10, 26},
		{Equal_Fail, 10, 100},
		{Equal_Fail, 10, 101},
		{Equal_Fail, 10, 102},
		{OrderedOrEqual, 10, 26},
		{OrderedOrEqual, 25, 26},
		{OrderedOrEqual, 13, 25},
		{OrderedOrEqual, 13, 10},

		{Undo, 0, 0},
		{OrderedOrEqual, 10, 25},
		{Equal_Fail, 10, 12},
		{Equal_Fail, 10, 15},
		{Equal_Fail, 10, 25},

		{Undo, 0, 0},
	})

	testPosetOps(t, false, []posetTestOp{
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 10, 15},
		{SetOrderOrEqual, 15, 20},
		{SetOrderOrEqual, 20, 25},
		{SetOrder, 10, 16},
		{SetOrderOrEqual, 16, 20},
		// Check that we cannot collapse here because of the strict relation 10<16
		{SetOrderOrEqual_Fail, 20, 10},
		{Undo, 0, 0},
	})
}

func TestPosetSetEqual(t *testing.T) {
	testPosetOps(t, false, []posetTestOp{
		// 10<=20<=30<40,  20<=100<110
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 10, 20},
		{SetOrderOrEqual, 20, 30},
		{SetOrder, 30, 40},
		{SetOrderOrEqual, 20, 100},
		{SetOrder, 100, 110},
		{OrderedOrEqual, 10, 30},
		{OrderedOrEqual_Fail, 30, 10},
		{Ordered_Fail, 10, 30},
		{Ordered_Fail, 30, 10},
		{Ordered, 10, 40},
		{Ordered_Fail, 40, 10},

		// Try learning 10==20.
		{Checkpoint, 0, 0},
		{SetEqual, 10, 20},
		{OrderedOrEqual, 10, 20},
		{Ordered_Fail, 10, 20},
		{Equal, 10, 20},
		{SetOrderOrEqual, 10, 20},
		{SetOrderOrEqual, 20, 10},
		{SetOrder_Fail, 10, 20},
		{SetOrder_Fail, 20, 10},
		{Undo, 0, 0},

		// Try learning 20==10.
		{Checkpoint, 0, 0},
		{SetEqual, 20, 10},
		{OrderedOrEqual, 10, 20},
		{Ordered_Fail, 10, 20},
		{Equal, 10, 20},
		{Undo, 0, 0},

		// Try learning 10==40 or 30==40 or 10==110.
		{Checkpoint, 0, 0},
		{SetEqual_Fail, 10, 40},
		{SetEqual_Fail, 40, 10},
		{SetEqual_Fail, 30, 40},
		{SetEqual_Fail, 40, 30},
		{SetEqual_Fail, 10, 110},
		{SetEqual_Fail, 110, 10},
		{Undo, 0, 0},

		// Try learning 40==110, and then 10==40 or 10=110
		{Checkpoint, 0, 0},
		{SetEqual, 40, 110},
		{SetEqual_Fail, 10, 40},
		{SetEqual_Fail, 40, 10},
		{SetEqual_Fail, 10, 110},
		{SetEqual_Fail, 110, 10},
		{Undo, 0, 0},

		// Try learning 40<20 or 30<20 or 110<10
		{Checkpoint, 0, 0},
		{SetOrder_Fail, 40, 20},
		{SetOrder_Fail, 30, 20},
		{SetOrder_Fail, 110, 10},
		{Undo, 0, 0},

		// Try learning 30<=20
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 30, 20},
		{Equal, 30, 20},
		{OrderedOrEqual, 30, 100},
		{Ordered, 30, 110},
		{Undo, 0, 0},

		{Undo, 0, 0},
	})
}

func TestPosetConst(t *testing.T) {
	testPosetOps(t, false, []posetTestOp{
		{Checkpoint, 0, 0},
		{SetOrder, 1, vconst(15)},
		{SetOrderOrEqual, 100, vconst(120)},
		{Ordered, 1, vconst(15)},
		{Ordered, 1, vconst(120)},
		{OrderedOrEqual, 1, vconst(120)},
		{OrderedOrEqual, 100, vconst(120)},
		{Ordered_Fail, 100, vconst(15)},
		{Ordered_Fail, vconst(15), 100},

		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 1, 5},
		{SetOrderOrEqual, 5, 25},
		{SetEqual, 20, vconst(20)},
		{SetEqual, 25, vconst(25)},
		{Ordered, 1, 20},
		{Ordered, 1, vconst(30)},
		{Undo, 0, 0},

		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 1, 5},
		{SetOrderOrEqual, 5, 25},
		{SetEqual, vconst(-20), 5},
		{SetEqual, vconst(-25), 1},
		{Ordered, 1, 5},
		{Ordered, vconst(-30), 1},
		{Undo, 0, 0},

		{Checkpoint, 0, 0},
		{SetNonEqual, 1, vconst(4)},
		{SetNonEqual, 1, vconst(6)},
		{NonEqual, 1, vconst(4)},
		{NonEqual_Fail, 1, vconst(5)},
		{NonEqual, 1, vconst(6)},
		{Equal_Fail, 1, vconst(4)},
		{Equal_Fail, 1, vconst(5)},
		{Equal_Fail, 1, vconst(6)},
		{Equal_Fail, 1, vconst(7)},
		{Undo, 0, 0},

		{Undo, 0, 0},
	})

	testPosetOps(t, true, []posetTestOp{
		{Checkpoint, 0, 0},
		{SetOrder, 1, vconst(15)},
		{SetOrderOrEqual, 100, vconst(-5)}, // -5 is a very big number in unsigned
		{Ordered, 1, vconst(15)},
		{Ordered, 1, vconst(-5)},
		{OrderedOrEqual, 1, vconst(-5)},
		{OrderedOrEqual, 100, vconst(-5)},
		{Ordered_Fail, 100, vconst(15)},
		{Ordered_Fail, vconst(15), 100},

		{Undo, 0, 0},
	})

	testPosetOps(t, false, []posetTestOp{
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, 1, vconst(3)},
		{SetNonEqual, 1, vconst(0)},
		{Ordered_Fail, 1, vconst(0)},
		{Undo, 0, 0},
	})

	testPosetOps(t, false, []posetTestOp{
		// Check relations of a constant with itself
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, vconst(3), vconst2(3)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetEqual, vconst(3), vconst2(3)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetNonEqual_Fail, vconst(3), vconst2(3)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetOrder_Fail, vconst(3), vconst2(3)},
		{Undo, 0, 0},

		// Check relations of two constants among them, using
		// different instances of the same constant
		{Checkpoint, 0, 0},
		{SetOrderOrEqual, vconst(3), vconst(4)},
		{OrderedOrEqual, vconst(3), vconst2(4)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetOrder, vconst(3), vconst(4)},
		{Ordered, vconst(3), vconst2(4)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetEqual_Fail, vconst(3), vconst(4)},
		{SetEqual_Fail, vconst(3), vconst2(4)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{NonEqual, vconst(3), vconst(4)},
		{NonEqual, vconst(3), vconst2(4)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{Equal_Fail, vconst(3), vconst(4)},
		{Equal_Fail, vconst(3), vconst2(4)},
		{Undo, 0, 0},
		{Checkpoint, 0, 0},
		{SetNonEqual, vconst(3), vconst(4)},
		{SetNonEqual, vconst(3), vconst2(4)},
		{Undo, 0, 0},
	})
}

func TestPosetNonEqual(t *testing.T) {
	testPosetOps(t, false, []posetTestOp{
		{Equal_Fail, 10, 20},
		{NonEqual_Fail, 10, 20},

		// Learn 10!=20
		{Checkpoint, 0, 0},
		{SetNonEqual, 10, 20},
		{Equal_Fail, 10, 20},
		{NonEqual, 10, 20},
		{SetEqual_Fail, 10, 20},

		// Learn again 10!=20
		{Checkpoint, 0, 0},
		{SetNonEqual, 10, 20},
		{Equal_Fail, 10, 20},
		{NonEqual, 10, 20},

		// Undo. We still know 10!=20
		{Undo, 0, 0},
		{Equal_Fail, 10, 20},
		{NonEqual, 10, 20},
		{SetEqual_Fail, 10, 20},

		// Undo again. Now we know nothing
		{Undo, 0, 0},
		{Equal_Fail, 10, 20},
		{NonEqual_Fail, 10, 20},

		// Learn 10==20
		{Checkpoint, 0, 0},
		{SetEqual, 10, 20},
		{Equal, 10, 20},
		{NonEqual_Fail, 10, 20},
		{SetNonEqual_Fail, 10, 20},

		// Learn again 10==20
		{Checkpoint, 0, 0},
		{SetEqual, 10, 20},
		{Equal, 10, 20},
		{NonEqual_Fail, 10, 20},
		{SetNonEqual_Fail, 10, 20},

		// Undo. We still know 10==20
		{Undo, 0, 0},
		{Equal, 10, 20},
		{NonEqual_Fail, 10, 20},
		{SetNonEqual_Fail, 10, 20},

		// Undo. We know nothing
		{Undo, 0, 0},
		{Equal_Fail, 10, 20},
		{NonEqual_Fail, 10, 20},
	})
}
