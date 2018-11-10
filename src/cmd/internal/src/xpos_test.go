// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package src

import (
	"testing"
	"unsafe"
)

func TestNoXPos(t *testing.T) {
	var tab PosTable
	if tab.Pos(NoXPos) != NoPos {
		t.Errorf("failed to translate NoXPos to Pos using zero PosTable")
	}
}

func TestConversion(t *testing.T) {
	b1 := NewFileBase("b1", "b1")
	b2 := NewFileBase("b2", "b2")
	b3 := NewLinePragmaBase(MakePos(b1, 10, 0), "b3", 123)

	var tab PosTable
	for _, want := range []Pos{
		NoPos,
		MakePos(nil, 0, 0), // same table entry as NoPos
		MakePos(b1, 0, 0),
		MakePos(nil, 10, 20), // same table entry as NoPos
		MakePos(b2, 10, 20),
		MakePos(b3, 10, 20),
		MakePos(b3, 123, 0), // same table entry as MakePos(b3, 10, 20)
	} {
		xpos := tab.XPos(want)
		got := tab.Pos(xpos)
		if got != want {
			t.Errorf("got %v; want %v", got, want)
		}
	}

	if len(tab.baseList) != len(tab.indexMap) {
		t.Errorf("table length discrepancy: %d != %d", len(tab.baseList), len(tab.indexMap))
	}

	const wantLen = 4
	if len(tab.baseList) != wantLen {
		t.Errorf("got table length %d; want %d", len(tab.baseList), wantLen)
	}

	if got := tab.XPos(NoPos); got != NoXPos {
		t.Errorf("XPos(NoPos): got %v; want %v", got, NoXPos)
	}

	if tab.baseList[0] != nil || tab.indexMap[nil] != 0 {
		t.Errorf("nil base not at index 0")
	}
}

func TestSize(t *testing.T) {
	var p XPos
	if unsafe.Alignof(p) != 4 {
		t.Errorf("alignment = %v; want 4", unsafe.Alignof(p))
	}
	if unsafe.Sizeof(p) != 8 {
		t.Errorf("size = %v; want 8", unsafe.Sizeof(p))
	}
}

func TestSetBase(t *testing.T) {
	var tab PosTable
	b1 := NewFileBase("b1", "b1")
	orig := MakePos(b1, 42, 7)
	xpos := tab.XPos(orig)

	pos := tab.Pos(xpos)
	new := NewInliningBase(b1, 2)
	pos.SetBase(new)
	xpos = tab.XPos(pos)

	pos = tab.Pos(xpos)
	if inl := pos.Base().InliningIndex(); inl != 2 {
		t.Fatalf("wrong inlining index: %d", inl)
	}
}
