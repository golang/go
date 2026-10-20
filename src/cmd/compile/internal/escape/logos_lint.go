// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"go/constant"

	"cmd/compile/internal/ir"
	"cmd/internal/src"
)

// LogosPinBytes is the backing size above which an escaping subslice or
// substring is a perf::large_backing_array_pin finding (SPEC Sec 1.2, #158):
// 64 KB, the size ir.MaxImplicitStackVarSize already treats as large.
const LogosPinBytes = 64 << 10

// LogosPinning makes escape analysis track the slice expressions whose
// backing store is statically larger than LogosPinBytes. The loGo/s lint
// hook sets it when perf::large_backing_array_pin is active for the package;
// it is off for every other compile, so the flow graph and the -m output stay
// those of plain Go.
var LogosPinning bool

// LogosPin is one slice or substring expression of a large backing store
// whose value escapes: it lives beyond the function or reaches the heap, so
// the whole backing store stays reachable.
type LogosPin struct {
	Pos       src.XPos
	Bytes     int64
	Substring bool
}

var logosPins []LogosPin

// LogosPins returns the pins found so far, in analysis order. A source
// expression the compiler analyzes more than once, such as an inlined body or
// a generic instantiation, can appear more than once.
func LogosPins() []LogosPin {
	return logosPins
}

// logosSlice returns the hole the operand of slice expression n flows to. A
// subslice of a large backing store gets a location of its own, as a heap
// allocation of n would: the address of the location flows to k, so the
// location escapes exactly when the sliced value does, and nothing else in
// the flow graph changes.
func (e *escape) logosSlice(k hole, n *ir.SliceExpr) hole {
	if !LogosPinning || !pinsLargeBacking(n) {
		return k
	}
	return e.spill(k, n)
}

// logosCollect records the slice expressions that logosSlice gave a location
// and that escape. It runs after the batch is solved.
func (b *batch) logosCollect() {
	if !LogosPinning {
		return
	}
	for _, loc := range b.allLocs {
		n, ok := loc.n.(*ir.SliceExpr)
		if !ok || !loc.hasAttr(attrEscapes) {
			continue
		}
		logosPins = append(logosPins, LogosPin{
			Pos:       n.Pos(),
			Bytes:     backingBytes(n),
			Substring: n.Op() == ir.OSLICESTR,
		})
	}
}

// pinsLargeBacking reports whether n cuts a part out of a backing store the
// compiler knows to be larger than LogosPinBytes.
func pinsLargeBacking(n *ir.SliceExpr) bool {
	size := backingBytes(n)
	return size > LogosPinBytes && !coversBacking(n, size)
}

// backingBytes returns the size in bytes of the backing store that the slice
// expression n slices, or 0 when it is not statically known.
func backingBytes(n *ir.SliceExpr) int64 {
	switch n.Op() {
	case ir.OSLICEARR, ir.OSLICE3ARR:
		if t := n.X.Type(); t.IsPtr() && t.Elem().IsArray() {
			return t.Elem().Size()
		}
	case ir.OSLICE, ir.OSLICE3:
		return sliceBacking(n.X)
	case ir.OSLICESTR:
		return stringLength(n.X)
	}
	return 0
}

// sliceBacking returns the size in bytes of the backing store of the slice
// value x, or 0 when it is not statically known: a make with a constant
// size, or a slice of one of those or of an array.
func sliceBacking(x ir.Node) int64 {
	switch v := ir.StaticValue(x); v.Op() {
	case ir.OMAKESLICE:
		m := v.(*ir.MakeExpr)
		size := m.Cap
		if size == nil {
			size = m.Len
		}
		count, _ := constInt(size)
		return count * v.Type().Elem().Size()
	case ir.OSLICE, ir.OSLICE3, ir.OSLICEARR, ir.OSLICE3ARR:
		return backingBytes(v.(*ir.SliceExpr))
	}
	return 0
}

// stringLength returns the length of the string x when it is a conversion of
// a slice of statically known length, which copies the bytes into a backing
// store of that size, or 0 when it is not statically known. A constant
// string lives in read-only data, so it pins nothing.
func stringLength(x ir.Node) int64 {
	if v := ir.StaticValue(x); v.Op() == ir.OBYTES2STR {
		return sliceLength(v.(*ir.ConvExpr).X)
	}
	return 0
}

// sliceLength returns the length of the slice value x when it is statically
// known, or 0.
func sliceLength(x ir.Node) int64 {
	switch v := ir.StaticValue(x); v.Op() {
	case ir.OMAKESLICE:
		n, _ := constInt(v.(*ir.MakeExpr).Len)
		return n
	case ir.OSLICE, ir.OSLICEARR:
		return slicedLength(v.(*ir.SliceExpr))
	}
	return 0
}

// slicedLength returns the length of the two-index slice expression n when
// its bounds are constant and its operand has a known length, or 0.
func slicedLength(n *ir.SliceExpr) int64 {
	lo, ok := boundOr(n.Low, 0)
	if !ok {
		return 0
	}
	hi, ok := boundOr(n.High, operandLength(n))
	if !ok || hi < lo {
		return 0
	}
	return hi - lo
}

// operandLength returns the length of what slice expression n slices: the
// array length for an array pointer, the known length of a slice, or 0.
func operandLength(n *ir.SliceExpr) int64 {
	if n.Op() == ir.OSLICEARR {
		if t := n.X.Type(); t.IsPtr() && t.Elem().IsArray() {
			return t.Elem().NumElem()
		}
		return 0
	}
	return sliceLength(n.X)
}

// boundOr returns the constant value of the bound b, def when the bound is
// absent, and false when the bound is not constant.
func boundOr(b ir.Node, def int64) (int64, bool) {
	if b == nil {
		return def, true
	}
	return constInt(b)
}

// constInt returns the value of n when n is, or statically resolves to, an
// integer constant.
func constInt(n ir.Node) (int64, bool) {
	if n == nil {
		return 0, false
	}
	v := ir.StaticValue(n)
	if v.Op() != ir.OLITERAL {
		return 0, false
	}
	return constant.Int64Val(v.Val())
}

// coversBacking reports whether n slices the whole backing store of size
// bytes: it starts at 0, ends at the end, and has no capacity bound. Such an
// expression is the allocation itself, not a subslice.
func coversBacking(n *ir.SliceExpr, size int64) bool {
	if n.Max != nil {
		return false
	}
	if lo, ok := boundOr(n.Low, 0); !ok || lo != 0 {
		return false
	}
	if n.High == nil {
		return true
	}
	elem := int64(1)
	if n.Op() != ir.OSLICESTR {
		elem = n.Type().Elem().Size()
	}
	if hi, ok := constInt(n.High); ok {
		return elem > 0 && hi == size/elem
	}
	return n.High.Op() == ir.OLEN && ir.SameSafeExpr(n.High.(*ir.UnaryExpr).X, n.X)
}
