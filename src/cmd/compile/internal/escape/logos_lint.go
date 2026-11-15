// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"go/constant"

	"cmd/compile/internal/ir"
	"cmd/internal/src"
)

const LogosPinBytes = 64 << 10

var LogosPinning bool

type LogosPin struct {
	Pos       src.XPos
	Bytes     int64
	Substring bool
}

var logosPins []LogosPin

func LogosPins() []LogosPin {
	return logosPins
}

func (e *escape) logosSlice(k hole, n *ir.SliceExpr) hole {
	if !LogosPinning || !pinsLargeBacking(n) {
		return k
	}
	return e.spill(k, n)
}

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

func pinsLargeBacking(n *ir.SliceExpr) bool {
	size := backingBytes(n)
	return size > LogosPinBytes && !coversBacking(n, size)
}

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

func stringLength(x ir.Node) int64 {
	if v := ir.StaticValue(x); v.Op() == ir.OBYTES2STR {
		return sliceLength(v.(*ir.ConvExpr).X)
	}
	return 0
}

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

func operandLength(n *ir.SliceExpr) int64 {
	if n.Op() == ir.OSLICEARR {
		if t := n.X.Type(); t.IsPtr() && t.Elem().IsArray() {
			return t.Elem().NumElem()
		}
		return 0
	}
	return sliceLength(n.X)
}

func boundOr(b ir.Node, def int64) (int64, bool) {
	if b == nil {
		return def, true
	}
	return constInt(b)
}

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
