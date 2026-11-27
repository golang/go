// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"bytes"
	"cmd/internal/obj"
	"testing"
)

// TestWriteOpcode checks one opcode from each encoding range. Vector
// instructions are the 0xFD prefix followed by a ULEB128-encoded opcode
// index, so the cases below bracket both points where that index grows a
// byte: 0x07F to 0x080 and 0x0FF to 0x100.
func TestWriteOpcode(t *testing.T) {
	tests := []struct {
		as   obj.As
		want []byte
	}{
		{AUnreachable, []byte{0x00}},
		{AEnd, []byte{0x0B}},
		{ADrop, []byte{0x1A}},
		{ALocalGet, []byte{0x20}},
		{AI32Load, []byte{0x28}},
		{ARefNull, []byte{0xD0}},
		{AContNew, []byte{0xE0}},
		{AI32TruncSatF32S, []byte{0xFC, 0x00}},

		{AV128Load, []byte{0xFD, 0x00}},                            // index 0x000
		{AI32x4ExtaddPairwiseI16x8U, []byte{0xFD, 0x7F}},           // index 0x07F
		{AI16x8Abs, []byte{0xFD, 0x80, 0x01}},                      // index 0x080
		{AF64x2ConvertLowI32x4U, []byte{0xFD, 0xFF, 0x01}},         // index 0x0FF
		{AI8x16RelaxedSwizzle, []byte{0xFD, 0x80, 0x02}},           // index 0x100
		{AI32x4RelaxedDotI8x16I7x16AddS, []byte{0xFD, 0x93, 0x02}}, // index 0x113
	}
	var buf bytes.Buffer
	for _, test := range tests {
		buf.Reset()
		writeOpcode(&buf, test.as)
		if got := buf.Bytes(); !bytes.Equal(got, test.want) {
			t.Errorf("writeOpcode(%v) = % x, want % x", test.as, got, test.want)
		}
	}
}

// TestMemarg checks the immediates of the vector loads and stores
// other than V128Load and V128Store.
func TestMemarg(t *testing.T) {
	tests := []struct {
		p    obj.Prog
		want []byte
	}{
		{obj.Prog{As: AV128Load8x8S, From: constAddr(5)}, []byte{0xFD, 0x01, 3, 5}},
		{obj.Prog{As: AV128Load8x8U, From: constAddr(5)}, []byte{0xFD, 0x02, 3, 5}},
		{obj.Prog{As: AV128Load16x4S, From: constAddr(5)}, []byte{0xFD, 0x03, 3, 5}},
		{obj.Prog{As: AV128Load16x4U, From: constAddr(5)}, []byte{0xFD, 0x04, 3, 5}},
		{obj.Prog{As: AV128Load32x2S, From: constAddr(5)}, []byte{0xFD, 0x05, 3, 5}},
		{obj.Prog{As: AV128Load32x2U, From: constAddr(5)}, []byte{0xFD, 0x06, 3, 5}},
		{obj.Prog{As: AV128Load8Splat, From: constAddr(5)}, []byte{0xFD, 0x07, 0, 5}},
		{obj.Prog{As: AV128Load16Splat, From: constAddr(5)}, []byte{0xFD, 0x08, 1, 5}},
		{obj.Prog{As: AV128Load32Splat, From: constAddr(5)}, []byte{0xFD, 0x09, 2, 5}},
		{obj.Prog{As: AV128Load64Splat, From: constAddr(5)}, []byte{0xFD, 0x0A, 3, 5}},
		{obj.Prog{As: AV128Load32Zero, From: constAddr(5)}, []byte{0xFD, 0x5C, 2, 5}},
		{obj.Prog{As: AV128Load64Zero, From: constAddr(200)}, []byte{0xFD, 0x5D, 3, 0xC8, 0x01}},
		{obj.Prog{As: AV128Load8Splat, From: obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: 5}}, []byte{0x20, 0x01, 0xFD, 0x07, 0, 5}},

		{obj.Prog{As: AV128Load8Lane, From: constAddr(5), To: constAddr(15)}, []byte{0xFD, 0x54, 0, 5, 15}},
		{obj.Prog{As: AV128Load16Lane, From: constAddr(5), To: constAddr(7)}, []byte{0xFD, 0x55, 1, 5, 7}},
		{obj.Prog{As: AV128Load32Lane, From: constAddr(5), To: constAddr(3)}, []byte{0xFD, 0x56, 2, 5, 3}},
		{obj.Prog{As: AV128Load64Lane, From: constAddr(5), To: constAddr(1)}, []byte{0xFD, 0x57, 3, 5, 1}},
		{obj.Prog{As: AV128Store8Lane, From: constAddr(5), To: constAddr(15)}, []byte{0xFD, 0x58, 0, 5, 15}},
		{obj.Prog{As: AV128Store16Lane, From: constAddr(5), To: constAddr(7)}, []byte{0xFD, 0x59, 1, 5, 7}},
		{obj.Prog{As: AV128Store32Lane, From: constAddr(5), To: constAddr(3)}, []byte{0xFD, 0x5A, 2, 5, 3}},
		{obj.Prog{As: AV128Store64Lane, From: constAddr(200), To: constAddr(1)}, []byte{0xFD, 0x5B, 3, 0xC8, 0x01, 1}},
	}
	for _, test := range tests {
		if got := encode(t, test.p); !bytes.Equal(got, test.want) {
			t.Errorf("encode(%v) = % x, want % x", test.p.As, got, test.want)
		}
	}
}

// encode returns the encoding of p in a function, before a Drop.
func encode(t *testing.T, p obj.Prog) []byte {
	t.Helper()
	drop := obj.Prog{As: ADrop}
	fn := assembleFunc(t, p, drop)
	ref := assembleFunc(t, drop)
	i := 0
	for i < len(ref) && i < len(fn) && fn[i] == ref[i] {
		i++
	}
	n := len(fn) - len(ref)
	if n <= 0 || !bytes.Equal(fn[i+n:], ref[i:]) {
		t.Fatalf("%v: % x is not % x with one instruction added", p.As, fn, ref)
	}
	return fn[i : i+n]
}

// assembleFunc returns the encoding of a function with body as its code.
func assembleFunc(t *testing.T, body ...obj.Prog) []byte {
	t.Helper()
	ctxt := obj.Linknew(&Linkwasm)
	ctxt.DiagFunc = t.Errorf
	instinit(ctxt)
	s := ctxt.Lookup("test")
	s.NewFuncInfo()
	p := ctxt.NewProg()
	p.As = obj.ATEXT
	p.From = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: s}
	p.To = obj.Addr{Type: obj.TYPE_TEXTSIZE, Val: int32(0)}
	s.Func().Text = p
	for _, b := range body {
		p = obj.Appendp(p, ctxt.NewProg)
		p.As, p.From, p.To = b.As, b.From, b.To
	}
	preprocess(ctxt, s, ctxt.NewProg)
	assemble(ctxt, s, ctxt.NewProg)
	return s.P
}
