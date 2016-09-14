// Derived from Inferno utils/8c/txt.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/8c/txt.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package x86

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"fmt"
)

var resvd = []int{
	//	REG_DI,	// for movstring
	//	REG_SI,	// for movstring

	x86.REG_AX, // for divide
	x86.REG_CX, // for shift
	x86.REG_DX, // for divide, context
	x86.REG_SP, // for stack
}

func samaddr(f *gc.Node, t *gc.Node) bool {
	if f.Op != t.Op {
		return false
	}

	switch f.Op {
	case gc.OREGISTER:
		if f.Reg != t.Reg {
			break
		}
		return true
	}

	return false
}

/*
 * generate one instruction:
 *	as f, t
 */
func gins(as obj.As, f *gc.Node, t *gc.Node) *obj.Prog {
	if as == x86.AFMOVF && f != nil && f.Op == gc.OREGISTER && t != nil && t.Op == gc.OREGISTER {
		gc.Fatalf("gins MOVF reg, reg")
	}
	if as == x86.ACVTSD2SS && f != nil && f.Op == gc.OLITERAL {
		gc.Fatalf("gins CVTSD2SS const")
	}
	if as == x86.AMOVSD && t != nil && t.Op == gc.OREGISTER && t.Reg == x86.REG_F0 {
		gc.Fatalf("gins MOVSD into F0")
	}

	if as == x86.AMOVL && f != nil && f.Op == gc.OADDR && f.Left.Op == gc.ONAME && f.Left.Class != gc.PEXTERN && f.Left.Class != gc.PFUNC {
		// Turn MOVL $xxx(FP/SP) into LEAL xxx.
		// These should be equivalent but most of the backend
		// only expects to see LEAL, because that's what we had
		// historically generated. Various hidden assumptions are baked in by now.
		as = x86.ALEAL
		f = f.Left
	}

	switch as {
	case x86.AMOVB,
		x86.AMOVW,
		x86.AMOVL:
		if f != nil && t != nil && samaddr(f, t) {
			return nil
		}

	case x86.ALEAL:
		if f != nil && gc.Isconst(f, gc.CTNIL) {
			gc.Fatalf("gins LEAL nil %v", f.Type)
		}
	}

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := 0
	switch as {
	case x86.AMOVB:
		w = 1

	case x86.AMOVW:
		w = 2

	case x86.AMOVL:
		w = 4
	}

	if true && w != 0 && f != nil && (p.From.Width > int64(w) || p.To.Width > int64(w)) {
		gc.Dump("bad width from:", f)
		gc.Dump("bad width to:", t)
		gc.Fatalf("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	if p.To.Type == obj.TYPE_ADDR && w > 0 {
		gc.Fatalf("bad use of addr: %v", p)
	}

	return p
}

func ginsnop() {
	var reg gc.Node
	gc.Nodreg(&reg, gc.Types[gc.TINT], x86.REG_AX)
	gins(x86.AXCHGL, &reg, &reg)
}
