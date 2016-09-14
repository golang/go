// Derived from Inferno utils/6c/txt.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6c/txt.c
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

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"fmt"
)

var resvd = []int{
	x86.REG_DI, // for movstring
	x86.REG_SI, // for movstring

	x86.REG_AX, // for divide
	x86.REG_CX, // for shift
	x86.REG_DX, // for divide
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
	//	Node nod;

	//	if(f != N && f->op == OINDEX) {
	//		gc.Regalloc(&nod, &regnode, Z);
	//		v = constnode.vconst;
	//		gc.Cgen(f->right, &nod);
	//		constnode.vconst = v;
	//		idx.reg = nod.reg;
	//		gc.Regfree(&nod);
	//	}
	//	if(t != N && t->op == OINDEX) {
	//		gc.Regalloc(&nod, &regnode, Z);
	//		v = constnode.vconst;
	//		gc.Cgen(t->right, &nod);
	//		constnode.vconst = v;
	//		idx.reg = nod.reg;
	//		gc.Regfree(&nod);
	//	}

	if f != nil && f.Op == gc.OADDR && (as == x86.AMOVL || as == x86.AMOVQ) {
		// Turn MOVL $xxx into LEAL xxx.
		// These should be equivalent but most of the backend
		// only expects to see LEAL, because that's what we had
		// historically generated. Various hidden assumptions are baked in by now.
		if as == x86.AMOVL {
			as = x86.ALEAL
		} else {
			as = x86.ALEAQ
		}
		f = f.Left
	}

	switch as {
	case x86.AMOVB,
		x86.AMOVW,
		x86.AMOVL,
		x86.AMOVQ,
		x86.AMOVSS,
		x86.AMOVSD:
		if f != nil && t != nil && samaddr(f, t) {
			return nil
		}

	case x86.ALEAQ:
		if f != nil && gc.Isconst(f, gc.CTNIL) {
			gc.Fatalf("gins LEAQ nil %v", f.Type)
		}
	}

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := int32(0)
	switch as {
	case x86.AMOVB:
		w = 1

	case x86.AMOVW:
		w = 2

	case x86.AMOVL:
		w = 4

	case x86.AMOVQ:
		w = 8
	}

	if w != 0 && ((f != nil && p.From.Width < int64(w)) || (t != nil && p.To.Width > int64(w))) {
		gc.Dump("f", f)
		gc.Dump("t", t)
		gc.Fatalf("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	if p.To.Type == obj.TYPE_ADDR && w > 0 {
		gc.Fatalf("bad use of addr: %v", p)
	}

	return p
}

func ginsnop() {
	// This is actually not the x86 NOP anymore,
	// but at the point where it gets used, AX is dead
	// so it's okay if we lose the high bits.
	var reg gc.Node
	gc.Nodreg(&reg, gc.Types[gc.TINT], x86.REG_AX)
	gins(x86.AXCHGL, &reg, &reg)
}
