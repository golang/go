// Derived from Inferno utils/5c/txt.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/5c/txt.c
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

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"fmt"
)

var resvd = []int{
	arm.REG_R9,  // formerly reserved for m; might be okay to reuse now; not sure about NaCl
	arm.REG_R10, // reserved for g
}

/*
 * generate one instruction:
 *	as f, t
 */
func gins(as obj.As, f *gc.Node, t *gc.Node) *obj.Prog {
	//	Node nod;
	//	int32 v;

	if f != nil && f.Op == gc.OINDEX {
		gc.Fatalf("gins OINDEX not implemented")
	}

	//		gc.Regalloc(&nod, &regnode, Z);
	//		v = constnode.vconst;
	//		gc.Cgen(f->right, &nod);
	//		constnode.vconst = v;
	//		idx.reg = nod.reg;
	//		gc.Regfree(&nod);
	if t != nil && t.Op == gc.OINDEX {
		gc.Fatalf("gins OINDEX not implemented")
	}

	//		gc.Regalloc(&nod, &regnode, Z);
	//		v = constnode.vconst;
	//		gc.Cgen(t->right, &nod);
	//		constnode.vconst = v;
	//		idx.reg = nod.reg;
	//		gc.Regfree(&nod);

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	switch as {
	case arm.ABL:
		if p.To.Type == obj.TYPE_REG {
			p.To.Type = obj.TYPE_MEM
		}

	case arm.ACMP, arm.ACMPF, arm.ACMPD:
		if t != nil {
			if f.Op != gc.OREGISTER {
				/* generate a comparison
				TODO(kaib): one of the args can actually be a small constant. relax the constraint and fix call sites.
				*/
				gc.Fatalf("bad operands to gcmp")
			}
			p.From = p.To
			p.To = obj.Addr{}
			raddr(f, p)
		}

	case arm.AMULU:
		if f != nil && f.Op != gc.OREGISTER {
			gc.Fatalf("bad operands to mul")
		}

	case arm.AMOVW:
		if (p.From.Type == obj.TYPE_MEM || p.From.Type == obj.TYPE_ADDR || p.From.Type == obj.TYPE_CONST) && (p.To.Type == obj.TYPE_MEM || p.To.Type == obj.TYPE_ADDR) {
			gc.Fatalf("gins double memory")
		}

	case arm.AADD:
		if p.To.Type == obj.TYPE_MEM {
			gc.Fatalf("gins arith to mem")
		}

	case arm.ARSB:
		if p.From.Type == obj.TYPE_NONE {
			gc.Fatalf("rsb with no from")
		}
	}

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}
	return p
}

/*
 * insert n into reg slot of p
 */
func raddr(n *gc.Node, p *obj.Prog) {
	var a obj.Addr
	gc.Naddr(&a, n)
	if a.Type != obj.TYPE_REG {
		if n != nil {
			gc.Fatalf("bad in raddr: %v", n.Op)
		} else {
			gc.Fatalf("bad in raddr: <null>")
		}
		p.Reg = 0
	} else {
		p.Reg = a.Reg
	}
}
