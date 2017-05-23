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

package gc

import "cmd/internal/obj"

func Prog(as obj.As) *obj.Prog {
	var p *obj.Prog

	p = pc
	pc = Ctxt.NewProg()
	Clearp(pc)
	p.Link = pc

	if lineno == 0 && Debug['K'] != 0 {
		Warn("prog: line 0")
	}

	p.As = as
	p.Lineno = lineno
	return p
}

func Clearp(p *obj.Prog) {
	obj.Nopout(p)
	p.As = obj.AEND
	p.Pc = int64(pcloc)
	pcloc++
}

func Appendpp(p *obj.Prog, as obj.As, ftype obj.AddrType, freg int16, foffset int64, ttype obj.AddrType, treg int16, toffset int64) *obj.Prog {
	q := Ctxt.NewProg()
	Clearp(q)
	q.As = as
	q.Lineno = p.Lineno
	q.From.Type = ftype
	q.From.Reg = freg
	q.From.Offset = foffset
	q.To.Type = ttype
	q.To.Reg = treg
	q.To.Offset = toffset
	q.Link = p.Link
	p.Link = q
	return q
}

func AddAsmAfter(as obj.As, p *obj.Prog) *obj.Prog {
	q := Ctxt.NewProg()
	Clearp(q)
	q.As = as
	q.Link = p.Link
	p.Link = q
	return q
}

func ggloblnod(nam *Node) {
	s := Linksym(nam.Sym)
	s.Gotype = Linksym(ngotype(nam))
	flags := 0
	if nam.Name.Readonly {
		flags = obj.RODATA
	}
	if nam.Type != nil && !haspointers(nam.Type) {
		flags |= obj.NOPTR
	}
	Ctxt.Globl(s, nam.Type.Width, flags)
}

func ggloblsym(s *Sym, width int32, flags int16) {
	ggloblLSym(Linksym(s), width, flags)
}

func ggloblLSym(s *obj.LSym, width int32, flags int16) {
	if flags&obj.LOCAL != 0 {
		s.Set(obj.AttrLocal, true)
		flags &^= obj.LOCAL
	}
	Ctxt.Globl(s, int64(width), int(flags))
}

func gtrack(s *Sym) {
	p := Gins(obj.AUSEFIELD, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
}

func isfat(t *Type) bool {
	if t != nil {
		switch t.Etype {
		case TSTRUCT, TARRAY, TSLICE, TSTRING,
			TINTER: // maybe remove later
			return true
		}
	}

	return false
}

// Naddr rewrites a to refer to n.
// It assumes that a is zeroed on entry.
func Naddr(a *obj.Addr, n *Node) {
	if n == nil {
		return
	}

	if n.Op != ONAME {
		Debug['h'] = 1
		Dump("naddr", n)
		Fatalf("naddr: bad %v %v", n.Op, Ctxt.Dconv(a))
	}

	a.Offset = n.Xoffset
	s := n.Sym
	a.Node = n.Orig

	if s == nil {
		Fatalf("naddr: nil sym %v", n)
	}

	a.Type = obj.TYPE_MEM
	switch n.Class {
	default:
		Fatalf("naddr: ONAME class %v %d\n", n.Sym, n.Class)

	case PEXTERN, PFUNC:
		a.Name = obj.NAME_EXTERN

	case PAUTO:
		a.Name = obj.NAME_AUTO

	case PPARAM, PPARAMOUT:
		a.Name = obj.NAME_PARAM
	}

	a.Sym = Linksym(s)
}

func Addrconst(a *obj.Addr, v int64) {
	a.Sym = nil
	a.Type = obj.TYPE_CONST
	a.Offset = v
}

func newplist() *obj.Plist {
	pl := obj.Linknewplist(Ctxt)

	pc = Ctxt.NewProg()
	Clearp(pc)
	pl.Firstpc = pc

	return pl
}

// nodarg returns a Node for the function argument denoted by t,
// which is either the entire function argument or result struct (t is a  struct *Type)
// or a specific argument (t is a *Field within a struct *Type).
//
// If fp is 0, the node is for use by a caller invoking the given
// function, preparing the arguments before the call
// or retrieving the results after the call.
// In this case, the node will correspond to an outgoing argument
// slot like 8(SP).
//
// If fp is 1, the node is for use by the function itself
// (the callee), to retrieve its arguments or write its results.
// In this case the node will be an ONAME with an appropriate
// type and offset.
func nodarg(t interface{}, fp int) *Node {
	var n *Node

	var funarg Funarg
	switch t := t.(type) {
	default:
		Fatalf("bad nodarg %T(%v)", t, t)

	case *Type:
		// Entire argument struct, not just one arg
		if !t.IsFuncArgStruct() {
			Fatalf("nodarg: bad type %v", t)
		}
		funarg = t.StructType().Funarg

		// Build fake variable name for whole arg struct.
		n = nod(ONAME, nil, nil)
		n.Sym = lookup(".args")
		n.Type = t
		first := t.Field(0)
		if first == nil {
			Fatalf("nodarg: bad struct")
		}
		if first.Offset == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = first.Offset
		n.Addable = true

	case *Field:
		funarg = t.Funarg
		if fp == 1 {
			// NOTE(rsc): This should be using t.Nname directly,
			// except in the case where t.Nname.Sym is the blank symbol and
			// so the assignment would be discarded during code generation.
			// In that case we need to make a new node, and there is no harm
			// in optimization passes to doing so. But otherwise we should
			// definitely be using the actual declaration and not a newly built node.
			// The extra Fatalf checks here are verifying that this is the case,
			// without changing the actual logic (at time of writing, it's getting
			// toward time for the Go 1.7 beta).
			// At some quieter time (assuming we've never seen these Fatalfs happen)
			// we could change this code to use "expect" directly.
			expect := t.Nname
			if expect.isParamHeapCopy() {
				expect = expect.Name.Param.Stackcopy
			}

			for _, n := range Curfn.Func.Dcl {
				if (n.Class == PPARAM || n.Class == PPARAMOUT) && !isblanksym(t.Sym) && n.Sym == t.Sym {
					if n != expect {
						Fatalf("nodarg: unexpected node: %v (%p %v) vs %v (%p %v)", n, n, n.Op, t.Nname, t.Nname, t.Nname.Op)
					}
					return n
				}
			}

			if !isblanksym(expect.Sym) {
				Fatalf("nodarg: did not find node in dcl list: %v", expect)
			}
		}

		// Build fake name for individual variable.
		// This is safe because if there was a real declared name
		// we'd have used it above.
		n = nod(ONAME, nil, nil)
		n.Type = t.Type
		n.Sym = t.Sym
		if t.Offset == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = t.Offset
		n.Addable = true
		n.Orig = t.Nname
	}

	// Rewrite argument named _ to __,
	// or else the assignment to _ will be
	// discarded during code generation.
	if isblank(n) {
		n.Sym = lookup("__")
	}

	switch fp {
	default:
		Fatalf("bad fp")

	case 0: // preparing arguments for call
		n.Op = OINDREGSP
		n.Xoffset += Ctxt.FixedFrameSize()

	case 1: // reading arguments inside call
		n.Class = PPARAM
		if funarg == FunargResults {
			n.Class = PPARAMOUT
		}
	}

	n.Typecheck = 1
	n.Addrtaken = true // keep optimizers at bay
	return n
}

func Patch(p *obj.Prog, to *obj.Prog) {
	if p.To.Type != obj.TYPE_BRANCH {
		Fatalf("patch: not a branch")
	}
	p.To.Val = to
	p.To.Offset = to.Pc
}

// Gins inserts instruction as. f is from, t is to.
func Gins(as obj.As, f, t *Node) *obj.Prog {
	switch as {
	case obj.AVARKILL, obj.AVARLIVE, obj.AVARDEF, obj.ATYPE,
		obj.ATEXT, obj.AFUNCDATA, obj.AUSEFIELD:
	default:
		Fatalf("unhandled gins op %v", as)
	}

	p := Prog(as)
	Naddr(&p.From, f)
	Naddr(&p.To, t)
	return p
}
