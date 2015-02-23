// Derived from Inferno utils/6c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/txt.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

var ddumped int

var dfirst *obj.Prog

var dpc *obj.Prog

/*
 * Is this node a memory operand?
 */
func Ismem(n *Node) bool {
	switch n.Op {
	case OITAB,
		OSPTR,
		OLEN,
		OCAP,
		OINDREG,
		ONAME,
		OPARAM,
		OCLOSUREVAR:
		return true

	case OADDR:
		return Thearch.Thechar == '6' || Thearch.Thechar == '9' // because 6g uses PC-relative addressing; TODO(rsc): not sure why 9g too
	}

	return false
}

func Samereg(a *Node, b *Node) bool {
	if a == nil || b == nil {
		return false
	}
	if a.Op != OREGISTER {
		return false
	}
	if b.Op != OREGISTER {
		return false
	}
	if a.Val.U.Reg != b.Val.U.Reg {
		return false
	}
	return true
}

/*
 * gsubr.c
 */
func Gbranch(as int, t *Type, likely int) *obj.Prog {
	p := Prog(as)
	p.To.Type = obj.TYPE_BRANCH
	p.To.U.Branch = nil
	if as != obj.AJMP && likely != 0 && Thearch.Thechar != '9' {
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(bool2int(likely > 0))
	}

	return p
}

func Prog(as int) *obj.Prog {
	var p *obj.Prog

	if as == obj.ADATA || as == obj.AGLOBL {
		if ddumped != 0 {
			Fatal("already dumped data")
		}
		if dpc == nil {
			dpc = Ctxt.NewProg()
			dfirst = dpc
		}

		p = dpc
		dpc = Ctxt.NewProg()
		p.Link = dpc
	} else {
		p = Pc
		Pc = Ctxt.NewProg()
		Clearp(Pc)
		p.Link = Pc
	}

	if lineno == 0 {
		if Debug['K'] != 0 {
			Warn("prog: line 0")
		}
	}

	p.As = int16(as)
	p.Lineno = lineno
	return p
}

func Nodreg(n *Node, t *Type, r int) {
	if t == nil {
		Fatal("nodreg: t nil")
	}

	*n = Node{}
	n.Op = OREGISTER
	n.Addable = 1
	ullmancalc(n)
	n.Val.U.Reg = int16(r)
	n.Type = t
}

func Nodindreg(n *Node, t *Type, r int) {
	Nodreg(n, t, r)
	n.Op = OINDREG
}

func Afunclit(a *obj.Addr, n *Node) {
	if a.Type == obj.TYPE_ADDR && a.Name == obj.NAME_EXTERN {
		a.Type = obj.TYPE_MEM
		a.Sym = Linksym(n.Sym)
	}
}

func Clearp(p *obj.Prog) {
	obj.Nopout(p)
	p.As = obj.AEND
	p.Pc = int64(pcloc)
	pcloc++
}

func dumpdata() {
	ddumped = 1
	if dfirst == nil {
		return
	}
	newplist()
	*Pc = *dfirst
	Pc = dpc
	Clearp(Pc)
}

func fixautoused(p *obj.Prog) {
	for lp := &p; ; {
		p = *lp
		if p == nil {
			break
		}
		if p.As == obj.ATYPE && p.From.Node != nil && p.From.Name == obj.NAME_AUTO && ((p.From.Node).(*Node)).Used == 0 {
			*lp = p.Link
			continue
		}

		if (p.As == obj.AVARDEF || p.As == obj.AVARKILL) && p.To.Node != nil && ((p.To.Node).(*Node)).Used == 0 {
			// Cannot remove VARDEF instruction, because - unlike TYPE handled above -
			// VARDEFs are interspersed with other code, and a jump might be using the
			// VARDEF as a target. Replace with a no-op instead. A later pass will remove
			// the no-ops.
			obj.Nopout(p)

			continue
		}

		if p.From.Name == obj.NAME_AUTO && p.From.Node != nil {
			p.From.Offset += ((p.From.Node).(*Node)).Stkdelta
		}

		if p.To.Name == obj.NAME_AUTO && p.To.Node != nil {
			p.To.Offset += ((p.To.Node).(*Node)).Stkdelta
		}

		lp = &p.Link
	}
}

func ggloblnod(nam *Node) {
	p := Thearch.Gins(obj.AGLOBL, nam, nil)
	p.Lineno = nam.Lineno
	p.From.Sym.Gotype = Linksym(ngotype(nam))
	p.To.Sym = nil
	p.To.Type = obj.TYPE_CONST
	p.To.Offset = nam.Type.Width
	if nam.Readonly != 0 {
		p.From3.Offset = obj.RODATA
	}
	if nam.Type != nil && !haspointers(nam.Type) {
		p.From3.Offset |= obj.NOPTR
	}
}

func ggloblsym(s *Sym, width int32, flags int8) {
	p := Thearch.Gins(obj.AGLOBL, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
	p.To.Type = obj.TYPE_CONST
	p.To.Offset = int64(width)
	p.From3.Offset = int64(flags)
}

func gjmp(to *obj.Prog) *obj.Prog {
	p := Gbranch(obj.AJMP, nil, 0)
	if to != nil {
		Patch(p, to)
	}
	return p
}

func gtrack(s *Sym) {
	p := Thearch.Gins(obj.AUSEFIELD, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
}

func gused(n *Node) {
	Thearch.Gins(obj.ANOP, n, nil) // used
}

func Isfat(t *Type) bool {
	if t != nil {
		switch t.Etype {
		case TSTRUCT,
			TARRAY,
			TSTRING,
			TINTER: // maybe remove later
			return true
		}
	}

	return false
}

func markautoused(p *obj.Prog) {
	for ; p != nil; p = p.Link {
		if p.As == obj.ATYPE || p.As == obj.AVARDEF || p.As == obj.AVARKILL {
			continue
		}

		if p.From.Node != nil {
			((p.From.Node).(*Node)).Used = 1
		}

		if p.To.Node != nil {
			((p.To.Node).(*Node)).Used = 1
		}
	}
}

func Naddr(n *Node, a *obj.Addr, canemitcode int) {
	*a = obj.Addr{}
	if n == nil {
		return
	}

	if n.Type != nil && n.Type.Etype != TIDEAL {
		// TODO(rsc): This is undone by the selective clearing of width below,
		// to match architectures that were not as aggressive in setting width
		// during naddr. Those widths must be cleared to avoid triggering
		// failures in gins when it detects real but heretofore latent (and one
		// hopes innocuous) type mismatches.
		// The type mismatches should be fixed and the clearing below removed.
		dowidth(n.Type)

		a.Width = n.Type.Width
	}

	switch n.Op {
	default:
		Fatal("naddr: bad %v %v", Oconv(int(n.Op), 0), Ctxt.Dconv(a))

	case OREGISTER:
		a.Type = obj.TYPE_REG
		a.Reg = n.Val.U.Reg
		a.Sym = nil
		if Thearch.Thechar == '8' { // TODO(rsc): Never clear a->width.
			a.Width = 0
		}

	case OINDREG:
		a.Type = obj.TYPE_MEM
		a.Reg = n.Val.U.Reg
		a.Sym = Linksym(n.Sym)
		a.Offset = n.Xoffset
		if a.Offset != int64(int32(a.Offset)) {
			Yyerror("offset %d too large for OINDREG", a.Offset)
		}
		if Thearch.Thechar == '8' { // TODO(rsc): Never clear a->width.
			a.Width = 0
		}

		// n->left is PHEAP ONAME for stack parameter.
	// compute address of actual parameter on stack.
	case OPARAM:
		a.Etype = Simtype[n.Left.Type.Etype]

		a.Width = n.Left.Type.Width
		a.Offset = n.Xoffset
		a.Sym = Linksym(n.Left.Sym)
		a.Type = obj.TYPE_MEM
		a.Name = obj.NAME_PARAM
		a.Node = n.Left.Orig

	case OCLOSUREVAR:
		if !Curfn.Needctxt {
			Fatal("closurevar without needctxt")
		}
		a.Type = obj.TYPE_MEM
		a.Reg = int16(Thearch.REGCTXT)
		a.Sym = nil
		a.Offset = n.Xoffset

	case OCFUNC:
		Naddr(n.Left, a, canemitcode)
		a.Sym = Linksym(n.Left.Sym)

	case ONAME:
		a.Etype = 0
		if n.Type != nil {
			a.Etype = Simtype[n.Type.Etype]
		}
		a.Offset = n.Xoffset
		s := n.Sym
		a.Node = n.Orig

		//if(a->node >= (Node*)&n)
		//	fatal("stack node");
		if s == nil {
			s = Lookup(".noname")
		}
		if n.Method != 0 {
			if n.Type != nil {
				if n.Type.Sym != nil {
					if n.Type.Sym.Pkg != nil {
						s = Pkglookup(s.Name, n.Type.Sym.Pkg)
					}
				}
			}
		}

		a.Type = obj.TYPE_MEM
		switch n.Class {
		default:
			Fatal("naddr: ONAME class %v %d\n", Sconv(n.Sym, 0), n.Class)

		case PEXTERN:
			a.Name = obj.NAME_EXTERN

		case PAUTO:
			a.Name = obj.NAME_AUTO

		case PPARAM,
			PPARAMOUT:
			a.Name = obj.NAME_PARAM

		case PFUNC:
			a.Name = obj.NAME_EXTERN
			a.Type = obj.TYPE_ADDR
			a.Width = int64(Widthptr)
			s = funcsym(s)
		}

		a.Sym = Linksym(s)

	case OLITERAL:
		if Thearch.Thechar == '8' {
			a.Width = 0
		}
		switch n.Val.Ctype {
		default:
			Fatal("naddr: const %v", Tconv(n.Type, obj.FmtLong))

		case CTFLT:
			a.Type = obj.TYPE_FCONST
			a.U.Dval = mpgetflt(n.Val.U.Fval)

		case CTINT,
			CTRUNE:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = Mpgetfix(n.Val.U.Xval)

		case CTSTR:
			datagostring(n.Val.U.Sval, a)

		case CTBOOL:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = int64(n.Val.U.Bval)

		case CTNIL:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = 0
		}

	case OADDR:
		Naddr(n.Left, a, canemitcode)
		a.Etype = uint8(Tptr)
		if Thearch.Thechar != '5' && Thearch.Thechar != '9' { // TODO(rsc): Do this even for arm, ppc64.
			a.Width = int64(Widthptr)
		}
		if a.Type != obj.TYPE_MEM {
			Fatal("naddr: OADDR %v (from %v)", Ctxt.Dconv(a), Oconv(int(n.Left.Op), 0))
		}
		a.Type = obj.TYPE_ADDR

		// itable of interface value
	case OITAB:
		Naddr(n.Left, a, canemitcode)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // itab(nil)
		}
		a.Etype = uint8(Tptr)
		a.Width = int64(Widthptr)

		// pointer in a string or slice
	case OSPTR:
		Naddr(n.Left, a, canemitcode)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // ptr(nil)
		}
		a.Etype = Simtype[Tptr]
		a.Offset += int64(Array_array)
		a.Width = int64(Widthptr)

		// len of string or slice
	case OLEN:
		Naddr(n.Left, a, canemitcode)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // len(nil)
		}
		a.Etype = Simtype[TUINT]
		if Thearch.Thechar == '9' {
			a.Etype = Simtype[TINT]
		}
		a.Offset += int64(Array_nel)
		if Thearch.Thechar != '5' { // TODO(rsc): Do this even on arm.
			a.Width = int64(Widthint)
		}

		// cap of string or slice
	case OCAP:
		Naddr(n.Left, a, canemitcode)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // cap(nil)
		}
		a.Etype = Simtype[TUINT]
		if Thearch.Thechar == '9' {
			a.Etype = Simtype[TINT]
		}
		a.Offset += int64(Array_cap)
		if Thearch.Thechar != '5' { // TODO(rsc): Do this even on arm.
			a.Width = int64(Widthint)
		}
	}
}

func newplist() *obj.Plist {
	pl := obj.Linknewplist(Ctxt)

	Pc = Ctxt.NewProg()
	Clearp(Pc)
	pl.Firstpc = Pc

	return pl
}

func nodarg(t *Type, fp int) *Node {
	var n *Node

	// entire argument struct, not just one arg
	if t.Etype == TSTRUCT && t.Funarg != 0 {
		n = Nod(ONAME, nil, nil)
		n.Sym = Lookup(".args")
		n.Type = t
		var savet Iter
		first := Structfirst(&savet, &t)
		if first == nil {
			Fatal("nodarg: bad struct")
		}
		if first.Width == BADWIDTH {
			Fatal("nodarg: offset not computed for %v", Tconv(t, 0))
		}
		n.Xoffset = first.Width
		n.Addable = 1
		goto fp
	}

	if t.Etype != TFIELD {
		Fatal("nodarg: not field %v", Tconv(t, 0))
	}

	if fp == 1 {
		var n *Node
		for l := Curfn.Dcl; l != nil; l = l.Next {
			n = l.N
			if (n.Class == PPARAM || n.Class == PPARAMOUT) && !isblanksym(t.Sym) && n.Sym == t.Sym {
				return n
			}
		}
	}

	n = Nod(ONAME, nil, nil)
	n.Type = t.Type
	n.Sym = t.Sym

	if t.Width == BADWIDTH {
		Fatal("nodarg: offset not computed for %v", Tconv(t, 0))
	}
	n.Xoffset = t.Width
	n.Addable = 1
	n.Orig = t.Nname

	// Rewrite argument named _ to __,
	// or else the assignment to _ will be
	// discarded during code generation.
fp:
	if isblank(n) {
		n.Sym = Lookup("__")
	}

	switch fp {
	case 0: // output arg
		n.Op = OINDREG

		n.Val.U.Reg = int16(Thearch.REGSP)
		if Thearch.Thechar == '5' {
			n.Xoffset += 4
		}
		if Thearch.Thechar == '9' {
			n.Xoffset += 8
		}

	case 1: // input arg
		n.Class = PPARAM

	case 2: // offset output arg
		Fatal("shouldn't be used")

		n.Op = OINDREG
		n.Val.U.Reg = int16(Thearch.REGSP)
		n.Xoffset += Types[Tptr].Width
	}

	n.Typecheck = 1
	return n
}

func Patch(p *obj.Prog, to *obj.Prog) {
	if p.To.Type != obj.TYPE_BRANCH {
		Fatal("patch: not a branch")
	}
	p.To.U.Branch = to
	p.To.Offset = to.Pc
}

func unpatch(p *obj.Prog) *obj.Prog {
	if p.To.Type != obj.TYPE_BRANCH {
		Fatal("unpatch: not a branch")
	}
	q := p.To.U.Branch
	p.To.U.Branch = nil
	p.To.Offset = 0
	return q
}
