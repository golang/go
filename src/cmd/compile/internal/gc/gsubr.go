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

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
	"fmt"
)

var (
	ddumped bool
	dfirst  *obj.Prog
	dpc     *obj.Prog
)

func Prog(as obj.As) *obj.Prog {
	var p *obj.Prog

	if as == obj.AGLOBL {
		if ddumped {
			Fatalf("already dumped data")
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

	if lineno == 0 && Debug['K'] != 0 {
		Warn("prog: line 0")
	}

	p.As = as
	p.Lineno = lineno
	return p
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

func dumpdata() {
	ddumped = true
	if dfirst == nil {
		return
	}
	newplist()
	*Pc = *dfirst
	Pc = dpc
	Clearp(Pc)
}

func flushdata() {
	if dfirst == nil {
		return
	}
	newplist()
	*Pc = *dfirst
	Pc = dpc
	Clearp(Pc)
	dfirst = nil
	dpc = nil
}

// Fixup instructions after allocauto (formerly compactframe) has moved all autos around.
func fixautoused(p *obj.Prog) {
	for lp := &p; ; {
		p = *lp
		if p == nil {
			break
		}
		if p.As == obj.ATYPE && p.From.Node != nil && p.From.Name == obj.NAME_AUTO && !((p.From.Node).(*Node)).Used {
			*lp = p.Link
			continue
		}

		if (p.As == obj.AVARDEF || p.As == obj.AVARKILL || p.As == obj.AVARLIVE) && p.To.Node != nil && !((p.To.Node).(*Node)).Used {
			// Cannot remove VARDEF instruction, because - unlike TYPE handled above -
			// VARDEFs are interspersed with other code, and a jump might be using the
			// VARDEF as a target. Replace with a no-op instead. A later pass will remove
			// the no-ops.
			obj.Nopout(p)

			continue
		}

		if p.From.Name == obj.NAME_AUTO && p.From.Node != nil {
			p.From.Offset += stkdelta[p.From.Node.(*Node)]
		}

		if p.To.Name == obj.NAME_AUTO && p.To.Node != nil {
			p.To.Offset += stkdelta[p.To.Node.(*Node)]
		}

		lp = &p.Link
	}
}

func ggloblnod(nam *Node) {
	p := Gins(obj.AGLOBL, nam, nil)
	p.Lineno = nam.Lineno
	p.From.Sym.Gotype = Linksym(ngotype(nam))
	p.To.Sym = nil
	p.To.Type = obj.TYPE_CONST
	p.To.Offset = nam.Type.Width
	p.From3 = new(obj.Addr)
	if nam.Name.Readonly {
		p.From3.Offset = obj.RODATA
	}
	if nam.Type != nil && !haspointers(nam.Type) {
		p.From3.Offset |= obj.NOPTR
	}
}

func ggloblsym(s *Sym, width int32, flags int16) {
	ggloblLSym(Linksym(s), width, flags)
}

func ggloblLSym(s *obj.LSym, width int32, flags int16) {
	p := Gins(obj.AGLOBL, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = s
	if flags&obj.LOCAL != 0 {
		p.From.Sym.Local = true
		flags &^= obj.LOCAL
	}
	p.To.Type = obj.TYPE_CONST
	p.To.Offset = int64(width)
	p.From3 = new(obj.Addr)
	p.From3.Offset = int64(flags)
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

// Sweep the prog list to mark any used nodes.
func markautoused(p *obj.Prog) {
	for ; p != nil; p = p.Link {
		if p.As == obj.ATYPE || p.As == obj.AVARDEF || p.As == obj.AVARKILL {
			continue
		}

		if p.From.Node != nil {
			((p.From.Node).(*Node)).Used = true
		}

		if p.To.Node != nil {
			((p.To.Node).(*Node)).Used = true
		}
	}
}

// Naddr rewrites a to refer to n.
// It assumes that a is zeroed on entry.
func Naddr(a *obj.Addr, n *Node) {
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
		a := a // copy to let escape into Ctxt.Dconv
		Debug['h'] = 1
		Dump("naddr", n)
		Fatalf("naddr: bad %v %v", n.Op, Ctxt.Dconv(a))

	case OREGISTER:
		a.Type = obj.TYPE_REG
		a.Reg = n.Reg
		a.Sym = nil
		if Thearch.LinkArch.Family == sys.I386 { // TODO(rsc): Never clear a->width.
			a.Width = 0
		}

	case OINDREG:
		a.Type = obj.TYPE_MEM
		a.Reg = n.Reg
		a.Sym = Linksym(n.Sym)
		a.Offset = n.Xoffset
		if a.Offset != int64(int32(a.Offset)) {
			yyerror("offset %d too large for OINDREG", a.Offset)
		}
		if Thearch.LinkArch.Family == sys.I386 { // TODO(rsc): Never clear a->width.
			a.Width = 0
		}

	case OCLOSUREVAR:
		if !Curfn.Func.Needctxt {
			Fatalf("closurevar without needctxt")
		}
		a.Type = obj.TYPE_MEM
		a.Reg = int16(Thearch.REGCTXT)
		a.Sym = nil
		a.Offset = n.Xoffset

	case OCFUNC:
		Naddr(a, n.Left)
		a.Sym = Linksym(n.Left.Sym)

	case ONAME:
		a.Etype = 0
		if n.Type != nil {
			a.Etype = uint8(simtype[n.Type.Etype])
		}
		a.Offset = n.Xoffset
		s := n.Sym
		a.Node = n.Orig

		//if(a->node >= (Node*)&n)
		//	fatal("stack node");
		if s == nil {
			s = lookup(".noname")
		}
		if n.Name.Method && n.Type != nil && n.Type.Sym != nil && n.Type.Sym.Pkg != nil {
			s = Pkglookup(s.Name, n.Type.Sym.Pkg)
		}

		a.Type = obj.TYPE_MEM
		switch n.Class {
		default:
			Fatalf("naddr: ONAME class %v %d\n", n.Sym, n.Class)

		case PEXTERN:
			a.Name = obj.NAME_EXTERN

		case PAUTO:
			a.Name = obj.NAME_AUTO

		case PPARAM, PPARAMOUT:
			a.Name = obj.NAME_PARAM

		case PFUNC:
			a.Name = obj.NAME_EXTERN
			a.Type = obj.TYPE_ADDR
			a.Width = int64(Widthptr)
			s = funcsym(s)
		}

		a.Sym = Linksym(s)

	case ODOT:
		// A special case to make write barriers more efficient.
		// Taking the address of the first field of a named struct
		// is the same as taking the address of the struct.
		if !n.Left.Type.IsStruct() || n.Left.Type.Field(0).Sym != n.Sym {
			Debug['h'] = 1
			Dump("naddr", n)
			Fatalf("naddr: bad %v %v", n.Op, Ctxt.Dconv(a))
		}
		Naddr(a, n.Left)

	case OLITERAL:
		if Thearch.LinkArch.Family == sys.I386 {
			a.Width = 0
		}
		switch u := n.Val().U.(type) {
		default:
			Fatalf("naddr: const %L", n.Type)

		case *Mpflt:
			a.Type = obj.TYPE_FCONST
			a.Val = u.Float64()

		case *Mpint:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = u.Int64()

		case string:
			datagostring(u, a)

		case bool:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = int64(obj.Bool2int(u))

		case *NilVal:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = 0
		}

	case OADDR:
		Naddr(a, n.Left)
		a.Etype = uint8(Tptr)
		if !Thearch.LinkArch.InFamily(sys.MIPS64, sys.ARM, sys.ARM64, sys.PPC64, sys.S390X) { // TODO(rsc): Do this even for these architectures.
			a.Width = int64(Widthptr)
		}
		if a.Type != obj.TYPE_MEM {
			a := a // copy to let escape into Ctxt.Dconv
			Fatalf("naddr: OADDR %v (from %v)", Ctxt.Dconv(a), n.Left.Op)
		}
		a.Type = obj.TYPE_ADDR

	case OITAB:
		// itable of interface value
		Naddr(a, n.Left)
		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // itab(nil)
		}
		a.Etype = uint8(Tptr)
		a.Width = int64(Widthptr)

	case OIDATA:
		// idata of interface value
		Naddr(a, n.Left)
		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // idata(nil)
		}
		if isdirectiface(n.Type) {
			a.Etype = uint8(simtype[n.Type.Etype])
		} else {
			a.Etype = uint8(Tptr)
		}
		a.Offset += int64(Widthptr)
		a.Width = int64(Widthptr)

		// pointer in a string or slice
	case OSPTR:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // ptr(nil)
		}
		a.Etype = uint8(simtype[Tptr])
		a.Offset += int64(array_array)
		a.Width = int64(Widthptr)

		// len of string or slice
	case OLEN:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // len(nil)
		}
		a.Etype = uint8(simtype[TUINT])
		a.Offset += int64(array_nel)
		if Thearch.LinkArch.Family != sys.ARM { // TODO(rsc): Do this even on arm.
			a.Width = int64(Widthint)
		}

		// cap of string or slice
	case OCAP:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // cap(nil)
		}
		a.Etype = uint8(simtype[TUINT])
		a.Offset += int64(array_cap)
		if Thearch.LinkArch.Family != sys.ARM { // TODO(rsc): Do this even on arm.
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
		n = Nod(ONAME, nil, nil)
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
		n = Nod(ONAME, nil, nil)
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
		n.Op = OINDREG
		n.Reg = int16(Thearch.REGSP)
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
		obj.ATEXT, obj.AFUNCDATA, obj.AUSEFIELD, obj.AGLOBL:
	default:
		Fatalf("unhandled gins op %v", as)
	}

	p := Prog(as)
	Naddr(&p.From, f)
	Naddr(&p.To, t)

	if Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}
	return p
}
