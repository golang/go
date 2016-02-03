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

import (
	"cmd/internal/obj"
	"fmt"
	"runtime"
	"strings"
)

var ddumped int

var dfirst *obj.Prog

var dpc *obj.Prog

// Is this node a memory operand?
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
	if a.Reg != b.Reg {
		return false
	}
	return true
}

func Gbranch(as int, t *Type, likely int) *obj.Prog {
	p := Prog(as)
	p.To.Type = obj.TYPE_BRANCH
	p.To.Val = nil
	if as != obj.AJMP && likely != 0 && Thearch.Thechar != '9' && Thearch.Thechar != '7' && Thearch.Thechar != '0' {
		p.From.Type = obj.TYPE_CONST
		if likely > 0 {
			p.From.Offset = 1
		}
	}

	if Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	return p
}

func Prog(as int) *obj.Prog {
	var p *obj.Prog

	if as == obj.ADATA || as == obj.AGLOBL {
		if ddumped != 0 {
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
		Fatalf("nodreg: t nil")
	}

	*n = Node{}
	n.Op = OREGISTER
	n.Addable = true
	ullmancalc(n)
	n.Reg = int16(r)
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
	p := Thearch.Gins(obj.AGLOBL, nam, nil)
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
	p := Thearch.Gins(obj.AGLOBL, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
	if flags&obj.LOCAL != 0 {
		p.From.Sym.Local = true
		flags &= ^obj.LOCAL
	}
	p.To.Type = obj.TYPE_CONST
	p.To.Offset = int64(width)
	p.From3 = new(obj.Addr)
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
		case TSTRUCT, TARRAY, TSTRING,
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
		Fatalf("naddr: bad %v %v", Oconv(int(n.Op), 0), Ctxt.Dconv(a))

	case OREGISTER:
		a.Type = obj.TYPE_REG
		a.Reg = n.Reg
		a.Sym = nil
		if Thearch.Thechar == '8' { // TODO(rsc): Never clear a->width.
			a.Width = 0
		}

	case OINDREG:
		a.Type = obj.TYPE_MEM
		a.Reg = n.Reg
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
		a.Etype = uint8(Simtype[n.Left.Type.Etype])

		a.Width = n.Left.Type.Width
		a.Offset = n.Xoffset
		a.Sym = Linksym(n.Left.Sym)
		a.Type = obj.TYPE_MEM
		a.Name = obj.NAME_PARAM
		a.Node = n.Left.Orig

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
			a.Etype = uint8(Simtype[n.Type.Etype])
		}
		a.Offset = n.Xoffset
		s := n.Sym
		a.Node = n.Orig

		//if(a->node >= (Node*)&n)
		//	fatal("stack node");
		if s == nil {
			s = Lookup(".noname")
		}
		if n.Name.Method {
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
		if n.Left.Type.Etype != TSTRUCT || n.Left.Type.Type.Sym != n.Right.Sym {
			Debug['h'] = 1
			Dump("naddr", n)
			Fatalf("naddr: bad %v %v", Oconv(int(n.Op), 0), Ctxt.Dconv(a))
		}
		Naddr(a, n.Left)

	case OLITERAL:
		if Thearch.Thechar == '8' {
			a.Width = 0
		}
		switch n.Val().Ctype() {
		default:
			Fatalf("naddr: const %v", Tconv(n.Type, obj.FmtLong))

		case CTFLT:
			a.Type = obj.TYPE_FCONST
			a.Val = mpgetflt(n.Val().U.(*Mpflt))

		case CTINT, CTRUNE:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = Mpgetfix(n.Val().U.(*Mpint))

		case CTSTR:
			datagostring(n.Val().U.(string), a)

		case CTBOOL:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = int64(obj.Bool2int(n.Val().U.(bool)))

		case CTNIL:
			a.Sym = nil
			a.Type = obj.TYPE_CONST
			a.Offset = 0
		}

	case OADDR:
		Naddr(a, n.Left)
		a.Etype = uint8(Tptr)
		if Thearch.Thechar != '0' && Thearch.Thechar != '5' && Thearch.Thechar != '7' && Thearch.Thechar != '9' { // TODO(rsc): Do this even for arm, ppc64.
			a.Width = int64(Widthptr)
		}
		if a.Type != obj.TYPE_MEM {
			a := a // copy to let escape into Ctxt.Dconv
			Fatalf("naddr: OADDR %v (from %v)", Ctxt.Dconv(a), Oconv(int(n.Left.Op), 0))
		}
		a.Type = obj.TYPE_ADDR

		// itable of interface value
	case OITAB:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // itab(nil)
		}
		a.Etype = uint8(Tptr)
		a.Width = int64(Widthptr)

		// pointer in a string or slice
	case OSPTR:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // ptr(nil)
		}
		a.Etype = uint8(Simtype[Tptr])
		a.Offset += int64(Array_array)
		a.Width = int64(Widthptr)

		// len of string or slice
	case OLEN:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // len(nil)
		}
		a.Etype = uint8(Simtype[TUINT])
		a.Offset += int64(Array_nel)
		if Thearch.Thechar != '5' { // TODO(rsc): Do this even on arm.
			a.Width = int64(Widthint)
		}

		// cap of string or slice
	case OCAP:
		Naddr(a, n.Left)

		if a.Type == obj.TYPE_CONST && a.Offset == 0 {
			break // cap(nil)
		}
		a.Etype = uint8(Simtype[TUINT])
		a.Offset += int64(Array_cap)
		if Thearch.Thechar != '5' { // TODO(rsc): Do this even on arm.
			a.Width = int64(Widthint)
		}
	}
	return
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
	if t.Etype == TSTRUCT && t.Funarg {
		n = Nod(ONAME, nil, nil)
		n.Sym = Lookup(".args")
		n.Type = t
		var savet Iter
		first := Structfirst(&savet, &t)
		if first == nil {
			Fatalf("nodarg: bad struct")
		}
		if first.Width == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = first.Width
		n.Addable = true
		goto fp
	}

	if t.Etype != TFIELD {
		Fatalf("nodarg: not field %v", t)
	}

	if fp == 1 {
		var n *Node
		for l := Curfn.Func.Dcl; l != nil; l = l.Next {
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
		Fatalf("nodarg: offset not computed for %v", t)
	}
	n.Xoffset = t.Width
	n.Addable = true
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

		n.Reg = int16(Thearch.REGSP)
		n.Xoffset += Ctxt.FixedFrameSize()

	case 1: // input arg
		n.Class = PPARAM

	case 2: // offset output arg
		Fatalf("shouldn't be used")
	}

	n.Typecheck = 1
	return n
}

func Patch(p *obj.Prog, to *obj.Prog) {
	if p.To.Type != obj.TYPE_BRANCH {
		Fatalf("patch: not a branch")
	}
	p.To.Val = to
	p.To.Offset = to.Pc
}

func unpatch(p *obj.Prog) *obj.Prog {
	if p.To.Type != obj.TYPE_BRANCH {
		Fatalf("unpatch: not a branch")
	}
	q, _ := p.To.Val.(*obj.Prog)
	p.To.Val = nil
	p.To.Offset = 0
	return q
}

var reg [100]int       // count of references to reg
var regstk [100][]byte // allocation sites, when -v is given

func GetReg(r int) int {
	return reg[r-Thearch.REGMIN]
}
func SetReg(r, v int) {
	reg[r-Thearch.REGMIN] = v
}

func ginit() {
	for r := range reg {
		reg[r] = 1
	}

	for r := Thearch.REGMIN; r <= Thearch.REGMAX; r++ {
		reg[r-Thearch.REGMIN] = 0
	}
	for r := Thearch.FREGMIN; r <= Thearch.FREGMAX; r++ {
		reg[r-Thearch.REGMIN] = 0
	}

	for _, r := range Thearch.ReservedRegs {
		reg[r-Thearch.REGMIN] = 1
	}
}

func gclean() {
	for _, r := range Thearch.ReservedRegs {
		reg[r-Thearch.REGMIN]--
	}

	for r := Thearch.REGMIN; r <= Thearch.REGMAX; r++ {
		n := reg[r-Thearch.REGMIN]
		if n != 0 {
			if Debug['v'] != 0 {
				Regdump()
			}
			Yyerror("reg %v left allocated", obj.Rconv(r))
		}
	}

	for r := Thearch.FREGMIN; r <= Thearch.FREGMAX; r++ {
		n := reg[r-Thearch.REGMIN]
		if n != 0 {
			if Debug['v'] != 0 {
				Regdump()
			}
			Yyerror("reg %v left allocated", obj.Rconv(r))
		}
	}
}

func Anyregalloc() bool {
	n := 0
	for r := Thearch.REGMIN; r <= Thearch.REGMAX; r++ {
		if reg[r-Thearch.REGMIN] == 0 {
			n++
		}
	}
	return n > len(Thearch.ReservedRegs)
}

// allocate register of type t, leave in n.
// if o != N, o may be reusable register.
// caller must Regfree(n).
func Regalloc(n *Node, t *Type, o *Node) {
	if t == nil {
		Fatalf("regalloc: t nil")
	}
	et := Simtype[t.Etype]
	if Ctxt.Arch.Regsize == 4 && (et == TINT64 || et == TUINT64) {
		Fatalf("regalloc 64bit")
	}

	var i int
Switch:
	switch et {
	default:
		Fatalf("regalloc: unknown type %v", t)

	case TINT8, TUINT8, TINT16, TUINT16, TINT32, TUINT32, TINT64, TUINT64, TPTR32, TPTR64, TBOOL:
		if o != nil && o.Op == OREGISTER {
			i = int(o.Reg)
			if Thearch.REGMIN <= i && i <= Thearch.REGMAX {
				break Switch
			}
		}
		for i = Thearch.REGMIN; i <= Thearch.REGMAX; i++ {
			if reg[i-Thearch.REGMIN] == 0 {
				break Switch
			}
		}
		Flusherrors()
		Regdump()
		Fatalf("out of fixed registers")

	case TFLOAT32, TFLOAT64:
		if Thearch.Use387 {
			i = Thearch.FREGMIN // x86.REG_F0
			break Switch
		}
		if o != nil && o.Op == OREGISTER {
			i = int(o.Reg)
			if Thearch.FREGMIN <= i && i <= Thearch.FREGMAX {
				break Switch
			}
		}
		for i = Thearch.FREGMIN; i <= Thearch.FREGMAX; i++ {
			if reg[i-Thearch.REGMIN] == 0 { // note: REGMIN, not FREGMIN
				break Switch
			}
		}
		Flusherrors()
		Regdump()
		Fatalf("out of floating registers")

	case TCOMPLEX64, TCOMPLEX128:
		Tempname(n, t)
		return
	}

	ix := i - Thearch.REGMIN
	if reg[ix] == 0 && Debug['v'] > 0 {
		if regstk[ix] == nil {
			regstk[ix] = make([]byte, 4096)
		}
		stk := regstk[ix]
		n := runtime.Stack(stk[:cap(stk)], false)
		regstk[ix] = stk[:n]
	}
	reg[ix]++
	Nodreg(n, t, i)
}

func Regfree(n *Node) {
	if n.Op == ONAME {
		return
	}
	if n.Op != OREGISTER && n.Op != OINDREG {
		Fatalf("regfree: not a register")
	}
	i := int(n.Reg)
	if i == Thearch.REGSP {
		return
	}
	switch {
	case Thearch.REGMIN <= i && i <= Thearch.REGMAX,
		Thearch.FREGMIN <= i && i <= Thearch.FREGMAX:
		// ok
	default:
		Fatalf("regfree: reg out of range")
	}

	i -= Thearch.REGMIN
	if reg[i] <= 0 {
		Fatalf("regfree: reg not allocated")
	}
	reg[i]--
	if reg[i] == 0 {
		regstk[i] = regstk[i][:0]
	}
}

// Reginuse reports whether r is in use.
func Reginuse(r int) bool {
	switch {
	case Thearch.REGMIN <= r && r <= Thearch.REGMAX,
		Thearch.FREGMIN <= r && r <= Thearch.FREGMAX:
		// ok
	default:
		Fatalf("reginuse: reg out of range")
	}

	return reg[r-Thearch.REGMIN] > 0
}

// Regrealloc(n) undoes the effect of Regfree(n),
// so that a register can be given up but then reclaimed.
func Regrealloc(n *Node) {
	if n.Op != OREGISTER && n.Op != OINDREG {
		Fatalf("regrealloc: not a register")
	}
	i := int(n.Reg)
	if i == Thearch.REGSP {
		return
	}
	switch {
	case Thearch.REGMIN <= i && i <= Thearch.REGMAX,
		Thearch.FREGMIN <= i && i <= Thearch.FREGMAX:
		// ok
	default:
		Fatalf("regrealloc: reg out of range")
	}

	i -= Thearch.REGMIN
	if reg[i] == 0 && Debug['v'] > 0 {
		if regstk[i] == nil {
			regstk[i] = make([]byte, 4096)
		}
		stk := regstk[i]
		n := runtime.Stack(stk[:cap(stk)], false)
		regstk[i] = stk[:n]
	}
	reg[i]++
}

func Regdump() {
	if Debug['v'] == 0 {
		fmt.Printf("run compiler with -v for register allocation sites\n")
		return
	}

	dump := func(r int) {
		stk := regstk[r-Thearch.REGMIN]
		if len(stk) == 0 {
			return
		}
		fmt.Printf("reg %v allocated at:\n", obj.Rconv(r))
		fmt.Printf("\t%s\n", strings.Replace(strings.TrimSpace(string(stk)), "\n", "\n\t", -1))
	}

	for r := Thearch.REGMIN; r <= Thearch.REGMAX; r++ {
		if reg[r-Thearch.REGMIN] != 0 {
			dump(r)
		}
	}
	for r := Thearch.FREGMIN; r <= Thearch.FREGMAX; r++ {
		if reg[r-Thearch.REGMIN] == 0 {
			dump(r)
		}
	}
}
