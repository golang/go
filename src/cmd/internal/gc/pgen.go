// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"strings"
)

// "Portable" code generation.
// Compiled separately for 5g, 6g, and 8g, so allowed to use gg.h, opt.h.
// Must code to the intersection of the three back ends.

//#include	"opt.h"

var makefuncdatasym_nsym int32

func makefuncdatasym(namefmt string, funcdatakind int64) *Sym {
	var nod Node

	namebuf = fmt.Sprintf(namefmt, makefuncdatasym_nsym)
	makefuncdatasym_nsym++
	sym := Lookup(namebuf)
	pnod := newname(sym)
	pnod.Class = PEXTERN
	Nodconst(&nod, Types[TINT32], funcdatakind)
	Thearch.Gins(obj.AFUNCDATA, &nod, pnod)
	return sym
}

// gvardef inserts a VARDEF for n into the instruction stream.
// VARDEF is an annotation for the liveness analysis, marking a place
// where a complete initialization (definition) of a variable begins.
// Since the liveness analysis can see initialization of single-word
// variables quite easy, gvardef is usually only called for multi-word
// or 'fat' variables, those satisfying isfat(n->type).
// However, gvardef is also called when a non-fat variable is initialized
// via a block move; the only time this happens is when you have
//	return f()
// for a function with multiple return values exactly matching the return
// types of the current function.
//
// A 'VARDEF x' annotation in the instruction stream tells the liveness
// analysis to behave as though the variable x is being initialized at that
// point in the instruction stream. The VARDEF must appear before the
// actual (multi-instruction) initialization, and it must also appear after
// any uses of the previous value, if any. For example, if compiling:
//
//	x = x[1:]
//
// it is important to generate code like:
//
//	base, len, cap = pieces of x[1:]
//	VARDEF x
//	x = {base, len, cap}
//
// If instead the generated code looked like:
//
//	VARDEF x
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//
// then the liveness analysis would decide the previous value of x was
// unnecessary even though it is about to be used by the x[1:] computation.
// Similarly, if the generated code looked like:
//
//	base, len, cap = pieces of x[1:]
//	x = {base, len, cap}
//	VARDEF x
//
// then the liveness analysis will not preserve the new value of x, because
// the VARDEF appears to have "overwritten" it.
//
// VARDEF is a bit of a kludge to work around the fact that the instruction
// stream is working on single-word values but the liveness analysis
// wants to work on individual variables, which might be multi-word
// aggregates. It might make sense at some point to look into letting
// the liveness analysis work on single-word values as well, although
// there are complications around interface values, slices, and strings,
// all of which cannot be treated as individual words.
//
// VARKILL is the opposite of VARDEF: it marks a value as no longer needed,
// even if its address has been taken. That is, a VARKILL annotation asserts
// that its argument is certainly dead, for use when the liveness analysis
// would not otherwise be able to deduce that fact.

func gvardefx(n *Node, as int) {
	if n == nil {
		Fatal("gvardef nil")
	}
	if n.Op != ONAME {
		Yyerror("gvardef %v; %v", Oconv(int(n.Op), obj.FmtSharp), Nconv(n, 0))
		return
	}

	switch n.Class {
	case PAUTO,
		PPARAM,
		PPARAMOUT:
		Thearch.Gins(as, nil, n)
	}
}

func Gvardef(n *Node) {
	gvardefx(n, obj.AVARDEF)
}

func gvarkill(n *Node) {
	gvardefx(n, obj.AVARKILL)
}

func removevardef(firstp *obj.Prog) {
	for p := firstp; p != nil; p = p.Link {
		for p.Link != nil && (p.Link.As == obj.AVARDEF || p.Link.As == obj.AVARKILL) {
			p.Link = p.Link.Link
		}
		if p.To.Type == obj.TYPE_BRANCH {
			for p.To.U.Branch != nil && (p.To.U.Branch.As == obj.AVARDEF || p.To.U.Branch.As == obj.AVARKILL) {
				p.To.U.Branch = p.To.U.Branch.Link
			}
		}
	}
}

func gcsymdup(s *Sym) {
	ls := Linksym(s)
	if len(ls.R) > 0 {
		Fatal("cannot rosymdup %s with relocations", ls.Name)
	}
	var d MD5
	md5reset(&d)
	md5write(&d, ls.P, len(ls.P))
	var hi uint64
	lo := md5sum(&d, &hi)
	ls.Name = fmt.Sprintf("gclocals·%016x%016x", lo, hi)
	ls.Dupok = 1
}

func emitptrargsmap() {
	sym := Lookup(fmt.Sprintf("%s.args_stackmap", Curfn.Nname.Sym.Name))

	nptr := int(Curfn.Type.Argwid / int64(Widthptr))
	bv := bvalloc(int32(nptr) * 2)
	nbitmap := 1
	if Curfn.Type.Outtuple > 0 {
		nbitmap = 2
	}
	off := duint32(sym, 0, uint32(nbitmap))
	off = duint32(sym, off, uint32(bv.n))
	var xoffset int64
	if Curfn.Type.Thistuple > 0 {
		xoffset = 0
		twobitwalktype1(getthisx(Curfn.Type), &xoffset, bv)
	}

	if Curfn.Type.Intuple > 0 {
		xoffset = 0
		twobitwalktype1(getinargx(Curfn.Type), &xoffset, bv)
	}

	for j := 0; int32(j) < bv.n; j += 32 {
		off = duint32(sym, off, bv.b[j/32])
	}
	if Curfn.Type.Outtuple > 0 {
		xoffset = 0
		twobitwalktype1(getoutargx(Curfn.Type), &xoffset, bv)
		for j := 0; int32(j) < bv.n; j += 32 {
			off = duint32(sym, off, bv.b[j/32])
		}
	}

	ggloblsym(sym, int32(off), obj.RODATA)
}

// Sort the list of stack variables. Autos after anything else,
// within autos, unused after used, within used, things with
// pointers first, zeroed things first, and then decreasing size.
// Because autos are laid out in decreasing addresses
// on the stack, pointers first, zeroed things first and decreasing size
// really means, in memory, things with pointers needing zeroing at
// the top of the stack and increasing in size.
// Non-autos sort on offset.
func cmpstackvar(a *Node, b *Node) int {
	if a.Class != b.Class {
		if a.Class == PAUTO {
			return +1
		}
		return -1
	}

	if a.Class != PAUTO {
		if a.Xoffset < b.Xoffset {
			return -1
		}
		if a.Xoffset > b.Xoffset {
			return +1
		}
		return 0
	}

	if (a.Used == 0) != (b.Used == 0) {
		return int(b.Used) - int(a.Used)
	}

	ap := bool2int(haspointers(a.Type))
	bp := bool2int(haspointers(b.Type))
	if ap != bp {
		return bp - ap
	}

	ap = int(a.Needzero)
	bp = int(b.Needzero)
	if ap != bp {
		return bp - ap
	}

	if a.Type.Width < b.Type.Width {
		return +1
	}
	if a.Type.Width > b.Type.Width {
		return -1
	}

	return stringsCompare(a.Sym.Name, b.Sym.Name)
}

// TODO(lvd) find out where the PAUTO/OLITERAL nodes come from.
func allocauto(ptxt *obj.Prog) {
	Stksize = 0
	stkptrsize = 0

	if Curfn.Dcl == nil {
		return
	}

	// Mark the PAUTO's unused.
	for ll := Curfn.Dcl; ll != nil; ll = ll.Next {
		if ll.N.Class == PAUTO {
			ll.N.Used = 0
		}
	}

	markautoused(ptxt)

	listsort(&Curfn.Dcl, cmpstackvar)

	// Unused autos are at the end, chop 'em off.
	ll := Curfn.Dcl

	n := ll.N
	if n.Class == PAUTO && n.Op == ONAME && n.Used == 0 {
		// No locals used at all
		Curfn.Dcl = nil

		fixautoused(ptxt)
		return
	}

	for ll := Curfn.Dcl; ll.Next != nil; ll = ll.Next {
		n = ll.Next.N
		if n.Class == PAUTO && n.Op == ONAME && n.Used == 0 {
			ll.Next = nil
			Curfn.Dcl.End = ll
			break
		}
	}

	// Reassign stack offsets of the locals that are still there.
	var w int64
	for ll := Curfn.Dcl; ll != nil; ll = ll.Next {
		n = ll.N
		if n.Class != PAUTO || n.Op != ONAME {
			continue
		}

		dowidth(n.Type)
		w = n.Type.Width
		if w >= Thearch.MAXWIDTH || w < 0 {
			Fatal("bad width")
		}
		Stksize += w
		Stksize = Rnd(Stksize, int64(n.Type.Align))
		if haspointers(n.Type) {
			stkptrsize = Stksize
		}
		if Thearch.Thechar == '5' || Thearch.Thechar == '9' {
			Stksize = Rnd(Stksize, int64(Widthptr))
		}
		if Stksize >= 1<<31 {
			setlineno(Curfn)
			Yyerror("stack frame too large (>2GB)")
		}

		n.Stkdelta = -Stksize - n.Xoffset
	}

	Stksize = Rnd(Stksize, int64(Widthreg))
	stkptrsize = Rnd(stkptrsize, int64(Widthreg))

	fixautoused(ptxt)

	// The debug information needs accurate offsets on the symbols.
	for ll := Curfn.Dcl; ll != nil; ll = ll.Next {
		if ll.N.Class != PAUTO || ll.N.Op != ONAME {
			continue
		}
		ll.N.Xoffset += ll.N.Stkdelta
		ll.N.Stkdelta = 0
	}
}

func movelarge(l *NodeList) {
	for ; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC {
			movelargefn(l.N)
		}
	}
}

func movelargefn(fn *Node) {
	var n *Node

	for l := fn.Dcl; l != nil; l = l.Next {
		n = l.N
		if n.Class == PAUTO && n.Type != nil && n.Type.Width > MaxStackVarSize {
			addrescapes(n)
		}
	}
}

func Cgen_checknil(n *Node) {
	if Disable_checknil != 0 {
		return
	}

	// Ideally we wouldn't see any integer types here, but we do.
	if n.Type == nil || (Isptr[n.Type.Etype] == 0 && Isint[n.Type.Etype] == 0 && n.Type.Etype != TUNSAFEPTR) {
		Dump("checknil", n)
		Fatal("bad checknil")
	}

	if ((Thearch.Thechar == '5' || Thearch.Thechar == '9') && n.Op != OREGISTER) || n.Addable == 0 || n.Op == OLITERAL {
		var reg Node
		Thearch.Regalloc(&reg, Types[Tptr], n)
		Thearch.Cgen(n, &reg)
		Thearch.Gins(obj.ACHECKNIL, &reg, nil)
		Thearch.Regfree(&reg)
		return
	}

	Thearch.Gins(obj.ACHECKNIL, n, nil)
}

/*
 * ggen.c
 */
func compile(fn *Node) {
	if Newproc == nil {
		Newproc = Sysfunc("newproc")
		Deferproc = Sysfunc("deferproc")
		Deferreturn = Sysfunc("deferreturn")
		Panicindex = Sysfunc("panicindex")
		panicslice = Sysfunc("panicslice")
		throwreturn = Sysfunc("throwreturn")
	}

	lno := setlineno(fn)

	Curfn = fn
	dowidth(Curfn.Type)

	var oldstksize int64
	var nod1 Node
	var ptxt *obj.Prog
	var pl *obj.Plist
	var p *obj.Prog
	var n *Node
	var nam *Node
	var gcargs *Sym
	var gclocals *Sym
	if fn.Nbody == nil {
		if pure_go != 0 || strings.HasPrefix(fn.Nname.Sym.Name, "init.") {
			Yyerror("missing function body", fn)
			goto ret
		}

		if Debug['A'] != 0 {
			goto ret
		}
		emitptrargsmap()
		goto ret
	}

	saveerrors()

	// set up domain for labels
	clearlabels()

	if Curfn.Type.Outnamed != 0 {
		// add clearing of the output parameters
		var save Iter
		t := Structfirst(&save, Getoutarg(Curfn.Type))

		for t != nil {
			if t.Nname != nil {
				n = Nod(OAS, t.Nname, nil)
				typecheck(&n, Etop)
				Curfn.Nbody = concat(list1(n), Curfn.Nbody)
			}

			t = structnext(&save)
		}
	}

	order(Curfn)
	if nerrors != 0 {
		goto ret
	}

	Hasdefer = 0
	walk(Curfn)
	if nerrors != 0 {
		goto ret
	}
	if flag_race != 0 {
		racewalk(Curfn)
	}
	if nerrors != 0 {
		goto ret
	}

	continpc = nil
	breakpc = nil

	pl = newplist()
	pl.Name = Linksym(Curfn.Nname.Sym)

	setlineno(Curfn)

	Nodconst(&nod1, Types[TINT32], 0)
	nam = Curfn.Nname
	if isblank(nam) {
		nam = nil
	}
	ptxt = Thearch.Gins(obj.ATEXT, nam, &nod1)
	if fn.Dupok != 0 {
		ptxt.From3.Offset |= obj.DUPOK
	}
	if fn.Wrapper != 0 {
		ptxt.From3.Offset |= obj.WRAPPER
	}
	if fn.Needctxt {
		ptxt.From3.Offset |= obj.NEEDCTXT
	}
	if fn.Nosplit {
		ptxt.From3.Offset |= obj.NOSPLIT
	}

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if myimportpath != "" && myimportpath == "reflect" {
		if Curfn.Nname.Sym.Name == "callReflect" || Curfn.Nname.Sym.Name == "callMethod" {
			ptxt.From3.Offset |= obj.WRAPPER
		}
	}

	Afunclit(&ptxt.From, Curfn.Nname)

	Thearch.Ginit()

	gcargs = makefuncdatasym("gcargs·%d", obj.FUNCDATA_ArgsPointerMaps)
	gclocals = makefuncdatasym("gclocals·%d", obj.FUNCDATA_LocalsPointerMaps)

	for t := Curfn.Paramfld; t != nil; t = t.Down {
		gtrack(tracksym(t.Type))
	}

	for l := fn.Dcl; l != nil; l = l.Next {
		n = l.N
		if n.Op != ONAME { // might be OTYPE or OLITERAL
			continue
		}
		switch n.Class {
		case PAUTO,
			PPARAM,
			PPARAMOUT:
			Nodconst(&nod1, Types[TUINTPTR], l.N.Type.Width)
			p = Thearch.Gins(obj.ATYPE, l.N, &nod1)
			p.From.Gotype = Linksym(ngotype(l.N))
		}
	}

	Genlist(Curfn.Enter)
	Genlist(Curfn.Nbody)
	Thearch.Gclean()
	checklabels()
	if nerrors != 0 {
		goto ret
	}
	if Curfn.Endlineno != 0 {
		lineno = Curfn.Endlineno
	}

	if Curfn.Type.Outtuple != 0 {
		Thearch.Ginscall(throwreturn, 0)
	}

	Thearch.Ginit()

	// TODO: Determine when the final cgen_ret can be omitted. Perhaps always?
	Thearch.Cgen_ret(nil)

	if Hasdefer != 0 {
		// deferreturn pretends to have one uintptr argument.
		// Reserve space for it so stack scanner is happy.
		if Maxarg < int64(Widthptr) {
			Maxarg = int64(Widthptr)
		}
	}

	Thearch.Gclean()
	if nerrors != 0 {
		goto ret
	}

	Pc.As = obj.ARET // overwrite AEND
	Pc.Lineno = lineno

	fixjmp(ptxt)
	if Debug['N'] == 0 || Debug['R'] != 0 || Debug['P'] != 0 {
		regopt(ptxt)
		nilopt(ptxt)
	}

	Thearch.Expandchecks(ptxt)

	oldstksize = Stksize
	allocauto(ptxt)

	if false {
		fmt.Printf("allocauto: %d to %d\n", oldstksize, int64(Stksize))
	}

	setlineno(Curfn)
	if int64(Stksize)+Maxarg > 1<<31 {
		Yyerror("stack frame too large (>2GB)")
		goto ret
	}

	// Emit garbage collection symbols.
	liveness(Curfn, ptxt, gcargs, gclocals)

	gcsymdup(gcargs)
	gcsymdup(gclocals)

	Thearch.Defframe(ptxt)

	if Debug['f'] != 0 {
		frame(0)
	}

	// Remove leftover instrumentation from the instruction stream.
	removevardef(ptxt)

ret:
	lineno = lno
}
