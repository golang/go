// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

// portable half of code generator.
// mainly statements and control flow.
var labellist *Label

var lastlabel *Label

func Sysfunc(name string) *Node {
	n := newname(Pkglookup(name, Runtimepkg))
	n.Class = PFUNC
	return n
}

// addrescapes tags node n as having had its address taken
// by "increasing" the "value" of n.Esc to EscHeap.
// Storage is allocated as necessary to allow the address
// to be taken.
func addrescapes(n *Node) {
	switch n.Op {
	// probably a type error already.
	// dump("addrescapes", n);
	default:
		break

	case ONAME:
		if n == nodfp {
			break
		}

		// if this is a tmpname (PAUTO), it was tagged by tmpname as not escaping.
		// on PPARAM it means something different.
		if n.Class == PAUTO && n.Esc == EscNever {
			break
		}

		switch n.Class {
		case PPARAMREF:
			addrescapes(n.Name.Defn)

		// if func param, need separate temporary
		// to hold heap pointer.
		// the function type has already been checked
		// (we're in the function body)
		// so the param already has a valid xoffset.

		// expression to refer to stack copy
		case PPARAM, PPARAMOUT:
			n.Name.Param.Stackparam = Nod(OPARAM, n, nil)

			n.Name.Param.Stackparam.Type = n.Type
			n.Name.Param.Stackparam.Addable = true
			if n.Xoffset == BADWIDTH {
				Fatalf("addrescapes before param assignment")
			}
			n.Name.Param.Stackparam.Xoffset = n.Xoffset
			fallthrough

		case PAUTO:
			n.Class |= PHEAP

			n.Addable = false
			n.Ullman = 2
			n.Xoffset = 0

			// create stack variable to hold pointer to heap
			oldfn := Curfn

			Curfn = n.Name.Curfn
			n.Name.Heapaddr = temp(Ptrto(n.Type))
			buf := fmt.Sprintf("&%v", n.Sym)
			n.Name.Heapaddr.Sym = Lookup(buf)
			n.Name.Heapaddr.Orig.Sym = n.Name.Heapaddr.Sym
			n.Esc = EscHeap
			if Debug['m'] != 0 {
				fmt.Printf("%v: moved to heap: %v\n", n.Line(), n)
			}
			Curfn = oldfn
		}

	case OIND, ODOTPTR:
		break

	// ODOTPTR has already been introduced,
	// so these are the non-pointer ODOT and OINDEX.
	// In &x[0], if x is a slice, then x does not
	// escape--the pointer inside x does, but that
	// is always a heap pointer anyway.
	case ODOT, OINDEX, OPAREN, OCONVNOP:
		if !Isslice(n.Left.Type) {
			addrescapes(n.Left)
		}
	}
}

func clearlabels() {
	for l := labellist; l != nil; l = l.Link {
		l.Sym.Label = nil
	}

	labellist = nil
	lastlabel = nil
}

func newlab(n *Node) *Label {
	s := n.Left.Sym
	lab := s.Label
	if lab == nil {
		lab = new(Label)
		if lastlabel == nil {
			labellist = lab
		} else {
			lastlabel.Link = lab
		}
		lastlabel = lab
		lab.Sym = s
		s.Label = lab
	}

	if n.Op == OLABEL {
		if lab.Def != nil {
			Yyerror("label %v already defined at %v", s, lab.Def.Line())
		} else {
			lab.Def = n
		}
	} else {
		lab.Use = append(lab.Use, n)
	}

	return lab
}

func checkgoto(from *Node, to *Node) {
	if from.Sym == to.Sym {
		return
	}

	nf := 0
	for fs := from.Sym; fs != nil; fs = fs.Link {
		nf++
	}
	nt := 0
	for fs := to.Sym; fs != nil; fs = fs.Link {
		nt++
	}
	fs := from.Sym
	for ; nf > nt; nf-- {
		fs = fs.Link
	}
	if fs != to.Sym {
		lno := int(lineno)
		setlineno(from)

		// decide what to complain about.
		// prefer to complain about 'into block' over declarations,
		// so scan backward to find most recent block or else dcl.
		var block *Sym

		var dcl *Sym
		ts := to.Sym
		for ; nt > nf; nt-- {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
		}

		for ts != fs {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
			fs = fs.Link
		}

		if block != nil {
			Yyerror("goto %v jumps into block starting at %v", from.Left.Sym, Ctxt.Line(int(block.Lastlineno)))
		} else {
			Yyerror("goto %v jumps over declaration of %v at %v", from.Left.Sym, dcl, Ctxt.Line(int(dcl.Lastlineno)))
		}
		lineno = int32(lno)
	}
}

func stmtlabel(n *Node) *Label {
	if n.Sym != nil {
		lab := n.Sym.Label
		if lab != nil {
			if lab.Def != nil {
				if lab.Def.Name.Defn == n {
					return lab
				}
			}
		}
	}
	return nil
}

// compile statements
func Genlist(l *NodeList) {
	for ; l != nil; l = l.Next {
		gen(l.N)
	}
}

// generate code to start new proc running call n.
func cgen_proc(n *Node, proc int) {
	switch n.Left.Op {
	default:
		Fatalf("cgen_proc: unknown call %v", Oconv(int(n.Left.Op), 0))

	case OCALLMETH:
		cgen_callmeth(n.Left, proc)

	case OCALLINTER:
		cgen_callinter(n.Left, nil, proc)

	case OCALLFUNC:
		cgen_call(n.Left, proc)
	}
}

// generate declaration.
// have to allocate heap copy
// for escaped variables.
func cgen_dcl(n *Node) {
	if Debug['g'] != 0 {
		Dump("\ncgen-dcl", n)
	}
	if n.Op != ONAME {
		Dump("cgen_dcl", n)
		Fatalf("cgen_dcl")
	}

	if n.Class&PHEAP == 0 {
		return
	}
	if compiling_runtime != 0 {
		Yyerror("%v escapes to heap, not allowed in runtime.", n)
	}
	if prealloc[n] == nil {
		prealloc[n] = callnew(n.Type)
	}
	Cgen_as(n.Name.Heapaddr, prealloc[n])
}

// generate discard of value
func cgen_discard(nr *Node) {
	if nr == nil {
		return
	}

	switch nr.Op {
	case ONAME:
		if nr.Class&PHEAP == 0 && nr.Class != PEXTERN && nr.Class != PFUNC && nr.Class != PPARAMREF {
			gused(nr)
		}

		// unary
	case OADD,
		OAND,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLSH,
		OLT,
		OMOD,
		OMUL,
		ONE,
		OOR,
		ORSH,
		OSUB,
		OXOR:
		cgen_discard(nr.Left)

		cgen_discard(nr.Right)

		// binary
	case OCAP,
		OCOM,
		OLEN,
		OMINUS,
		ONOT,
		OPLUS:
		cgen_discard(nr.Left)

	case OIND:
		Cgen_checknil(nr.Left)

		// special enough to just evaluate
	default:
		var tmp Node
		Tempname(&tmp, nr.Type)

		Cgen_as(&tmp, nr)
		gused(&tmp)
	}
}

// clearslim generates code to zero a slim node.
func Clearslim(n *Node) {
	var z Node
	z.Op = OLITERAL
	z.Type = n.Type
	z.Addable = true

	switch Simtype[n.Type.Etype] {
	case TCOMPLEX64, TCOMPLEX128:
		z.SetVal(Val{new(Mpcplx)})
		Mpmovecflt(&z.Val().U.(*Mpcplx).Real, 0.0)
		Mpmovecflt(&z.Val().U.(*Mpcplx).Imag, 0.0)

	case TFLOAT32, TFLOAT64:
		var zero Mpflt
		Mpmovecflt(&zero, 0.0)
		z.SetVal(Val{&zero})

	case TPTR32, TPTR64, TCHAN, TMAP:
		z.SetVal(Val{new(NilVal)})

	case TBOOL:
		z.SetVal(Val{false})

	case TINT8,
		TINT16,
		TINT32,
		TINT64,
		TUINT8,
		TUINT16,
		TUINT32,
		TUINT64:
		z.SetVal(Val{new(Mpint)})
		Mpmovecfix(z.Val().U.(*Mpint), 0)

	default:
		Fatalf("clearslim called on type %v", n.Type)
	}

	ullmancalc(&z)
	Cgen(&z, n)
}

// generate:
//	res = iface{typ, data}
// n->left is typ
// n->right is data
func Cgen_eface(n *Node, res *Node) {
	// the right node of an eface may contain function calls that uses res as an argument,
	// so it's important that it is done first

	tmp := temp(Types[Tptr])
	Cgen(n.Right, tmp)

	Gvardef(res)

	dst := *res
	dst.Type = Types[Tptr]
	dst.Xoffset += int64(Widthptr)
	Cgen(tmp, &dst)

	dst.Xoffset -= int64(Widthptr)
	Cgen(n.Left, &dst)
}

// generate one of:
//	res, resok = x.(T)
//	res = x.(T) (when resok == nil)
// n.Left is x
// n.Type is T
func cgen_dottype(n *Node, res, resok *Node, wb bool) {
	if Debug_typeassert > 0 {
		Warn("type assertion inlined")
	}
	//	iface := n.Left
	//	r1 := iword(iface)
	//	if n.Left is non-empty interface {
	//		r1 = *r1
	//	}
	//	if r1 == T {
	//		res = idata(iface)
	//		resok = true
	//	} else {
	//		assert[EI]2T(x, T, nil) // (when resok == nil; does not return)
	//		resok = false // (when resok != nil)
	//	}
	//
	var iface Node
	Igen(n.Left, &iface, res)
	var r1, r2 Node
	byteptr := Ptrto(Types[TUINT8]) // type used in runtime prototypes for runtime type (*byte)
	Regalloc(&r1, byteptr, nil)
	iface.Type = byteptr
	Cgen(&iface, &r1)
	if !isnilinter(n.Left.Type) {
		// Holding itab, want concrete type in second word.
		p := Thearch.Ginscmp(OEQ, byteptr, &r1, Nodintconst(0), -1)
		r2 = r1
		r2.Op = OINDREG
		r2.Xoffset = int64(Widthptr)
		Cgen(&r2, &r1)
		Patch(p, Pc)
	}
	Regalloc(&r2, byteptr, nil)
	Cgen(typename(n.Type), &r2)
	p := Thearch.Ginscmp(ONE, byteptr, &r1, &r2, -1)
	Regfree(&r2) // not needed for success path; reclaimed on one failure path
	iface.Xoffset += int64(Widthptr)
	Cgen(&iface, &r1)
	Regfree(&iface)

	if resok == nil {
		r1.Type = res.Type
		cgen_wb(&r1, res, wb)
		q := Gbranch(obj.AJMP, nil, 0)
		Patch(p, Pc)
		Regrealloc(&r2) // reclaim from above, for this failure path
		fn := syslook("panicdottype", 0)
		dowidth(fn.Type)
		call := Nod(OCALLFUNC, fn, nil)
		r1.Type = byteptr
		r2.Type = byteptr
		call.List = list(list(list1(&r1), &r2), typename(n.Left.Type))
		call.List = ascompatte(OCALLFUNC, call, false, getinarg(fn.Type), call.List, 0, nil)
		gen(call)
		Regfree(&r1)
		Regfree(&r2)
		Thearch.Gins(obj.AUNDEF, nil, nil)
		Patch(q, Pc)
	} else {
		// This half is handling the res, resok = x.(T) case,
		// which is called from gen, not cgen, and is consequently fussier
		// about blank assignments. We have to avoid calling cgen for those.
		r1.Type = res.Type
		if !isblank(res) {
			cgen_wb(&r1, res, wb)
		}
		Regfree(&r1)
		if !isblank(resok) {
			Cgen(Nodbool(true), resok)
		}
		q := Gbranch(obj.AJMP, nil, 0)
		Patch(p, Pc)
		if !isblank(res) {
			n := nodnil()
			n.Type = res.Type
			Cgen(n, res)
		}
		if !isblank(resok) {
			Cgen(Nodbool(false), resok)
		}
		Patch(q, Pc)
	}
}

// generate:
//	res, resok = x.(T)
// n.Left is x
// n.Type is T
func Cgen_As2dottype(n, res, resok *Node) {
	if Debug_typeassert > 0 {
		Warn("type assertion inlined")
	}
	//	iface := n.Left
	//	r1 := iword(iface)
	//	if n.Left is non-empty interface {
	//		r1 = *r1
	//	}
	//	if r1 == T {
	//		res = idata(iface)
	//		resok = true
	//	} else {
	//		res = nil
	//		resok = false
	//	}
	//
	var iface Node
	Igen(n.Left, &iface, nil)
	var r1, r2 Node
	byteptr := Ptrto(Types[TUINT8]) // type used in runtime prototypes for runtime type (*byte)
	Regalloc(&r1, byteptr, res)
	iface.Type = byteptr
	Cgen(&iface, &r1)
	if !isnilinter(n.Left.Type) {
		// Holding itab, want concrete type in second word.
		p := Thearch.Ginscmp(OEQ, byteptr, &r1, Nodintconst(0), -1)
		r2 = r1
		r2.Op = OINDREG
		r2.Xoffset = int64(Widthptr)
		Cgen(&r2, &r1)
		Patch(p, Pc)
	}
	Regalloc(&r2, byteptr, nil)
	Cgen(typename(n.Type), &r2)
	p := Thearch.Ginscmp(ONE, byteptr, &r1, &r2, -1)
	iface.Type = n.Type
	iface.Xoffset += int64(Widthptr)
	Cgen(&iface, &r1)
	if iface.Op != 0 {
		Regfree(&iface)
	}
	Cgen(&r1, res)
	q := Gbranch(obj.AJMP, nil, 0)
	Patch(p, Pc)

	fn := syslook("panicdottype", 0)
	dowidth(fn.Type)
	call := Nod(OCALLFUNC, fn, nil)
	call.List = list(list(list1(&r1), &r2), typename(n.Left.Type))
	call.List = ascompatte(OCALLFUNC, call, false, getinarg(fn.Type), call.List, 0, nil)
	gen(call)
	Regfree(&r1)
	Regfree(&r2)
	Thearch.Gins(obj.AUNDEF, nil, nil)
	Patch(q, Pc)
}

// gather series of offsets
// >=0 is direct addressed field
// <0 is pointer to next field (+1)
func Dotoffset(n *Node, oary []int64, nn **Node) int {
	var i int

	switch n.Op {
	case ODOT:
		if n.Xoffset == BADWIDTH {
			Dump("bad width in dotoffset", n)
			Fatalf("bad width in dotoffset")
		}

		i = Dotoffset(n.Left, oary, nn)
		if i > 0 {
			if oary[i-1] >= 0 {
				oary[i-1] += n.Xoffset
			} else {
				oary[i-1] -= n.Xoffset
			}
			break
		}

		if i < 10 {
			oary[i] = n.Xoffset
			i++
		}

	case ODOTPTR:
		if n.Xoffset == BADWIDTH {
			Dump("bad width in dotoffset", n)
			Fatalf("bad width in dotoffset")
		}

		i = Dotoffset(n.Left, oary, nn)
		if i < 10 {
			oary[i] = -(n.Xoffset + 1)
			i++
		}

	default:
		*nn = n
		return 0
	}

	if i >= 10 {
		*nn = nil
	}
	return i
}

// make a new off the books
func Tempname(nn *Node, t *Type) {
	if Curfn == nil {
		Fatalf("no curfn for tempname")
	}

	if t == nil {
		Yyerror("tempname called with nil type")
		t = Types[TINT32]
	}

	// give each tmp a different name so that there
	// a chance to registerizer them
	s := Lookupf("autotmp_%.4d", statuniqgen)
	statuniqgen++
	n := Nod(ONAME, nil, nil)
	n.Sym = s
	s.Def = n
	n.Type = t
	n.Class = PAUTO
	n.Addable = true
	n.Ullman = 1
	n.Esc = EscNever
	n.Name.Curfn = Curfn
	Curfn.Func.Dcl = list(Curfn.Func.Dcl, n)
	if Debug['h'] != 0 {
		println("H", n, n.Orig, funcSym(Curfn).Name)
	}

	dowidth(t)
	n.Xoffset = 0
	*nn = *n
}

func temp(t *Type) *Node {
	n := Nod(OXXX, nil, nil)
	Tempname(n, t)
	n.Sym.Def.Used = true
	return n.Orig
}

func gen(n *Node) {
	//dump("gen", n);

	lno := setlineno(n)

	wasregalloc := Anyregalloc()

	if n == nil {
		goto ret
	}

	if n.Ninit != nil {
		Genlist(n.Ninit)
	}

	setlineno(n)

	switch n.Op {
	default:
		Fatalf("gen: unknown op %v", Nconv(n, obj.FmtShort|obj.FmtSign))

	case OCASE,
		OFALL,
		OXCASE,
		OXFALL,
		ODCLCONST,
		ODCLFUNC,
		ODCLTYPE:
		break

	case OEMPTY:
		break

	case OBLOCK:
		Genlist(n.List)

	case OLABEL:
		if isblanksym(n.Left.Sym) {
			break
		}

		lab := newlab(n)

		// if there are pending gotos, resolve them all to the current pc.
		var p2 *obj.Prog
		for p1 := lab.Gotopc; p1 != nil; p1 = p2 {
			p2 = unpatch(p1)
			Patch(p1, Pc)
		}

		lab.Gotopc = nil
		if lab.Labelpc == nil {
			lab.Labelpc = Pc
		}

		if n.Name.Defn != nil {
			switch n.Name.Defn.Op {
			// so stmtlabel can find the label
			case OFOR, OSWITCH, OSELECT:
				n.Name.Defn.Sym = lab.Sym
			}
		}

		// if label is defined, emit jump to it.
	// otherwise save list of pending gotos in lab->gotopc.
	// the list is linked through the normal jump target field
	// to avoid a second list.  (the jumps are actually still
	// valid code, since they're just going to another goto
	// to the same label.  we'll unwind it when we learn the pc
	// of the label in the OLABEL case above.)
	case OGOTO:
		lab := newlab(n)

		if lab.Labelpc != nil {
			gjmp(lab.Labelpc)
		} else {
			lab.Gotopc = gjmp(lab.Gotopc)
		}

	case OBREAK:
		if n.Left != nil {
			lab := n.Left.Sym.Label
			if lab == nil {
				Yyerror("break label not defined: %v", n.Left.Sym)
				break
			}

			lab.Used = true
			if lab.Breakpc == nil {
				Yyerror("invalid break label %v", n.Left.Sym)
				break
			}

			gjmp(lab.Breakpc)
			break
		}

		if breakpc == nil {
			Yyerror("break is not in a loop")
			break
		}

		gjmp(breakpc)

	case OCONTINUE:
		if n.Left != nil {
			lab := n.Left.Sym.Label
			if lab == nil {
				Yyerror("continue label not defined: %v", n.Left.Sym)
				break
			}

			lab.Used = true
			if lab.Continpc == nil {
				Yyerror("invalid continue label %v", n.Left.Sym)
				break
			}

			gjmp(lab.Continpc)
			break
		}

		if continpc == nil {
			Yyerror("continue is not in a loop")
			break
		}

		gjmp(continpc)

	case OFOR:
		sbreak := breakpc
		p1 := gjmp(nil)     //		goto test
		breakpc = gjmp(nil) // break:	goto done
		scontin := continpc
		continpc = Pc

		// define break and continue labels
		lab := stmtlabel(n)
		if lab != nil {
			lab.Breakpc = breakpc
			lab.Continpc = continpc
		}

		gen(n.Right)                     // contin:	incr
		Patch(p1, Pc)                    // test:
		Bgen(n.Left, false, -1, breakpc) //		if(!test) goto break
		Genlist(n.Nbody)                 //		body
		gjmp(continpc)
		Patch(breakpc, Pc) // done:
		continpc = scontin
		breakpc = sbreak
		if lab != nil {
			lab.Breakpc = nil
			lab.Continpc = nil
		}

	case OIF:
		p1 := gjmp(nil)                         //		goto test
		p2 := gjmp(nil)                         // p2:		goto else
		Patch(p1, Pc)                           // test:
		Bgen(n.Left, false, int(-n.Likely), p2) //		if(!test) goto p2
		Genlist(n.Nbody)                        //		then
		p3 := gjmp(nil)                         //		goto done
		Patch(p2, Pc)                           // else:
		Genlist(n.Rlist)                        //		else
		Patch(p3, Pc)                           // done:

	case OSWITCH:
		sbreak := breakpc
		p1 := gjmp(nil)     //		goto test
		breakpc = gjmp(nil) // break:	goto done

		// define break label
		lab := stmtlabel(n)
		if lab != nil {
			lab.Breakpc = breakpc
		}

		Patch(p1, Pc)      // test:
		Genlist(n.Nbody)   //		switch(test) body
		Patch(breakpc, Pc) // done:
		breakpc = sbreak
		if lab != nil {
			lab.Breakpc = nil
		}

	case OSELECT:
		sbreak := breakpc
		p1 := gjmp(nil)     //		goto test
		breakpc = gjmp(nil) // break:	goto done

		// define break label
		lab := stmtlabel(n)
		if lab != nil {
			lab.Breakpc = breakpc
		}

		Patch(p1, Pc)      // test:
		Genlist(n.Nbody)   //		select() body
		Patch(breakpc, Pc) // done:
		breakpc = sbreak
		if lab != nil {
			lab.Breakpc = nil
		}

	case ODCL:
		cgen_dcl(n.Left)

	case OAS:
		if gen_as_init(n) {
			break
		}
		Cgen_as(n.Left, n.Right)

	case OASWB:
		Cgen_as_wb(n.Left, n.Right, true)

	case OAS2DOTTYPE:
		cgen_dottype(n.Rlist.N, n.List.N, n.List.Next.N, false)

	case OCALLMETH:
		cgen_callmeth(n, 0)

	case OCALLINTER:
		cgen_callinter(n, nil, 0)

	case OCALLFUNC:
		cgen_call(n, 0)

	case OPROC:
		cgen_proc(n, 1)

	case ODEFER:
		cgen_proc(n, 2)

	case ORETURN, ORETJMP:
		cgen_ret(n)

	// Function calls turned into compiler intrinsics.
	// At top level, can just ignore the call and make sure to preserve side effects in the argument, if any.
	case OGETG:
		// nothing
	case OSQRT:
		cgen_discard(n.Left)

	case OCHECKNIL:
		Cgen_checknil(n.Left)

	case OVARKILL:
		gvarkill(n.Left)

	case OVARLIVE:
		gvarlive(n.Left)
	}

ret:
	if Anyregalloc() != wasregalloc {
		Dump("node", n)
		Fatalf("registers left allocated")
	}

	lineno = lno
}

func Cgen_as(nl, nr *Node) {
	Cgen_as_wb(nl, nr, false)
}

func Cgen_as_wb(nl, nr *Node, wb bool) {
	if Debug['g'] != 0 {
		op := "cgen_as"
		if wb {
			op = "cgen_as_wb"
		}
		Dump(op, nl)
		Dump(op+" = ", nr)
	}

	for nr != nil && nr.Op == OCONVNOP {
		nr = nr.Left
	}

	if nl == nil || isblank(nl) {
		cgen_discard(nr)
		return
	}

	if nr == nil || iszero(nr) {
		// heaps should already be clear
		if nr == nil && (nl.Class&PHEAP != 0) {
			return
		}

		tl := nl.Type
		if tl == nil {
			return
		}
		if Isfat(tl) {
			if nl.Op == ONAME {
				Gvardef(nl)
			}
			Thearch.Clearfat(nl)
			return
		}

		Clearslim(nl)
		return
	}

	tl := nl.Type
	if tl == nil {
		return
	}

	cgen_wb(nr, nl, wb)
}

func cgen_callmeth(n *Node, proc int) {
	// generate a rewrite in n2 for the method call
	// (p.f)(...) goes to (f)(p,...)

	l := n.Left

	if l.Op != ODOTMETH {
		Fatalf("cgen_callmeth: not dotmethod: %v", l)
	}

	n2 := *n
	n2.Op = OCALLFUNC
	n2.Left = l.Right
	n2.Left.Type = l.Type

	if n2.Left.Op == ONAME {
		n2.Left.Class = PFUNC
	}
	cgen_call(&n2, proc)
}

// CgenTemp creates a temporary node, assigns n to it, and returns it.
func CgenTemp(n *Node) *Node {
	var tmp Node
	Tempname(&tmp, n.Type)
	Cgen(n, &tmp)
	return &tmp
}

func checklabels() {
	for lab := labellist; lab != nil; lab = lab.Link {
		if lab.Def == nil {
			for _, n := range lab.Use {
				yyerrorl(int(n.Lineno), "label %v not defined", lab.Sym)
			}
			continue
		}

		if lab.Use == nil && !lab.Used {
			yyerrorl(int(lab.Def.Lineno), "label %v defined and not used", lab.Sym)
			continue
		}

		if lab.Gotopc != nil {
			Fatalf("label %v never resolved", lab.Sym)
		}
		for _, n := range lab.Use {
			checkgoto(n, lab.Def)
		}
	}
}

// Componentgen copies a composite value by moving its individual components.
// Slices, strings and interfaces are supported. Small structs or arrays with
// elements of basic type are also supported.
// nr is nil when assigning a zero value.
func Componentgen(nr, nl *Node) bool {
	return componentgen_wb(nr, nl, false)
}

// componentgen_wb is like componentgen but if wb==true emits write barriers for pointer updates.
func componentgen_wb(nr, nl *Node, wb bool) bool {
	// Don't generate any code for complete copy of a variable into itself.
	// It's useless, and the VARDEF will incorrectly mark the old value as dead.
	// (This check assumes that the arguments passed to componentgen did not
	// themselves come from Igen, or else we could have Op==ONAME but
	// with a Type and Xoffset describing an individual field, not the entire
	// variable.)
	if nl.Op == ONAME && nl == nr {
		return true
	}

	// Count number of moves required to move components.
	// If using write barrier, can only emit one pointer.
	// TODO(rsc): Allow more pointers, for reflect.Value.
	const maxMoves = 8
	n := 0
	numPtr := 0
	visitComponents(nl.Type, 0, func(t *Type, offset int64) bool {
		n++
		if Simtype[t.Etype] == Tptr && t != itable {
			numPtr++
		}
		return n <= maxMoves && (!wb || numPtr <= 1)
	})
	if n > maxMoves || wb && numPtr > 1 {
		return false
	}

	// Must call emitVardef after evaluating rhs but before writing to lhs.
	emitVardef := func() {
		// Emit vardef if needed.
		if nl.Op == ONAME {
			switch nl.Type.Etype {
			case TARRAY, TSTRING, TINTER, TSTRUCT:
				Gvardef(nl)
			}
		}
	}

	isConstString := Isconst(nr, CTSTR)

	if !cadable(nl) && nr != nil && !cadable(nr) && !isConstString {
		return false
	}

	var nodl Node
	if cadable(nl) {
		nodl = *nl
	} else {
		if nr != nil && !cadable(nr) && !isConstString {
			return false
		}
		if nr == nil || isConstString || nl.Ullman >= nr.Ullman {
			Igen(nl, &nodl, nil)
			defer Regfree(&nodl)
		}
	}
	lbase := nodl.Xoffset

	// Special case: zeroing.
	var nodr Node
	if nr == nil {
		// When zeroing, prepare a register containing zero.
		// TODO(rsc): Check that this is actually generating the best code.
		if Thearch.REGZERO != 0 {
			// cpu has a dedicated zero register
			Nodreg(&nodr, Types[TUINT], Thearch.REGZERO)
		} else {
			// no dedicated zero register
			var zero Node
			Nodconst(&zero, nl.Type, 0)
			Regalloc(&nodr, Types[TUINT], nil)
			Thearch.Gmove(&zero, &nodr)
			defer Regfree(&nodr)
		}

		emitVardef()
		visitComponents(nl.Type, 0, func(t *Type, offset int64) bool {
			nodl.Type = t
			nodl.Xoffset = lbase + offset
			nodr.Type = t
			if Isfloat[t.Etype] {
				// TODO(rsc): Cache zero register like we do for integers?
				Clearslim(&nodl)
			} else {
				Thearch.Gmove(&nodr, &nodl)
			}
			return true
		})
		return true
	}

	// Special case: assignment of string constant.
	if isConstString {
		emitVardef()

		// base
		nodl.Type = Ptrto(Types[TUINT8])
		Regalloc(&nodr, Types[Tptr], nil)
		p := Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), nil, &nodr)
		Datastring(nr.Val().U.(string), &p.From)
		p.From.Type = obj.TYPE_ADDR
		Thearch.Gmove(&nodr, &nodl)
		Regfree(&nodr)

		// length
		nodl.Type = Types[Simtype[TUINT]]
		nodl.Xoffset += int64(Array_nel) - int64(Array_array)
		Nodconst(&nodr, nodl.Type, int64(len(nr.Val().U.(string))))
		Thearch.Gmove(&nodr, &nodl)
		return true
	}

	// General case: copy nl = nr.
	nodr = *nr
	if !cadable(nr) {
		if nr.Ullman >= UINF && nodl.Op == OINDREG {
			Fatalf("miscompile")
		}
		Igen(nr, &nodr, nil)
		defer Regfree(&nodr)
	}
	rbase := nodr.Xoffset

	if nodl.Op == 0 {
		Igen(nl, &nodl, nil)
		defer Regfree(&nodl)
		lbase = nodl.Xoffset
	}

	emitVardef()
	var (
		ptrType   *Type
		ptrOffset int64
	)
	visitComponents(nl.Type, 0, func(t *Type, offset int64) bool {
		if wb && Simtype[t.Etype] == Tptr && t != itable {
			if ptrType != nil {
				Fatalf("componentgen_wb %v", Tconv(nl.Type, 0))
			}
			ptrType = t
			ptrOffset = offset
			return true
		}
		nodl.Type = t
		nodl.Xoffset = lbase + offset
		nodr.Type = t
		nodr.Xoffset = rbase + offset
		Thearch.Gmove(&nodr, &nodl)
		return true
	})
	if ptrType != nil {
		nodl.Type = ptrType
		nodl.Xoffset = lbase + ptrOffset
		nodr.Type = ptrType
		nodr.Xoffset = rbase + ptrOffset
		cgen_wbptr(&nodr, &nodl)
	}
	return true
}

// visitComponents walks the individual components of the type t,
// walking into array elements, struct fields, the real and imaginary
// parts of complex numbers, and on 32-bit systems the high and
// low halves of 64-bit integers.
// It calls f for each such component, passing the component (aka element)
// type and memory offset, assuming t starts at startOffset.
// If f ever returns false, visitComponents returns false without any more
// calls to f. Otherwise visitComponents returns true.
func visitComponents(t *Type, startOffset int64, f func(elem *Type, elemOffset int64) bool) bool {
	switch t.Etype {
	case TINT64:
		if Widthreg == 8 {
			break
		}
		// NOTE: Assuming little endian (signed top half at offset 4).
		// We don't have any 32-bit big-endian systems.
		if Thearch.Thechar != '5' && Thearch.Thechar != '8' {
			Fatalf("unknown 32-bit architecture")
		}
		return f(Types[TUINT32], startOffset) &&
			f(Types[TINT32], startOffset+4)

	case TUINT64:
		if Widthreg == 8 {
			break
		}
		return f(Types[TUINT32], startOffset) &&
			f(Types[TUINT32], startOffset+4)

	case TCOMPLEX64:
		return f(Types[TFLOAT32], startOffset) &&
			f(Types[TFLOAT32], startOffset+4)

	case TCOMPLEX128:
		return f(Types[TFLOAT64], startOffset) &&
			f(Types[TFLOAT64], startOffset+8)

	case TINTER:
		return f(itable, startOffset) &&
			f(Ptrto(Types[TUINT8]), startOffset+int64(Widthptr))

	case TSTRING:
		return f(Ptrto(Types[TUINT8]), startOffset) &&
			f(Types[Simtype[TUINT]], startOffset+int64(Widthptr))

	case TARRAY:
		if Isslice(t) {
			return f(Ptrto(t.Type), startOffset+int64(Array_array)) &&
				f(Types[Simtype[TUINT]], startOffset+int64(Array_nel)) &&
				f(Types[Simtype[TUINT]], startOffset+int64(Array_cap))
		}

		// Short-circuit [1e6]struct{}.
		if t.Type.Width == 0 {
			return true
		}

		for i := int64(0); i < t.Bound; i++ {
			if !visitComponents(t.Type, startOffset+i*t.Type.Width, f) {
				return false
			}
		}
		return true

	case TSTRUCT:
		if t.Type != nil && t.Type.Width != 0 {
			// NOTE(rsc): If this happens, the right thing to do is to say
			//	startOffset -= t.Type.Width
			// but I want to see if it does.
			// The old version of componentgen handled this,
			// in code introduced in CL 6932045 to fix issue #4518.
			// But the test case in issue 4518 does not trigger this anymore,
			// so maybe this complication is no longer needed.
			Fatalf("struct not at offset 0")
		}

		for field := t.Type; field != nil; field = field.Down {
			if field.Etype != TFIELD {
				Fatalf("bad struct")
			}
			if !visitComponents(field.Type, startOffset+field.Width, f) {
				return false
			}
		}
		return true
	}
	return f(t, startOffset)
}

func cadable(n *Node) bool {
	// Note: Not sure why you can have n.Op == ONAME without n.Addable, but you can.
	return n.Addable && n.Op == ONAME
}
