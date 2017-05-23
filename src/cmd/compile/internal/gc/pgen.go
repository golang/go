// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/sys"
	"fmt"
	"sort"
	"strings"
)

// "Portable" code generation.

var makefuncdatasym_nsym int

func makefuncdatasym(nameprefix string, funcdatakind int64) *Sym {
	var nod Node

	sym := LookupN(nameprefix, makefuncdatasym_nsym)
	makefuncdatasym_nsym++
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

func gvardefx(n *Node, as obj.As) {
	if n == nil {
		Fatalf("gvardef nil")
	}
	if n.Op != ONAME {
		Yyerror("gvardef %#v; %v", n.Op, n)
		return
	}

	switch n.Class {
	case PAUTO, PPARAM, PPARAMOUT:
		if as == obj.AVARLIVE {
			Thearch.Gins(as, n, nil)
		} else {
			Thearch.Gins(as, nil, n)
		}
	}
}

func Gvardef(n *Node) {
	gvardefx(n, obj.AVARDEF)
}

func Gvarkill(n *Node) {
	gvardefx(n, obj.AVARKILL)
}

func Gvarlive(n *Node) {
	gvardefx(n, obj.AVARLIVE)
}

func removevardef(firstp *obj.Prog) {
	for p := firstp; p != nil; p = p.Link {
		for p.Link != nil && (p.Link.As == obj.AVARDEF || p.Link.As == obj.AVARKILL || p.Link.As == obj.AVARLIVE) {
			p.Link = p.Link.Link
		}
		if p.To.Type == obj.TYPE_BRANCH {
			for p.To.Val.(*obj.Prog) != nil && (p.To.Val.(*obj.Prog).As == obj.AVARDEF || p.To.Val.(*obj.Prog).As == obj.AVARKILL || p.To.Val.(*obj.Prog).As == obj.AVARLIVE) {
				p.To.Val = p.To.Val.(*obj.Prog).Link
			}
		}
	}
}

func emitptrargsmap() {
	if Curfn.Func.Nname.Sym.Name == "_" {
		return
	}
	sym := Lookup(fmt.Sprintf("%s.args_stackmap", Curfn.Func.Nname.Sym.Name))

	nptr := int(Curfn.Type.ArgWidth() / int64(Widthptr))
	bv := bvalloc(int32(nptr) * 2)
	nbitmap := 1
	if Curfn.Type.Results().NumFields() > 0 {
		nbitmap = 2
	}
	off := duint32(sym, 0, uint32(nbitmap))
	off = duint32(sym, off, uint32(bv.n))
	var xoffset int64
	if Curfn.Type.Recv() != nil {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Recvs(), &xoffset, bv)
	}

	if Curfn.Type.Params().NumFields() > 0 {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Params(), &xoffset, bv)
	}

	for j := 0; int32(j) < bv.n; j += 32 {
		off = duint32(sym, off, bv.b[j/32])
	}
	if Curfn.Type.Results().NumFields() > 0 {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Results(), &xoffset, bv)
		for j := 0; int32(j) < bv.n; j += 32 {
			off = duint32(sym, off, bv.b[j/32])
		}
	}

	ggloblsym(sym, int32(off), obj.RODATA|obj.LOCAL)
}

// cmpstackvarlt reports whether the stack variable a sorts before b.
//
// Sort the list of stack variables. Autos after anything else,
// within autos, unused after used, within used, things with
// pointers first, zeroed things first, and then decreasing size.
// Because autos are laid out in decreasing addresses
// on the stack, pointers first, zeroed things first and decreasing size
// really means, in memory, things with pointers needing zeroing at
// the top of the stack and increasing in size.
// Non-autos sort on offset.
func cmpstackvarlt(a, b *Node) bool {
	if (a.Class == PAUTO) != (b.Class == PAUTO) {
		return b.Class == PAUTO
	}

	if a.Class != PAUTO {
		return a.Xoffset < b.Xoffset
	}

	if a.Used != b.Used {
		return a.Used
	}

	ap := haspointers(a.Type)
	bp := haspointers(b.Type)
	if ap != bp {
		return ap
	}

	ap = a.Name.Needzero
	bp = b.Name.Needzero
	if ap != bp {
		return ap
	}

	if a.Type.Width != b.Type.Width {
		return a.Type.Width > b.Type.Width
	}

	return a.Sym.Name < b.Sym.Name
}

// byStackvar implements sort.Interface for []*Node using cmpstackvarlt.
type byStackVar []*Node

func (s byStackVar) Len() int           { return len(s) }
func (s byStackVar) Less(i, j int) bool { return cmpstackvarlt(s[i], s[j]) }
func (s byStackVar) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// stkdelta records the stack offset delta for a node
// during the compaction of the stack frame to remove
// unused stack slots.
var stkdelta = map[*Node]int64{}

// TODO(lvd) find out where the PAUTO/OLITERAL nodes come from.
func allocauto(ptxt *obj.Prog) {
	Stksize = 0
	stkptrsize = 0

	if len(Curfn.Func.Dcl) == 0 {
		return
	}

	// Mark the PAUTO's unused.
	for _, ln := range Curfn.Func.Dcl {
		if ln.Class == PAUTO {
			ln.Used = false
		}
	}

	markautoused(ptxt)

	sort.Sort(byStackVar(Curfn.Func.Dcl))

	// Unused autos are at the end, chop 'em off.
	n := Curfn.Func.Dcl[0]
	if n.Class == PAUTO && n.Op == ONAME && !n.Used {
		// No locals used at all
		Curfn.Func.Dcl = nil

		fixautoused(ptxt)
		return
	}

	for i := 1; i < len(Curfn.Func.Dcl); i++ {
		n = Curfn.Func.Dcl[i]
		if n.Class == PAUTO && n.Op == ONAME && !n.Used {
			Curfn.Func.Dcl = Curfn.Func.Dcl[:i]
			break
		}
	}

	// Reassign stack offsets of the locals that are still there.
	var w int64
	for _, n := range Curfn.Func.Dcl {
		if n.Class != PAUTO || n.Op != ONAME {
			continue
		}

		dowidth(n.Type)
		w = n.Type.Width
		if w >= Thearch.MAXWIDTH || w < 0 {
			Fatalf("bad width")
		}
		Stksize += w
		Stksize = Rnd(Stksize, int64(n.Type.Align))
		if haspointers(n.Type) {
			stkptrsize = Stksize
		}
		if Thearch.LinkArch.InFamily(sys.MIPS64, sys.ARM, sys.ARM64, sys.PPC64, sys.S390X) {
			Stksize = Rnd(Stksize, int64(Widthptr))
		}
		if Stksize >= 1<<31 {
			setlineno(Curfn)
			Yyerror("stack frame too large (>2GB)")
		}

		stkdelta[n] = -Stksize - n.Xoffset
	}

	Stksize = Rnd(Stksize, int64(Widthreg))
	stkptrsize = Rnd(stkptrsize, int64(Widthreg))

	fixautoused(ptxt)

	// The debug information needs accurate offsets on the symbols.
	for _, ln := range Curfn.Func.Dcl {
		if ln.Class != PAUTO || ln.Op != ONAME {
			continue
		}
		ln.Xoffset += stkdelta[ln]
		delete(stkdelta, ln)
	}
}

func Cgen_checknil(n *Node) {
	if Disable_checknil != 0 {
		return
	}

	// Ideally we wouldn't see any integer types here, but we do.
	if n.Type == nil || (!n.Type.IsPtr() && !n.Type.IsInteger() && n.Type.Etype != TUNSAFEPTR) {
		Dump("checknil", n)
		Fatalf("bad checknil")
	}

	// Most architectures require that the address to be checked is
	// in a register (it could be in memory).
	needsReg := !Thearch.LinkArch.InFamily(sys.AMD64, sys.I386)

	// Move the address to be checked into a register if necessary.
	if (needsReg && n.Op != OREGISTER) || !n.Addable || n.Op == OLITERAL {
		var reg Node
		Regalloc(&reg, Types[Tptr], n)
		Cgen(n, &reg)
		Thearch.Gins(obj.ACHECKNIL, &reg, nil)
		Regfree(&reg)
		return
	}

	Thearch.Gins(obj.ACHECKNIL, n, nil)
}

func compile(fn *Node) {
	if Newproc == nil {
		Newproc = Sysfunc("newproc")
		Deferproc = Sysfunc("deferproc")
		Deferreturn = Sysfunc("deferreturn")
		Panicindex = Sysfunc("panicindex")
		panicslice = Sysfunc("panicslice")
		panicdivide = Sysfunc("panicdivide")
		throwreturn = Sysfunc("throwreturn")
		growslice = Sysfunc("growslice")
		writebarrierptr = Sysfunc("writebarrierptr")
		typedmemmove = Sysfunc("typedmemmove")
		panicdottype = Sysfunc("panicdottype")
	}

	defer func(lno int32) {
		lineno = lno
	}(setlineno(fn))

	Curfn = fn
	dowidth(Curfn.Type)

	if fn.Nbody.Len() == 0 {
		if pure_go || strings.HasPrefix(fn.Func.Nname.Sym.Name, "init.") {
			Yyerror("missing function body for %q", fn.Func.Nname.Sym.Name)
			return
		}

		if Debug['A'] != 0 {
			return
		}
		emitptrargsmap()
		return
	}

	saveerrors()

	// set up domain for labels
	clearlabels()

	if Curfn.Type.FuncType().Outnamed {
		// add clearing of the output parameters
		for _, t := range Curfn.Type.Results().Fields().Slice() {
			if t.Nname != nil {
				n := Nod(OAS, t.Nname, nil)
				n = typecheck(n, Etop)
				Curfn.Nbody.Set(append([]*Node{n}, Curfn.Nbody.Slice()...))
			}
		}
	}

	order(Curfn)
	if nerrors != 0 {
		return
	}

	hasdefer = false
	walk(Curfn)
	if nerrors != 0 {
		return
	}
	if instrumenting {
		instrument(Curfn)
	}
	if nerrors != 0 {
		return
	}

	// Build an SSA backend function.
	var ssafn *ssa.Func
	if shouldssa(Curfn) {
		ssafn = buildssa(Curfn)
	}

	continpc = nil
	breakpc = nil

	pl := newplist()
	pl.Name = Linksym(Curfn.Func.Nname.Sym)

	setlineno(Curfn)

	var nod1 Node
	Nodconst(&nod1, Types[TINT32], 0)
	nam := Curfn.Func.Nname
	if isblank(nam) {
		nam = nil
	}
	ptxt := Thearch.Gins(obj.ATEXT, nam, &nod1)
	Afunclit(&ptxt.From, Curfn.Func.Nname)
	ptxt.From3 = new(obj.Addr)
	if fn.Func.Dupok {
		ptxt.From3.Offset |= obj.DUPOK
	}
	if fn.Func.Wrapper {
		ptxt.From3.Offset |= obj.WRAPPER
	}
	if fn.Func.Needctxt {
		ptxt.From3.Offset |= obj.NEEDCTXT
	}
	if fn.Func.Pragma&Nosplit != 0 {
		ptxt.From3.Offset |= obj.NOSPLIT
	}
	if fn.Func.ReflectMethod {
		ptxt.From3.Offset |= obj.REFLECTMETHOD
	}
	if fn.Func.Pragma&Systemstack != 0 {
		ptxt.From.Sym.Cfunc = true
	}

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if myimportpath == "reflect" {
		if Curfn.Func.Nname.Sym.Name == "callReflect" || Curfn.Func.Nname.Sym.Name == "callMethod" {
			ptxt.From3.Offset |= obj.WRAPPER
		}
	}

	ginit()

	gcargs := makefuncdatasym("gcargs·", obj.FUNCDATA_ArgsPointerMaps)
	gclocals := makefuncdatasym("gclocals·", obj.FUNCDATA_LocalsPointerMaps)

	if obj.Fieldtrack_enabled != 0 && len(Curfn.Func.FieldTrack) > 0 {
		trackSyms := make([]*Sym, 0, len(Curfn.Func.FieldTrack))
		for sym := range Curfn.Func.FieldTrack {
			trackSyms = append(trackSyms, sym)
		}
		sort.Sort(symByName(trackSyms))
		for _, sym := range trackSyms {
			gtrack(sym)
		}
	}

	for _, n := range fn.Func.Dcl {
		if n.Op != ONAME { // might be OTYPE or OLITERAL
			continue
		}
		switch n.Class {
		case PAUTO, PPARAM, PPARAMOUT:
			Nodconst(&nod1, Types[TUINTPTR], n.Type.Width)
			p := Thearch.Gins(obj.ATYPE, n, &nod1)
			p.From.Gotype = Linksym(ngotype(n))
		}
	}

	if ssafn != nil {
		genssa(ssafn, ptxt, gcargs, gclocals)
		ssafn.Free()
	} else {
		genlegacy(ptxt, gcargs, gclocals)
	}
}

type symByName []*Sym

func (a symByName) Len() int           { return len(a) }
func (a symByName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a symByName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// genlegacy compiles Curfn using the legacy non-SSA code generator.
func genlegacy(ptxt *obj.Prog, gcargs, gclocals *Sym) {
	Genlist(Curfn.Func.Enter)
	Genlist(Curfn.Nbody)
	gclean()
	checklabels()
	if nerrors != 0 {
		return
	}
	if Curfn.Func.Endlineno != 0 {
		lineno = Curfn.Func.Endlineno
	}

	if Curfn.Type.Results().NumFields() != 0 {
		Ginscall(throwreturn, 0)
	}

	ginit()

	// TODO: Determine when the final cgen_ret can be omitted. Perhaps always?
	cgen_ret(nil)

	if hasdefer {
		// deferreturn pretends to have one uintptr argument.
		// Reserve space for it so stack scanner is happy.
		if Maxarg < int64(Widthptr) {
			Maxarg = int64(Widthptr)
		}
	}

	gclean()
	if nerrors != 0 {
		return
	}

	Pc.As = obj.ARET // overwrite AEND
	Pc.Lineno = lineno

	fixjmp(ptxt)
	if Debug['N'] == 0 || Debug['R'] != 0 || Debug['P'] != 0 {
		regopt(ptxt)
		nilopt(ptxt)
	}

	Thearch.Expandchecks(ptxt)

	allocauto(ptxt)

	setlineno(Curfn)
	if Stksize+Maxarg > 1<<31 {
		Yyerror("stack frame too large (>2GB)")
		return
	}

	// Emit garbage collection symbols.
	liveness(Curfn, ptxt, gcargs, gclocals)

	Thearch.Defframe(ptxt)

	if Debug['f'] != 0 {
		frame(0)
	}

	// Remove leftover instrumentation from the instruction stream.
	removevardef(ptxt)
}
