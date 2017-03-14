// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
	"cmd/internal/sys"
	"fmt"
	"sort"
	"strings"
)

// "Portable" code generation.

var makefuncdatasym_nsym int

func makefuncdatasym(nameprefix string, funcdatakind int64) *Sym {
	sym := lookupN(nameprefix, makefuncdatasym_nsym)
	makefuncdatasym_nsym++
	pnod := newname(sym)
	pnod.Class = PEXTERN
	p := Gins(obj.AFUNCDATA, nil, pnod)
	Addrconst(&p.From, funcdatakind)
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
		Fatalf("gvardef %#v; %v", n.Op, n)
		return
	}

	switch n.Class {
	case PAUTO, PPARAM, PPARAMOUT:
		if !n.Used() {
			Prog(obj.ANOP)
			return
		}

		if as == obj.AVARLIVE {
			Gins(as, n, nil)
		} else {
			Gins(as, nil, n)
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
	sym := lookup(fmt.Sprintf("%s.args_stackmap", Curfn.Func.Nname.Sym.Name))

	nptr := int(Curfn.Type.ArgWidth() / int64(Widthptr))
	bv := bvalloc(int32(nptr) * 2)
	nbitmap := 1
	if Curfn.Type.Results().NumFields() > 0 {
		nbitmap = 2
	}
	off := duint32(sym, 0, uint32(nbitmap))
	off = duint32(sym, off, uint32(bv.n))
	var xoffset int64
	if Curfn.IsMethod() {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Recvs(), &xoffset, bv)
	}

	if Curfn.Type.Params().NumFields() > 0 {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Params(), &xoffset, bv)
	}

	off = dbvec(sym, off, bv)
	if Curfn.Type.Results().NumFields() > 0 {
		xoffset = 0
		onebitwalktype1(Curfn.Type.Results(), &xoffset, bv)
		off = dbvec(sym, off, bv)
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

	if a.Used() != b.Used() {
		return a.Used()
	}

	ap := haspointers(a.Type)
	bp := haspointers(b.Type)
	if ap != bp {
		return ap
	}

	ap = a.Name.Needzero()
	bp = b.Name.Needzero()
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

var scratchFpMem *Node

func (s *ssaExport) AllocFrame(f *ssa.Func) {
	Stksize = 0
	stkptrsize = 0

	// Mark the PAUTO's unused.
	for _, ln := range Curfn.Func.Dcl {
		if ln.Class == PAUTO {
			ln.SetUsed(false)
		}
	}

	for _, l := range f.RegAlloc {
		if ls, ok := l.(ssa.LocalSlot); ok {
			ls.N.(*Node).SetUsed(true)
		}

	}

	scratchUsed := false
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch a := v.Aux.(type) {
			case *ssa.ArgSymbol:
				a.Node.(*Node).SetUsed(true)
			case *ssa.AutoSymbol:
				a.Node.(*Node).SetUsed(true)
			}

			if !scratchUsed {
				scratchUsed = v.Op.UsesScratch()
			}
		}
	}

	if f.Config.NeedsFpScratch {
		scratchFpMem = temp(Types[TUINT64])
		scratchFpMem.SetUsed(scratchUsed)
	}

	sort.Sort(byStackVar(Curfn.Func.Dcl))

	// Reassign stack offsets of the locals that are used.
	for i, n := range Curfn.Func.Dcl {
		if n.Op != ONAME || n.Class != PAUTO {
			continue
		}
		if !n.Used() {
			Curfn.Func.Dcl = Curfn.Func.Dcl[:i]
			break
		}

		dowidth(n.Type)
		w := n.Type.Width
		if w >= Thearch.MAXWIDTH || w < 0 {
			Fatalf("bad width")
		}
		Stksize += w
		Stksize = Rnd(Stksize, int64(n.Type.Align))
		if haspointers(n.Type) {
			stkptrsize = Stksize
		}
		if Thearch.LinkArch.InFamily(sys.MIPS, sys.MIPS64, sys.ARM, sys.ARM64, sys.PPC64, sys.S390X) {
			Stksize = Rnd(Stksize, int64(Widthptr))
		}
		if Stksize >= 1<<31 {
			setlineno(Curfn)
			yyerror("stack frame too large (>2GB)")
		}

		n.Xoffset = -Stksize
	}

	Stksize = Rnd(Stksize, int64(Widthreg))
	stkptrsize = Rnd(stkptrsize, int64(Widthreg))
}

func compile(fn *Node) {
	if Newproc == nil {
		Newproc = Sysfunc("newproc")
		Deferproc = Sysfunc("deferproc")
		Deferreturn = Sysfunc("deferreturn")
		Duffcopy = Sysfunc("duffcopy")
		Duffzero = Sysfunc("duffzero")
		panicindex = Sysfunc("panicindex")
		panicslice = Sysfunc("panicslice")
		panicdivide = Sysfunc("panicdivide")
		growslice = Sysfunc("growslice")
		panicdottypeE = Sysfunc("panicdottypeE")
		panicdottypeI = Sysfunc("panicdottypeI")
		panicnildottype = Sysfunc("panicnildottype")
		assertE2I = Sysfunc("assertE2I")
		assertE2I2 = Sysfunc("assertE2I2")
		assertI2I = Sysfunc("assertI2I")
		assertI2I2 = Sysfunc("assertI2I2")
	}

	defer func(lno src.XPos) {
		lineno = lno
	}(setlineno(fn))

	Curfn = fn
	dowidth(fn.Type)

	if fn.Nbody.Len() == 0 {
		if pure_go || strings.HasPrefix(fn.Func.Nname.Sym.Name, "init.") {
			yyerror("missing function body for %q", fn.Func.Nname.Sym.Name)
			return
		}

		emitptrargsmap()
		return
	}

	saveerrors()

	order(fn)
	if nerrors != 0 {
		return
	}

	hasdefer = false
	walk(fn)
	if nerrors != 0 {
		return
	}
	if instrumenting {
		instrument(fn)
	}
	if nerrors != 0 {
		return
	}

	// Build an SSA backend function.
	ssafn := buildssa(fn)
	if nerrors != 0 {
		return
	}

	plist := new(obj.Plist)
	pc = Ctxt.NewProg()
	Clearp(pc)
	plist.Firstpc = pc

	setlineno(fn)

	nam := fn.Func.Nname
	if isblank(nam) {
		nam = nil
	}
	ptxt := Gins(obj.ATEXT, nam, nil)
	fnsym := ptxt.From.Sym

	ptxt.From3 = new(obj.Addr)
	if fn.Func.Dupok() {
		ptxt.From3.Offset |= obj.DUPOK
	}
	if fn.Func.Wrapper() {
		ptxt.From3.Offset |= obj.WRAPPER
	}
	if fn.Func.NoFramePointer() {
		ptxt.From3.Offset |= obj.NOFRAME
	}
	if fn.Func.Needctxt() {
		ptxt.From3.Offset |= obj.NEEDCTXT
	}
	if fn.Func.Pragma&Nosplit != 0 {
		ptxt.From3.Offset |= obj.NOSPLIT
	}
	if fn.Func.ReflectMethod() {
		ptxt.From3.Offset |= obj.REFLECTMETHOD
	}
	if fn.Func.Pragma&Systemstack != 0 {
		ptxt.From.Sym.Set(obj.AttrCFunc, true)
	}

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if myimportpath == "reflect" {
		if fn.Func.Nname.Sym.Name == "callReflect" || fn.Func.Nname.Sym.Name == "callMethod" {
			ptxt.From3.Offset |= obj.WRAPPER
		}
	}

	gcargs := makefuncdatasym("gcargs·", obj.FUNCDATA_ArgsPointerMaps)
	gclocals := makefuncdatasym("gclocals·", obj.FUNCDATA_LocalsPointerMaps)

	genssa(ssafn, ptxt, gcargs, gclocals)
	ssafn.Free()

	obj.Flushplist(Ctxt, plist) // convert from Prog list to machine code
	ptxt = nil                  // nil to prevent misuse; Prog may have been freed by Flushplist

	fieldtrack(fnsym, fn.Func.FieldTrack)
}

func debuginfo(fnsym *obj.LSym) []*dwarf.Var {
	if expect := Linksym(Curfn.Func.Nname.Sym); fnsym != expect {
		Fatalf("unexpected fnsym: %v != %v", fnsym, expect)
	}

	var vars []*dwarf.Var
	for _, n := range Curfn.Func.Dcl {
		if n.Op != ONAME { // might be OTYPE or OLITERAL
			continue
		}

		var name obj.AddrName
		var abbrev int
		offs := n.Xoffset

		switch n.Class {
		case PAUTO:
			if !n.Used() {
				continue
			}
			name = obj.NAME_AUTO

			abbrev = dwarf.DW_ABRV_AUTO
			if Ctxt.FixedFrameSize() == 0 {
				offs -= int64(Widthptr)
			}
			if obj.Framepointer_enabled(obj.GOOS, obj.GOARCH) {
				offs -= int64(Widthptr)
			}

		case PPARAM, PPARAMOUT:
			name = obj.NAME_PARAM

			abbrev = dwarf.DW_ABRV_PARAM
			offs += Ctxt.FixedFrameSize()

		default:
			continue
		}

		gotype := Linksym(ngotype(n))
		fnsym.Autom = append(fnsym.Autom, &obj.Auto{
			Asym:    obj.Linklookup(Ctxt, n.Sym.Name, 0),
			Aoffset: int32(n.Xoffset),
			Name:    name,
			Gotype:  gotype,
		})

		if n.IsAutoTmp() {
			continue
		}

		typename := dwarf.InfoPrefix + gotype.Name[len("type."):]
		vars = append(vars, &dwarf.Var{
			Name:   n.Sym.Name,
			Abbrev: abbrev,
			Offset: int32(offs),
			Type:   obj.Linklookup(Ctxt, typename, 0),
		})
	}

	// Stable sort so that ties are broken with declaration order.
	sort.Stable(dwarf.VarsByOffset(vars))

	return vars
}

// fieldtrack adds R_USEFIELD relocations to fnsym to record any
// struct fields that it used.
func fieldtrack(fnsym *obj.LSym, tracked map[*Sym]struct{}) {
	if fnsym == nil {
		return
	}
	if obj.Fieldtrack_enabled == 0 || len(tracked) == 0 {
		return
	}

	trackSyms := make([]*Sym, 0, len(tracked))
	for sym := range tracked {
		trackSyms = append(trackSyms, sym)
	}
	sort.Sort(symByName(trackSyms))
	for _, sym := range trackSyms {
		r := obj.Addrel(fnsym)
		r.Sym = Linksym(sym)
		r.Type = obj.R_USEFIELD
	}
}

type symByName []*Sym

func (a symByName) Len() int           { return len(a) }
func (a symByName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a symByName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
