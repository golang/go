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
)

// "Portable" code generation.

var makefuncdatasym_nsym int

func makefuncdatasym(pp *Progs, nameprefix string, funcdatakind int64) *Sym {
	sym := lookupN(nameprefix, makefuncdatasym_nsym)
	makefuncdatasym_nsym++
	p := pp.Prog(obj.AFUNCDATA)
	Addrconst(&p.From, funcdatakind)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = Linksym(sym)
	return sym
}

// TODO(mdempsky): Update to reference OpVar{Def,Kill,Live} instead
// and move to plive.go.

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

func (s *ssafn) AllocFrame(f *ssa.Func) {
	s.stksize = 0
	s.stkptrsize = 0
	fn := s.curfn.Func

	// Mark the PAUTO's unused.
	for _, ln := range fn.Dcl {
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
		scratchFpMem = tempAt(src.NoXPos, s.curfn, Types[TUINT64])
		scratchFpMem.SetUsed(scratchUsed)
	}

	sort.Sort(byStackVar(fn.Dcl))

	// Reassign stack offsets of the locals that are used.
	for i, n := range fn.Dcl {
		if n.Op != ONAME || n.Class != PAUTO {
			continue
		}
		if !n.Used() {
			fn.Dcl = fn.Dcl[:i]
			break
		}

		dowidth(n.Type)
		w := n.Type.Width
		if w >= thearch.MAXWIDTH || w < 0 {
			Fatalf("bad width")
		}
		s.stksize += w
		s.stksize = Rnd(s.stksize, int64(n.Type.Align))
		if haspointers(n.Type) {
			s.stkptrsize = s.stksize
		}
		if thearch.LinkArch.InFamily(sys.MIPS, sys.MIPS64, sys.ARM, sys.ARM64, sys.PPC64, sys.S390X) {
			s.stksize = Rnd(s.stksize, int64(Widthptr))
		}
		if s.stksize >= 1<<31 {
			yyerrorl(s.curfn.Pos, "stack frame too large (>2GB)")
		}

		n.Xoffset = -s.stksize
	}

	s.stksize = Rnd(s.stksize, int64(Widthreg))
	s.stkptrsize = Rnd(s.stkptrsize, int64(Widthreg))
}

func compile(fn *Node) {
	Curfn = fn
	dowidth(fn.Type)

	if fn.Nbody.Len() == 0 {
		emitptrargsmap()
		return
	}

	saveerrors()

	order(fn)
	if nerrors != 0 {
		return
	}

	walk(fn)
	if nerrors != 0 {
		return
	}
	checkcontrolflow(fn)
	if nerrors != 0 {
		return
	}
	if instrumenting {
		instrument(fn)
	}
	if nerrors != 0 {
		return
	}

	// From this point, there should be no uses of Curfn. Enforce that.
	Curfn = nil

	// Build an SSA backend function.
	ssafn := buildssa(fn)
	if nerrors != 0 {
		return
	}

	pp := newProgs(fn)
	genssa(ssafn, pp)
	fieldtrack(pp.Text.From.Sym, fn.Func.FieldTrack)
	pp.Flush()
}

func debuginfo(fnsym *obj.LSym, curfn interface{}) []*dwarf.Var {
	fn := curfn.(*Node)
	if expect := Linksym(fn.Func.Nname.Sym); fnsym != expect {
		Fatalf("unexpected fnsym: %v != %v", fnsym, expect)
	}

	var vars []*dwarf.Var
	for _, n := range fn.Func.Dcl {
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
