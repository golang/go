// Derived from Inferno utils/6c/txt.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6c/txt.c
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
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"fmt"
	"os"
)

var sharedProgArray = new([10000]obj.Prog) // *T instead of T to work around issue 19839

// Progs accumulates Progs for a function and converts them into machine code.
type Progs struct {
	Text      *obj.Prog  // ATEXT Prog for this function
	next      *obj.Prog  // next Prog
	pc        int64      // virtual PC; count of Progs
	pos       src.XPos   // position to use for new Progs
	curfn     *ir.Func   // fn these Progs are for
	progcache []obj.Prog // local progcache
	cacheidx  int        // first free element of progcache

	nextLive LivenessIndex // liveness index for the next Prog
	prevLive LivenessIndex // last emitted liveness index
}

// newProgs returns a new Progs for fn.
// worker indicates which of the backend workers will use the Progs.
func newProgs(fn *ir.Func, worker int) *Progs {
	pp := new(Progs)
	if base.Ctxt.CanReuseProgs() {
		sz := len(sharedProgArray) / base.Flag.LowerC
		pp.progcache = sharedProgArray[sz*worker : sz*(worker+1)]
	}
	pp.curfn = fn

	// prime the pump
	pp.next = pp.NewProg()
	pp.clearp(pp.next)

	pp.pos = fn.Pos()
	pp.settext(fn)
	// PCDATA tables implicitly start with index -1.
	pp.prevLive = LivenessIndex{-1, false}
	pp.nextLive = pp.prevLive
	return pp
}

func (pp *Progs) NewProg() *obj.Prog {
	var p *obj.Prog
	if pp.cacheidx < len(pp.progcache) {
		p = &pp.progcache[pp.cacheidx]
		pp.cacheidx++
	} else {
		p = new(obj.Prog)
	}
	p.Ctxt = base.Ctxt
	return p
}

// Flush converts from pp to machine code.
func (pp *Progs) Flush() {
	plist := &obj.Plist{Firstpc: pp.Text, Curfn: pp.curfn}
	obj.Flushplist(base.Ctxt, plist, pp.NewProg, base.Ctxt.Pkgpath)
}

// Free clears pp and any associated resources.
func (pp *Progs) Free() {
	if base.Ctxt.CanReuseProgs() {
		// Clear progs to enable GC and avoid abuse.
		s := pp.progcache[:pp.cacheidx]
		for i := range s {
			s[i] = obj.Prog{}
		}
	}
	// Clear pp to avoid abuse.
	*pp = Progs{}
}

// Prog adds a Prog with instruction As to pp.
func (pp *Progs) Prog(as obj.As) *obj.Prog {
	if pp.nextLive.StackMapValid() && pp.nextLive.stackMapIndex != pp.prevLive.stackMapIndex {
		// Emit stack map index change.
		idx := pp.nextLive.stackMapIndex
		pp.prevLive.stackMapIndex = idx
		p := pp.Prog(obj.APCDATA)
		Addrconst(&p.From, objabi.PCDATA_StackMapIndex)
		Addrconst(&p.To, int64(idx))
	}
	if pp.nextLive.isUnsafePoint != pp.prevLive.isUnsafePoint {
		// Emit unsafe-point marker.
		pp.prevLive.isUnsafePoint = pp.nextLive.isUnsafePoint
		p := pp.Prog(obj.APCDATA)
		Addrconst(&p.From, objabi.PCDATA_UnsafePoint)
		if pp.nextLive.isUnsafePoint {
			Addrconst(&p.To, objabi.PCDATA_UnsafePointUnsafe)
		} else {
			Addrconst(&p.To, objabi.PCDATA_UnsafePointSafe)
		}
	}

	p := pp.next
	pp.next = pp.NewProg()
	pp.clearp(pp.next)
	p.Link = pp.next

	if !pp.pos.IsKnown() && base.Flag.K != 0 {
		base.Warn("prog: unknown position (line 0)")
	}

	p.As = as
	p.Pos = pp.pos
	if pp.pos.IsStmt() == src.PosIsStmt {
		// Clear IsStmt for later Progs at this pos provided that as can be marked as a stmt
		if ssa.LosesStmtMark(as) {
			return p
		}
		pp.pos = pp.pos.WithNotStmt()
	}
	return p
}

func (pp *Progs) clearp(p *obj.Prog) {
	obj.Nopout(p)
	p.As = obj.AEND
	p.Pc = pp.pc
	pp.pc++
}

func (pp *Progs) Appendpp(p *obj.Prog, as obj.As, ftype obj.AddrType, freg int16, foffset int64, ttype obj.AddrType, treg int16, toffset int64) *obj.Prog {
	q := pp.NewProg()
	pp.clearp(q)
	q.As = as
	q.Pos = p.Pos
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

func (pp *Progs) settext(fn *ir.Func) {
	if pp.Text != nil {
		base.Fatalf("Progs.settext called twice")
	}
	ptxt := pp.Prog(obj.ATEXT)
	pp.Text = ptxt

	fn.LSym.Func().Text = ptxt
	ptxt.From.Type = obj.TYPE_MEM
	ptxt.From.Name = obj.NAME_EXTERN
	ptxt.From.Sym = fn.LSym
}

// makeABIWrapper creates a new function that wraps a cross-ABI call
// to "f".  The wrapper is marked as an ABIWRAPPER.
func makeABIWrapper(f *ir.Func, wrapperABI obj.ABI) {

	// Q: is this needed?
	savepos := base.Pos
	savedclcontext := dclcontext
	savedcurfn := Curfn

	base.Pos = autogeneratedPos
	dclcontext = ir.PEXTERN

	// At the moment we don't support wrapping a method, we'd need machinery
	// below to handle the receiver. Panic if we see this scenario.
	ft := f.Nname.Ntype.Type()
	if ft.NumRecvs() != 0 {
		panic("makeABIWrapper support for wrapping methods not implemented")
	}

	// Manufacture a new func type to use for the wrapper.
	var noReceiver *ir.Field
	tfn := ir.NewFuncType(base.Pos,
		noReceiver,
		structargs(ft.Params(), true),
		structargs(ft.Results(), false))

	// Reuse f's types.Sym to create a new ODCLFUNC/function.
	fn := dclfunc(f.Nname.Sym(), tfn)
	fn.SetDupok(true)
	fn.SetWrapper(true) // ignore frame for panic+recover matching

	// Select LSYM now.
	asym := base.Ctxt.LookupABI(f.LSym.Name, wrapperABI)
	asym.Type = objabi.STEXT
	if fn.LSym != nil {
		panic("unexpected")
	}
	fn.LSym = asym

	// ABI0-to-ABIInternal wrappers will be mainly loading params from
	// stack into registers (and/or storing stack locations back to
	// registers after the wrapped call); in most cases they won't
	// need to allocate stack space, so it should be OK to mark them
	// as NOSPLIT in these cases. In addition, my assumption is that
	// functions written in assembly are NOSPLIT in most (but not all)
	// cases. In the case of an ABIInternal target that has too many
	// parameters to fit into registers, the wrapper would need to
	// allocate stack space, but this seems like an unlikely scenario.
	// Hence: mark these wrappers NOSPLIT.
	//
	// ABIInternal-to-ABI0 wrappers on the other hand will be taking
	// things in registers and pushing them onto the stack prior to
	// the ABI0 call, meaning that they will always need to allocate
	// stack space. If the compiler marks them as NOSPLIT this seems
	// as though it could lead to situations where the the linker's
	// nosplit-overflow analysis would trigger a link failure. On the
	// other hand if they not tagged NOSPLIT then this could cause
	// problems when building the runtime (since there may be calls to
	// asm routine in cases where it's not safe to grow the stack). In
	// most cases the wrapper would be (in effect) inlined, but are
	// there (perhaps) indirect calls from the runtime that could run
	// into trouble here.
	// FIXME: at the moment all.bash does not pass when I leave out
	// NOSPLIT for these wrappers, so all are currently tagged with NOSPLIT.
	setupTextLSym(fn, obj.NOSPLIT|obj.ABIWRAPPER)

	// Generate call. Use tail call if no params and no returns,
	// but a regular call otherwise.
	//
	// Note: ideally we would be using a tail call in cases where
	// there are params but no returns for ABI0->ABIInternal wrappers,
	// provided that all params fit into registers (e.g. we don't have
	// to allocate any stack space). Doing this will require some
	// extra work in typecheck/walk/ssa, might want to add a new node
	// OTAILCALL or something to this effect.
	var tail ir.Node
	if tfn.Type().NumResults() == 0 && tfn.Type().NumParams() == 0 && tfn.Type().NumRecvs() == 0 {
		tail = ir.NewBranchStmt(base.Pos, ir.ORETJMP, f.Nname.Sym())
	} else {
		call := ir.NewCallExpr(base.Pos, ir.OCALL, f.Nname, nil)
		call.PtrList().Set(paramNnames(tfn.Type()))
		call.SetIsDDD(tfn.Type().IsVariadic())
		tail = call
		if tfn.Type().NumResults() > 0 {
			n := ir.NewReturnStmt(base.Pos, nil)
			n.PtrList().Set1(call)
			tail = n
		}
	}
	fn.PtrBody().Append(tail)

	funcbody()
	if base.Debug.DclStack != 0 {
		testdclstack()
	}

	typecheckFunc(fn)
	Curfn = fn
	typecheckslice(fn.Body().Slice(), ctxStmt)

	escapeFuncs([]*ir.Func{fn}, false)

	Target.Decls = append(Target.Decls, fn)

	// Restore previous context.
	base.Pos = savepos
	dclcontext = savedclcontext
	Curfn = savedcurfn
}

// initLSym defines f's obj.LSym and initializes it based on the
// properties of f. This includes setting the symbol flags and ABI and
// creating and initializing related DWARF symbols.
//
// initLSym must be called exactly once per function and must be
// called for both functions with bodies and functions without bodies.
// For body-less functions, we only create the LSym; for functions
// with bodies call a helper to setup up / populate the LSym.
func initLSym(f *ir.Func, hasBody bool) {
	// FIXME: for new-style ABI wrappers, we set up the lsym at the
	// point the wrapper is created.
	if f.LSym != nil && base.Flag.ABIWrap {
		return
	}
	selectLSym(f, hasBody)
	if hasBody {
		setupTextLSym(f, 0)
	}
}

// selectLSym sets up the LSym for a given function, and
// makes calls to helpers to create ABI wrappers if needed.
func selectLSym(f *ir.Func, hasBody bool) {
	if f.LSym != nil {
		base.Fatalf("Func.initLSym called twice")
	}

	if nam := f.Nname; !ir.IsBlank(nam) {

		var wrapperABI obj.ABI
		needABIWrapper := false
		defABI, hasDefABI := symabiDefs[nam.Sym().LinksymName()]
		if hasDefABI && defABI == obj.ABI0 {
			// Symbol is defined as ABI0. Create an
			// Internal -> ABI0 wrapper.
			f.LSym = nam.Sym().LinksymABI0()
			needABIWrapper, wrapperABI = true, obj.ABIInternal
		} else {
			f.LSym = nam.Sym().Linksym()
			// No ABI override. Check that the symbol is
			// using the expected ABI.
			want := obj.ABIInternal
			if f.LSym.ABI() != want {
				base.Fatalf("function symbol %s has the wrong ABI %v, expected %v", f.LSym.Name, f.LSym.ABI(), want)
			}
		}
		if f.Pragma&ir.Systemstack != 0 {
			f.LSym.Set(obj.AttrCFunc, true)
		}

		isLinknameExported := nam.Sym().Linkname != "" && (hasBody || hasDefABI)
		if abi, ok := symabiRefs[f.LSym.Name]; (ok && abi == obj.ABI0) || isLinknameExported {
			// Either 1) this symbol is definitely
			// referenced as ABI0 from this package; or 2)
			// this symbol is defined in this package but
			// given a linkname, indicating that it may be
			// referenced from another package. Create an
			// ABI0 -> Internal wrapper so it can be
			// called as ABI0. In case 2, it's important
			// that we know it's defined in this package
			// since other packages may "pull" symbols
			// using linkname and we don't want to create
			// duplicate ABI wrappers.
			if f.LSym.ABI() != obj.ABI0 {
				needABIWrapper, wrapperABI = true, obj.ABI0
			}
		}

		if needABIWrapper {
			if !useABIWrapGen(f) {
				// Fallback: use alias instead. FIXME.

				// These LSyms have the same name as the
				// native function, so we create them directly
				// rather than looking them up. The uniqueness
				// of f.lsym ensures uniqueness of asym.
				asym := &obj.LSym{
					Name: f.LSym.Name,
					Type: objabi.SABIALIAS,
					R:    []obj.Reloc{{Sym: f.LSym}}, // 0 size, so "informational"
				}
				asym.SetABI(wrapperABI)
				asym.Set(obj.AttrDuplicateOK, true)
				base.Ctxt.ABIAliases = append(base.Ctxt.ABIAliases, asym)
			} else {
				if base.Debug.ABIWrap != 0 {
					fmt.Fprintf(os.Stderr, "=-= %v to %v wrapper for %s.%s\n",
						wrapperABI, 1-wrapperABI, types.LocalPkg.Path, f.LSym.Name)
				}
				makeABIWrapper(f, wrapperABI)
			}
		}
	}
}

// setupTextLsym initializes the LSym for a with-body text symbol.
func setupTextLSym(f *ir.Func, flag int) {
	if f.Dupok() {
		flag |= obj.DUPOK
	}
	if f.Wrapper() {
		flag |= obj.WRAPPER
	}
	if f.Needctxt() {
		flag |= obj.NEEDCTXT
	}
	if f.Pragma&ir.Nosplit != 0 {
		flag |= obj.NOSPLIT
	}
	if f.ReflectMethod() {
		flag |= obj.REFLECTMETHOD
	}

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if base.Ctxt.Pkgpath == "reflect" {
		switch f.Sym().Name {
		case "callReflect", "callMethod":
			flag |= obj.WRAPPER
		}
	}

	base.Ctxt.InitTextSym(f.LSym, flag)
}

func ggloblnod(nam ir.Node) {
	s := nam.Sym().Linksym()
	s.Gotype = ngotype(nam).Linksym()
	flags := 0
	if nam.Name().Readonly() {
		flags = obj.RODATA
	}
	if nam.Type() != nil && !nam.Type().HasPointers() {
		flags |= obj.NOPTR
	}
	base.Ctxt.Globl(s, nam.Type().Width, flags)
	if nam.Name().LibfuzzerExtraCounter() {
		s.Type = objabi.SLIBFUZZER_EXTRA_COUNTER
	}
	if nam.Sym().Linkname != "" {
		// Make sure linkname'd symbol is non-package. When a symbol is
		// both imported and linkname'd, s.Pkg may not set to "_" in
		// types.Sym.Linksym because LSym already exists. Set it here.
		s.Pkg = "_"
	}
}

func ggloblsym(s *obj.LSym, width int32, flags int16) {
	if flags&obj.LOCAL != 0 {
		s.Set(obj.AttrLocal, true)
		flags &^= obj.LOCAL
	}
	base.Ctxt.Globl(s, int64(width), int(flags))
}

func Addrconst(a *obj.Addr, v int64) {
	a.SetConst(v)
}

func Patch(p *obj.Prog, to *obj.Prog) {
	p.To.SetTarget(to)
}
