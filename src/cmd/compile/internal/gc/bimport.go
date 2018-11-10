// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Binary package import.
// See bexport.go for the export data format and how
// to make a format change.

package gc

import (
	"bufio"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"encoding/binary"
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

// The overall structure of Import is symmetric to Export: For each
// export method in bexport.go there is a matching and symmetric method
// in bimport.go. Changing the export format requires making symmetric
// changes to bimport.go and bexport.go.

type importer struct {
	in      *bufio.Reader
	imp     *types.Pkg // imported package
	buf     []byte     // reused for reading strings
	version int        // export format version

	// object lists, in order of deserialization
	strList       []string
	pathList      []string
	pkgList       []*types.Pkg
	typList       []*types.Type
	funcList      []*Node // nil entry means already declared
	trackAllTypes bool

	// for delayed type verification
	cmpList []struct{ pt, t *types.Type }

	// position encoding
	posInfoFormat bool
	prevFile      string
	prevLine      int
	posBase       *src.PosBase

	// debugging support
	debugFormat bool
	read        int // bytes read
}

// Import populates imp from the serialized package data read from in.
func Import(imp *types.Pkg, in *bufio.Reader) {
	inimport = true
	defer func() { inimport = false }()

	p := importer{
		in:       in,
		imp:      imp,
		version:  -1,           // unknown version
		strList:  []string{""}, // empty string is mapped to 0
		pathList: []string{""}, // empty path is mapped to 0
	}

	// read version info
	var versionstr string
	if b := p.rawByte(); b == 'c' || b == 'd' {
		// Go1.7 encoding; first byte encodes low-level
		// encoding format (compact vs debug).
		// For backward-compatibility only (avoid problems with
		// old installed packages). Newly compiled packages use
		// the extensible format string.
		// TODO(gri) Remove this support eventually; after Go1.8.
		if b == 'd' {
			p.debugFormat = true
		}
		p.trackAllTypes = p.rawByte() == 'a'
		p.posInfoFormat = p.bool()
		versionstr = p.string()
		if versionstr == "v1" {
			p.version = 0
		}
	} else {
		// Go1.8 extensible encoding
		// read version string and extract version number (ignore anything after the version number)
		versionstr = p.rawStringln(b)
		if s := strings.SplitN(versionstr, " ", 3); len(s) >= 2 && s[0] == "version" {
			if v, err := strconv.Atoi(s[1]); err == nil && v > 0 {
				p.version = v
			}
		}
	}

	// read version specific flags - extend as necessary
	switch p.version {
	// case 6:
	// 	...
	//	fallthrough
	case 5, 4, 3, 2, 1:
		p.debugFormat = p.rawStringln(p.rawByte()) == "debug"
		p.trackAllTypes = p.bool()
		p.posInfoFormat = p.bool()
	case 0:
		// Go1.7 encoding format - nothing to do here
	default:
		p.formatErrorf("unknown export format version %d (%q)", p.version, versionstr)
	}

	// --- generic export data ---

	// populate typList with predeclared "known" types
	p.typList = append(p.typList, predeclared()...)

	// read package data
	p.pkg()

	// defer some type-checking until all types are read in completely
	tcok := typecheckok
	typecheckok = true
	defercheckwidth()

	// read objects

	// phase 1
	objcount := 0
	for {
		tag := p.tagOrIndex()
		if tag == endTag {
			break
		}
		p.obj(tag)
		objcount++
	}

	// self-verification
	if count := p.int(); count != objcount {
		p.formatErrorf("got %d objects; want %d", objcount, count)
	}

	// --- compiler-specific export data ---

	// read compiler-specific flags

	// phase 2
	objcount = 0
	for {
		tag := p.tagOrIndex()
		if tag == endTag {
			break
		}
		p.obj(tag)
		objcount++
	}

	// self-verification
	if count := p.int(); count != objcount {
		p.formatErrorf("got %d objects; want %d", objcount, count)
	}

	// read inlineable functions bodies
	if dclcontext != PEXTERN {
		p.formatErrorf("unexpected context %d", dclcontext)
	}

	objcount = 0
	for i0 := -1; ; {
		i := p.int() // index of function with inlineable body
		if i < 0 {
			break
		}

		// don't process the same function twice
		if i <= i0 {
			p.formatErrorf("index not increasing: %d <= %d", i, i0)
		}
		i0 = i

		if funcdepth != 0 {
			p.formatErrorf("unexpected Funcdepth %d", funcdepth)
		}

		// Note: In the original code, funchdr and funcbody are called for
		// all functions (that were not yet imported). Now, we are calling
		// them only for functions with inlineable bodies. funchdr does
		// parameter renaming which doesn't matter if we don't have a body.

		if f := p.funcList[i]; f != nil {
			// function not yet imported - read body and set it
			funchdr(f)
			body := p.stmtList()
			if body == nil {
				// Make sure empty body is not interpreted as
				// no inlineable body (see also parser.fnbody)
				// (not doing so can cause significant performance
				// degradation due to unnecessary calls to empty
				// functions).
				body = []*Node{nod(OEMPTY, nil, nil)}
			}
			f.Func.Inl.Set(body)
			funcbody(f)
		} else {
			// function already imported - read body but discard declarations
			dclcontext = PDISCARD // throw away any declarations
			p.stmtList()
			dclcontext = PEXTERN
		}

		objcount++
	}

	// self-verification
	if count := p.int(); count != objcount {
		p.formatErrorf("got %d functions; want %d", objcount, count)
	}

	if dclcontext != PEXTERN {
		p.formatErrorf("unexpected context %d", dclcontext)
	}

	p.verifyTypes()

	// --- end of export data ---

	typecheckok = tcok
	resumecheckwidth()

	if debug_dclstack != 0 {
		testdclstack()
	}
}

func (p *importer) formatErrorf(format string, args ...interface{}) {
	if debugFormat {
		Fatalf(format, args...)
	}

	yyerror("cannot import %q due to version skew - reinstall package (%s)",
		p.imp.Path, fmt.Sprintf(format, args...))
	errorexit()
}

func (p *importer) verifyTypes() {
	for _, pair := range p.cmpList {
		pt := pair.pt
		t := pair.t
		if !eqtype(pt.Orig, t) {
			p.formatErrorf("inconsistent definition for type %v during import\n\t%L (in %q)\n\t%L (in %q)", pt.Sym, pt, pt.Sym.Importdef.Path, t, p.imp.Path)
		}
	}
}

// numImport tracks how often a package with a given name is imported.
// It is used to provide a better error message (by using the package
// path to disambiguate) if a package that appears multiple times with
// the same name appears in an error message.
var numImport = make(map[string]int)

func (p *importer) pkg() *types.Pkg {
	// if the package was seen before, i is its index (>= 0)
	i := p.tagOrIndex()
	if i >= 0 {
		return p.pkgList[i]
	}

	// otherwise, i is the package tag (< 0)
	if i != packageTag {
		p.formatErrorf("expected package tag, found tag = %d", i)
	}

	// read package data
	name := p.string()
	var path string
	if p.version >= 5 {
		path = p.path()
	} else {
		path = p.string()
	}

	// we should never see an empty package name
	if name == "" {
		p.formatErrorf("empty package name for path %q", path)
	}

	// we should never see a bad import path
	if isbadimport(path, true) {
		p.formatErrorf("bad package path %q for package %s", path, name)
	}

	// an empty path denotes the package we are currently importing;
	// it must be the first package we see
	if (path == "") != (len(p.pkgList) == 0) {
		p.formatErrorf("package path %q for pkg index %d", path, len(p.pkgList))
	}

	// add package to pkgList
	pkg := p.imp
	if path != "" {
		pkg = types.NewPkg(path, "")
	}
	if pkg.Name == "" {
		pkg.Name = name
		numImport[name]++
	} else if pkg.Name != name {
		yyerror("conflicting package names %s and %s for path %q", pkg.Name, name, path)
	}
	if myimportpath != "" && path == myimportpath {
		yyerror("import %q: package depends on %q (import cycle)", p.imp.Path, path)
		errorexit()
	}
	p.pkgList = append(p.pkgList, pkg)

	return pkg
}

func idealType(typ *types.Type) *types.Type {
	if typ.IsUntyped() {
		// canonicalize ideal types
		typ = types.Types[TIDEAL]
	}
	return typ
}

func (p *importer) obj(tag int) {
	switch tag {
	case constTag:
		p.pos()
		sym := p.qualifiedName()
		typ := p.typ()
		val := p.value(typ)
		importconst(p.imp, sym, idealType(typ), nodlit(val))

	case aliasTag:
		p.pos()
		sym := p.qualifiedName()
		typ := p.typ()
		importalias(p.imp, sym, typ)

	case typeTag:
		p.typ()

	case varTag:
		p.pos()
		sym := p.qualifiedName()
		typ := p.typ()
		importvar(p.imp, sym, typ)

	case funcTag:
		p.pos()
		sym := p.qualifiedName()
		params := p.paramList()
		result := p.paramList()

		sig := functypefield(nil, params, result)
		importsym(p.imp, sym, ONAME)
		if asNode(sym.Def) != nil && asNode(sym.Def).Op == ONAME {
			// function was imported before (via another import)
			if !eqtype(sig, asNode(sym.Def).Type) {
				p.formatErrorf("inconsistent definition for func %v during import\n\t%v\n\t%v", sym, asNode(sym.Def).Type, sig)
			}
			p.funcList = append(p.funcList, nil)
			break
		}

		n := newfuncname(sym)
		n.Type = sig
		declare(n, PFUNC)
		p.funcList = append(p.funcList, n)
		importlist = append(importlist, n)

		if Debug['E'] > 0 {
			fmt.Printf("import [%q] func %v \n", p.imp.Path, n)
			if Debug['m'] > 2 && n.Func.Inl.Len() != 0 {
				fmt.Printf("inl body: %v\n", n.Func.Inl)
			}
		}

	default:
		p.formatErrorf("unexpected object (tag = %d)", tag)
	}
}

func (p *importer) pos() src.XPos {
	if !p.posInfoFormat {
		return src.NoXPos
	}

	file := p.prevFile
	line := p.prevLine
	delta := p.int()
	line += delta
	if p.version >= 5 {
		if delta == deltaNewFile {
			if n := p.int(); n >= 0 {
				// file changed
				file = p.path()
				line = n
			}
		}
	} else {
		if delta == 0 {
			if n := p.int(); n >= 0 {
				// file changed
				file = p.prevFile[:n] + p.string()
				line = p.int()
			}
		}
	}
	if file != p.prevFile {
		p.prevFile = file
		p.posBase = src.NewFileBase(file, file)
	}
	p.prevLine = line

	pos := src.MakePos(p.posBase, uint(line), 0)
	xpos := Ctxt.PosTable.XPos(pos)
	return xpos
}

func (p *importer) path() string {
	// if the path was seen before, i is its index (>= 0)
	// (the empty string is at index 0)
	i := p.int()
	if i >= 0 {
		return p.pathList[i]
	}
	// otherwise, i is the negative path length (< 0)
	a := make([]string, -i)
	for n := range a {
		a[n] = p.string()
	}
	s := strings.Join(a, "/")
	p.pathList = append(p.pathList, s)
	return s
}

func (p *importer) newtyp(etype types.EType) *types.Type {
	t := types.New(etype)
	if p.trackAllTypes {
		p.typList = append(p.typList, t)
	}
	return t
}

// importtype declares that pt, an imported named type, has underlying type t.
func (p *importer) importtype(pt, t *types.Type) {
	if pt.Etype == TFORW {
		copytype(asNode(pt.Nod), t)
		pt.Sym.Importdef = p.imp
		pt.Sym.Lastlineno = lineno
		declare(asNode(pt.Nod), PEXTERN)
		checkwidth(pt)
	} else {
		// pt.Orig and t must be identical.
		if p.trackAllTypes {
			// If we track all types, t may not be fully set up yet.
			// Collect the types and verify identity later.
			p.cmpList = append(p.cmpList, struct{ pt, t *types.Type }{pt, t})
		} else if !eqtype(pt.Orig, t) {
			yyerror("inconsistent definition for type %v during import\n\t%L (in %q)\n\t%L (in %q)", pt.Sym, pt, pt.Sym.Importdef.Path, t, p.imp.Path)
		}
	}

	if Debug['E'] != 0 {
		fmt.Printf("import type %v %L\n", pt, t)
	}
}

func (p *importer) typ() *types.Type {
	// if the type was seen before, i is its index (>= 0)
	i := p.tagOrIndex()
	if i >= 0 {
		return p.typList[i]
	}

	// otherwise, i is the type tag (< 0)
	var t *types.Type
	switch i {
	case namedTag:
		p.pos()
		tsym := p.qualifiedName()

		t = pkgtype(p.imp, tsym)
		p.typList = append(p.typList, t)

		// read underlying type
		t0 := p.typ()
		p.importtype(t, t0)

		// interfaces don't have associated methods
		if t0.IsInterface() {
			break
		}

		// set correct import context (since p.typ() may be called
		// while importing the body of an inlined function)
		savedContext := dclcontext
		dclcontext = PEXTERN

		// read associated methods
		for i := p.int(); i > 0; i-- {
			p.pos()
			sym := p.fieldSym()

			// during import unexported method names should be in the type's package
			if !exportname(sym.Name) && sym.Pkg != tsym.Pkg {
				Fatalf("imported method name %+v in wrong package %s\n", sym, tsym.Pkg.Name)
			}

			recv := p.paramList() // TODO(gri) do we need a full param list for the receiver?
			params := p.paramList()
			result := p.paramList()
			nointerface := p.bool()

			n := newfuncname(methodname(sym, recv[0].Type))
			n.Type = functypefield(recv[0], params, result)
			checkwidth(n.Type)
			addmethod(sym, n.Type, false, nointerface)
			p.funcList = append(p.funcList, n)
			importlist = append(importlist, n)

			// (comment from parser.go)
			// inl.C's inlnode in on a dotmeth node expects to find the inlineable body as
			// (dotmeth's type).Nname.Inl, and dotmeth's type has been pulled
			// out by typecheck's lookdot as this $$.ttype. So by providing
			// this back link here we avoid special casing there.
			n.Type.FuncType().Nname = asTypesNode(n)

			if Debug['E'] > 0 {
				fmt.Printf("import [%q] meth %v \n", p.imp.Path, n)
				if Debug['m'] > 2 && n.Func.Inl.Len() != 0 {
					fmt.Printf("inl body: %v\n", n.Func.Inl)
				}
			}
		}

		dclcontext = savedContext

	case arrayTag:
		t = p.newtyp(TARRAY)
		bound := p.int64()
		elem := p.typ()
		t.Extra = &types.Array{Elem: elem, Bound: bound}

	case sliceTag:
		t = p.newtyp(TSLICE)
		elem := p.typ()
		t.Extra = types.Slice{Elem: elem}

	case dddTag:
		t = p.newtyp(TDDDFIELD)
		t.Extra = types.DDDField{T: p.typ()}

	case structTag:
		t = p.newtyp(TSTRUCT)
		t.SetFields(p.fieldList())
		checkwidth(t)

	case pointerTag:
		t = p.newtyp(types.Tptr)
		t.Extra = types.Ptr{Elem: p.typ()}

	case signatureTag:
		t = p.newtyp(TFUNC)
		params := p.paramList()
		result := p.paramList()
		functypefield0(t, nil, params, result)

	case interfaceTag:
		if ml := p.methodList(); len(ml) == 0 {
			t = types.Types[TINTER]
		} else {
			t = p.newtyp(TINTER)
			t.SetInterface(ml)
		}

	case mapTag:
		t = p.newtyp(TMAP)
		mt := t.MapType()
		mt.Key = p.typ()
		mt.Val = p.typ()

	case chanTag:
		t = p.newtyp(TCHAN)
		ct := t.ChanType()
		ct.Dir = types.ChanDir(p.int())
		ct.Elem = p.typ()

	default:
		p.formatErrorf("unexpected type (tag = %d)", i)
	}

	if t == nil {
		p.formatErrorf("nil type (type tag = %d)", i)
	}

	return t
}

func (p *importer) qualifiedName() *types.Sym {
	name := p.string()
	pkg := p.pkg()
	return pkg.Lookup(name)
}

func (p *importer) fieldList() (fields []*types.Field) {
	if n := p.int(); n > 0 {
		fields = make([]*types.Field, n)
		for i := range fields {
			fields[i] = p.field()
		}
	}
	return
}

func (p *importer) field() *types.Field {
	p.pos()
	sym, alias := p.fieldName()
	typ := p.typ()
	note := p.string()

	f := types.NewField()
	if sym.Name == "" {
		// anonymous field: typ must be T or *T and T must be a type name
		s := typ.Sym
		if s == nil && typ.IsPtr() {
			s = typ.Elem().Sym // deref
		}
		sym = sym.Pkg.Lookup(s.Name)
		f.Embedded = 1
	} else if alias {
		// anonymous field: we have an explicit name because it's a type alias
		f.Embedded = 1
	}

	f.Sym = sym
	f.Nname = asTypesNode(newname(sym))
	f.Type = typ
	f.Note = note

	return f
}

func (p *importer) methodList() (methods []*types.Field) {
	for n := p.int(); n > 0; n-- {
		f := types.NewField()
		f.Nname = asTypesNode(newname(nblank.Sym))
		asNode(f.Nname).Pos = p.pos()
		f.Type = p.typ()
		methods = append(methods, f)
	}

	for n := p.int(); n > 0; n-- {
		methods = append(methods, p.method())
	}

	return
}

func (p *importer) method() *types.Field {
	p.pos()
	sym := p.methodName()
	params := p.paramList()
	result := p.paramList()

	f := types.NewField()
	f.Sym = sym
	f.Nname = asTypesNode(newname(sym))
	f.Type = functypefield(fakeRecvField(), params, result)
	return f
}

func (p *importer) fieldName() (*types.Sym, bool) {
	name := p.string()
	if p.version == 0 && name == "_" {
		// version 0 didn't export a package for _ field names
		// but used the builtin package instead
		return builtinpkg.Lookup(name), false
	}
	pkg := localpkg
	alias := false
	switch name {
	case "":
		// 1) field name matches base type name and is exported: nothing to do
	case "?":
		// 2) field name matches base type name and is not exported: need package
		name = ""
		pkg = p.pkg()
	case "@":
		// 3) field name doesn't match base type name (alias name): need name and possibly package
		name = p.string()
		alias = true
		fallthrough
	default:
		if !exportname(name) {
			pkg = p.pkg()
		}
	}
	return pkg.Lookup(name), alias
}

func (p *importer) methodName() *types.Sym {
	name := p.string()
	if p.version == 0 && name == "_" {
		// version 0 didn't export a package for _ method names
		// but used the builtin package instead
		return builtinpkg.Lookup(name)
	}
	pkg := localpkg
	if !exportname(name) {
		pkg = p.pkg()
	}
	return pkg.Lookup(name)
}

func (p *importer) paramList() []*types.Field {
	i := p.int()
	if i == 0 {
		return nil
	}
	// negative length indicates unnamed parameters
	named := true
	if i < 0 {
		i = -i
		named = false
	}
	// i > 0
	fs := make([]*types.Field, i)
	for i := range fs {
		fs[i] = p.param(named)
	}
	return fs
}

func (p *importer) param(named bool) *types.Field {
	f := types.NewField()
	f.Type = p.typ()
	if f.Type.Etype == TDDDFIELD {
		// TDDDFIELD indicates wrapped ... slice type
		f.Type = types.NewSlice(f.Type.DDDField())
		f.SetIsddd(true)
	}

	if named {
		name := p.string()
		if name == "" {
			p.formatErrorf("expected named parameter")
		}
		// TODO(gri) Supply function/method package rather than
		// encoding the package for each parameter repeatedly.
		pkg := localpkg
		if name != "_" {
			pkg = p.pkg()
		}
		f.Sym = pkg.Lookup(name)
		f.Nname = asTypesNode(newname(f.Sym))
	}

	// TODO(gri) This is compiler-specific (escape info).
	// Move into compiler-specific section eventually?
	f.Note = p.string()

	return f
}

func (p *importer) value(typ *types.Type) (x Val) {
	switch tag := p.tagOrIndex(); tag {
	case falseTag:
		x.U = false

	case trueTag:
		x.U = true

	case int64Tag:
		u := new(Mpint)
		u.SetInt64(p.int64())
		u.Rune = typ == types.Idealrune
		x.U = u

	case floatTag:
		f := newMpflt()
		p.float(f)
		if typ == types.Idealint || typ.IsInteger() {
			// uncommon case: large int encoded as float
			u := new(Mpint)
			u.SetFloat(f)
			x.U = u
			break
		}
		x.U = f

	case complexTag:
		u := new(Mpcplx)
		p.float(&u.Real)
		p.float(&u.Imag)
		x.U = u

	case stringTag:
		x.U = p.string()

	case unknownTag:
		p.formatErrorf("unknown constant (importing package with errors)")

	case nilTag:
		x.U = new(NilVal)

	default:
		p.formatErrorf("unexpected value tag %d", tag)
	}

	// verify ideal type
	if typ.IsUntyped() && untype(x.Ctype()) != typ {
		p.formatErrorf("value %v and type %v don't match", x, typ)
	}

	return
}

func (p *importer) float(x *Mpflt) {
	sign := p.int()
	if sign == 0 {
		x.SetFloat64(0)
		return
	}

	exp := p.int()
	mant := new(big.Int).SetBytes([]byte(p.string()))

	m := x.Val.SetInt(mant)
	m.SetMantExp(m, exp-mant.BitLen())
	if sign < 0 {
		m.Neg(m)
	}
}

// ----------------------------------------------------------------------------
// Inlined function bodies

// Approach: Read nodes and use them to create/declare the same data structures
// as done originally by the (hidden) parser by closely following the parser's
// original code. In other words, "parsing" the import data (which happens to
// be encoded in binary rather textual form) is the best way at the moment to
// re-establish the syntax tree's invariants. At some future point we might be
// able to avoid this round-about way and create the rewritten nodes directly,
// possibly avoiding a lot of duplicate work (name resolution, type checking).
//
// Refined nodes (e.g., ODOTPTR as a refinement of OXDOT) are exported as their
// unrefined nodes (since this is what the importer uses). The respective case
// entries are unreachable in the importer.

func (p *importer) stmtList() []*Node {
	var list []*Node
	for {
		n := p.node()
		if n == nil {
			break
		}
		// OBLOCK nodes may be created when importing ODCL nodes - unpack them
		if n.Op == OBLOCK {
			list = append(list, n.List.Slice()...)
		} else {
			list = append(list, n)
		}
	}
	return list
}

func (p *importer) exprList() []*Node {
	var list []*Node
	for {
		n := p.expr()
		if n == nil {
			break
		}
		list = append(list, n)
	}
	return list
}

func (p *importer) elemList() []*Node {
	c := p.int()
	list := make([]*Node, c)
	for i := range list {
		s := p.fieldSym()
		list[i] = nodSym(OSTRUCTKEY, p.expr(), s)
	}
	return list
}

func (p *importer) expr() *Node {
	n := p.node()
	if n != nil && n.Op == OBLOCK {
		Fatalf("unexpected block node: %v", n)
	}
	return n
}

func npos(pos src.XPos, n *Node) *Node {
	n.Pos = pos
	return n
}

// TODO(gri) split into expr and stmt
func (p *importer) node() *Node {
	switch op := p.op(); op {
	// expressions
	// case OPAREN:
	// 	unreachable - unpacked by exporter

	// case ODDDARG:
	//	unimplemented

	case OLITERAL:
		pos := p.pos()
		typ := p.typ()
		n := npos(pos, nodlit(p.value(typ)))
		if !typ.IsUntyped() {
			// Type-checking simplifies unsafe.Pointer(uintptr(c))
			// to unsafe.Pointer(c) which then cannot type-checked
			// again. Re-introduce explicit uintptr(c) conversion.
			// (issue 16317).
			if typ.IsUnsafePtr() {
				n = nod(OCONV, n, nil)
				n.Type = types.Types[TUINTPTR]
			}
			n = nod(OCONV, n, nil)
			n.Type = typ
		}
		return n

	case ONAME:
		return npos(p.pos(), mkname(p.sym()))

	// case OPACK, ONONAME:
	// 	unreachable - should have been resolved by typechecking

	case OTYPE:
		pos := p.pos()
		if p.bool() {
			return npos(pos, mkname(p.sym()))
		}
		return npos(pos, typenod(p.typ()))

	// case OTARRAY, OTMAP, OTCHAN, OTSTRUCT, OTINTER, OTFUNC:
	//      unreachable - should have been resolved by typechecking

	// case OCLOSURE:
	//	unimplemented

	case OPTRLIT:
		n := npos(p.pos(), p.expr())
		if !p.bool() /* !implicit, i.e. '&' operator */ {
			if n.Op == OCOMPLIT {
				// Special case for &T{...}: turn into (*T){...}.
				n.Right = nod(OIND, n.Right, nil)
				n.Right.SetImplicit(true)
			} else {
				n = nod(OADDR, n, nil)
			}
		}
		return n

	case OSTRUCTLIT:
		n := nodl(p.pos(), OCOMPLIT, nil, typenod(p.typ()))
		n.List.Set(p.elemList()) // special handling of field names
		return n

	// case OARRAYLIT, OSLICELIT, OMAPLIT:
	// 	unreachable - mapped to case OCOMPLIT below by exporter

	case OCOMPLIT:
		n := nodl(p.pos(), OCOMPLIT, nil, typenod(p.typ()))
		n.List.Set(p.exprList())
		return n

	case OKEY:
		pos := p.pos()
		left, right := p.exprsOrNil()
		return nodl(pos, OKEY, left, right)

	// case OSTRUCTKEY:
	//	unreachable - handled in case OSTRUCTLIT by elemList

	// case OCALLPART:
	//	unimplemented

	// case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
	// 	unreachable - mapped to case OXDOT below by exporter

	case OXDOT:
		// see parser.new_dotname
		return npos(p.pos(), nodSym(OXDOT, p.expr(), p.fieldSym()))

	// case ODOTTYPE, ODOTTYPE2:
	// 	unreachable - mapped to case ODOTTYPE below by exporter

	case ODOTTYPE:
		n := nodl(p.pos(), ODOTTYPE, p.expr(), nil)
		n.Type = p.typ()
		return n

	// case OINDEX, OINDEXMAP, OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
	// 	unreachable - mapped to cases below by exporter

	case OINDEX:
		return nodl(p.pos(), op, p.expr(), p.expr())

	case OSLICE, OSLICE3:
		n := nodl(p.pos(), op, p.expr(), nil)
		low, high := p.exprsOrNil()
		var max *Node
		if n.Op.IsSlice3() {
			max = p.expr()
		}
		n.SetSliceBounds(low, high, max)
		return n

	// case OCONV, OCONVIFACE, OCONVNOP, OARRAYBYTESTR, OARRAYRUNESTR, OSTRARRAYBYTE, OSTRARRAYRUNE, ORUNESTR:
	// 	unreachable - mapped to OCONV case below by exporter

	case OCONV:
		n := nodl(p.pos(), OCONV, p.expr(), nil)
		n.Type = p.typ()
		return n

	case OCOPY, OCOMPLEX, OREAL, OIMAG, OAPPEND, OCAP, OCLOSE, ODELETE, OLEN, OMAKE, ONEW, OPANIC, ORECOVER, OPRINT, OPRINTN:
		n := npos(p.pos(), builtinCall(op))
		n.List.Set(p.exprList())
		if op == OAPPEND {
			n.SetIsddd(p.bool())
		}
		return n

	// case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER, OGETG:
	// 	unreachable - mapped to OCALL case below by exporter

	case OCALL:
		n := nodl(p.pos(), OCALL, p.expr(), nil)
		n.List.Set(p.exprList())
		n.SetIsddd(p.bool())
		return n

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		n := npos(p.pos(), builtinCall(OMAKE))
		n.List.Append(typenod(p.typ()))
		n.List.Append(p.exprList()...)
		return n

	// unary expressions
	case OPLUS, OMINUS, OADDR, OCOM, OIND, ONOT, ORECV:
		return nodl(p.pos(), op, p.expr(), nil)

	// binary expressions
	case OADD, OAND, OANDAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE, OLT,
		OLSH, OMOD, OMUL, ONE, OOR, OOROR, ORSH, OSEND, OSUB, OXOR:
		return nodl(p.pos(), op, p.expr(), p.expr())

	case OADDSTR:
		pos := p.pos()
		list := p.exprList()
		x := npos(pos, list[0])
		for _, y := range list[1:] {
			x = nodl(pos, OADD, x, y)
		}
		return x

	// case OCMPSTR, OCMPIFACE:
	// 	unreachable - mapped to std comparison operators by exporter

	case ODCLCONST:
		// TODO(gri) these should not be exported in the first place
		return nodl(p.pos(), OEMPTY, nil, nil)

	// --------------------------------------------------------------------
	// statements
	case ODCL:
		if p.version < 2 {
			// versions 0 and 1 exported a bool here but it
			// was always false - simply ignore in this case
			p.bool()
		}
		pos := p.pos()
		lhs := dclname(p.sym())
		typ := typenod(p.typ())
		return npos(pos, liststmt(variter([]*Node{lhs}, typ, nil))) // TODO(gri) avoid list creation

	// case ODCLFIELD:
	//	unimplemented

	// case OAS, OASWB:
	// 	unreachable - mapped to OAS case below by exporter

	case OAS:
		return nodl(p.pos(), OAS, p.expr(), p.expr())

	case OASOP:
		n := nodl(p.pos(), OASOP, nil, nil)
		n.Etype = types.EType(p.int())
		n.Left = p.expr()
		if !p.bool() {
			n.Right = nodintconst(1)
			n.SetImplicit(true)
		} else {
			n.Right = p.expr()
		}
		return n

	// case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
	// 	unreachable - mapped to OAS2 case below by exporter

	case OAS2:
		n := nodl(p.pos(), OAS2, nil, nil)
		n.List.Set(p.exprList())
		n.Rlist.Set(p.exprList())
		return n

	case ORETURN:
		n := nodl(p.pos(), ORETURN, nil, nil)
		n.List.Set(p.exprList())
		return n

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampolin routines (not exported)

	case OPROC, ODEFER:
		return nodl(p.pos(), op, p.expr(), nil)

	case OIF:
		types.Markdcl()
		n := nodl(p.pos(), OIF, nil, nil)
		n.Ninit.Set(p.stmtList())
		n.Left = p.expr()
		n.Nbody.Set(p.stmtList())
		n.Rlist.Set(p.stmtList())
		types.Popdcl()
		return n

	case OFOR:
		types.Markdcl()
		n := nodl(p.pos(), OFOR, nil, nil)
		n.Ninit.Set(p.stmtList())
		n.Left, n.Right = p.exprsOrNil()
		n.Nbody.Set(p.stmtList())
		types.Popdcl()
		return n

	case ORANGE:
		types.Markdcl()
		n := nodl(p.pos(), ORANGE, nil, nil)
		n.List.Set(p.stmtList())
		n.Right = p.expr()
		n.Nbody.Set(p.stmtList())
		types.Popdcl()
		return n

	case OSELECT, OSWITCH:
		types.Markdcl()
		n := nodl(p.pos(), op, nil, nil)
		n.Ninit.Set(p.stmtList())
		n.Left, _ = p.exprsOrNil()
		n.List.Set(p.stmtList())
		types.Popdcl()
		return n

	// case OCASE, OXCASE:
	// 	unreachable - mapped to OXCASE case below by exporter

	case OXCASE:
		types.Markdcl()
		n := nodl(p.pos(), OXCASE, nil, nil)
		n.Xoffset = int64(types.Block)
		n.List.Set(p.exprList())
		// TODO(gri) eventually we must declare variables for type switch
		// statements (type switch statements are not yet exported)
		n.Nbody.Set(p.stmtList())
		types.Popdcl()
		return n

	// case OFALL:
	// 	unreachable - mapped to OXFALL case below by exporter

	case OXFALL:
		n := nodl(p.pos(), OXFALL, nil, nil)
		n.Xoffset = int64(types.Block)
		return n

	case OBREAK, OCONTINUE:
		pos := p.pos()
		left, _ := p.exprsOrNil()
		if left != nil {
			left = newname(left.Sym)
		}
		return nodl(pos, op, left, nil)

	// case OEMPTY:
	// 	unreachable - not emitted by exporter

	case OGOTO, OLABEL:
		return nodl(p.pos(), op, newname(p.expr().Sym), nil)

	case OEND:
		return nil

	default:
		Fatalf("cannot import %v (%d) node\n"+
			"==> please file an issue and assign to gri@\n", op, int(op))
		panic("unreachable") // satisfy compiler
	}
}

func builtinCall(op Op) *Node {
	return nod(OCALL, mkname(builtinpkg.Lookup(goopnames[op])), nil)
}

func (p *importer) exprsOrNil() (a, b *Node) {
	ab := p.int()
	if ab&1 != 0 {
		a = p.expr()
	}
	if ab&2 != 0 {
		b = p.expr()
	}
	return
}

func (p *importer) fieldSym() *types.Sym {
	name := p.string()
	pkg := localpkg
	if !exportname(name) {
		pkg = p.pkg()
	}
	return pkg.Lookup(name)
}

func (p *importer) sym() *types.Sym {
	name := p.string()
	pkg := localpkg
	if name != "_" {
		pkg = p.pkg()
	}
	linkname := p.string()
	sym := pkg.Lookup(name)
	sym.Linkname = linkname
	return sym
}

func (p *importer) bool() bool {
	return p.int() != 0
}

func (p *importer) op() Op {
	return Op(p.int())
}

// ----------------------------------------------------------------------------
// Low-level decoders

func (p *importer) tagOrIndex() int {
	if p.debugFormat {
		p.marker('t')
	}

	return int(p.rawInt64())
}

func (p *importer) int() int {
	x := p.int64()
	if int64(int(x)) != x {
		p.formatErrorf("exported integer too large")
	}
	return int(x)
}

func (p *importer) int64() int64 {
	if p.debugFormat {
		p.marker('i')
	}

	return p.rawInt64()
}

func (p *importer) string() string {
	if p.debugFormat {
		p.marker('s')
	}
	// if the string was seen before, i is its index (>= 0)
	// (the empty string is at index 0)
	i := p.rawInt64()
	if i >= 0 {
		return p.strList[i]
	}
	// otherwise, i is the negative string length (< 0)
	if n := int(-i); n <= cap(p.buf) {
		p.buf = p.buf[:n]
	} else {
		p.buf = make([]byte, n)
	}
	for i := range p.buf {
		p.buf[i] = p.rawByte()
	}
	s := string(p.buf)
	p.strList = append(p.strList, s)
	return s
}

func (p *importer) marker(want byte) {
	if got := p.rawByte(); got != want {
		p.formatErrorf("incorrect marker: got %c; want %c (pos = %d)", got, want, p.read)
	}

	pos := p.read
	if n := int(p.rawInt64()); n != pos {
		p.formatErrorf("incorrect position: got %d; want %d", n, pos)
	}
}

// rawInt64 should only be used by low-level decoders.
func (p *importer) rawInt64() int64 {
	i, err := binary.ReadVarint(p)
	if err != nil {
		p.formatErrorf("read error: %v", err)
	}
	return i
}

// rawStringln should only be used to read the initial version string.
func (p *importer) rawStringln(b byte) string {
	p.buf = p.buf[:0]
	for b != '\n' {
		p.buf = append(p.buf, b)
		b = p.rawByte()
	}
	return string(p.buf)
}

// needed for binary.ReadVarint in rawInt64
func (p *importer) ReadByte() (byte, error) {
	return p.rawByte(), nil
}

// rawByte is the bottleneck interface for reading from p.in.
// It unescapes '|' 'S' to '$' and '|' '|' to '|'.
// rawByte should only be used by low-level decoders.
func (p *importer) rawByte() byte {
	c, err := p.in.ReadByte()
	p.read++
	if err != nil {
		p.formatErrorf("read error: %v", err)
	}
	if c == '|' {
		c, err = p.in.ReadByte()
		p.read++
		if err != nil {
			p.formatErrorf("read error: %v", err)
		}
		switch c {
		case 'S':
			c = '$'
		case '|':
			// nothing to do
		default:
			p.formatErrorf("unexpected escape sequence in export data")
		}
	}
	return c
}
