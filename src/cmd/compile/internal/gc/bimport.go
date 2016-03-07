// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Binary package import.
// Based loosely on x/tools/go/importer.

package gc

import (
	"cmd/compile/internal/big"
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
)

// The overall structure of Import is symmetric to Export: For each
// export method in bexport.go there is a matching and symmetric method
// in bimport.go. Changing the export format requires making symmetric
// changes to bimport.go and bexport.go.

// Import populates importpkg from the serialized package data.
func Import(in *obj.Biobuf) {
	p := importer{in: in}
	p.buf = p.bufarray[:]

	// read low-level encoding format
	switch format := p.byte(); format {
	case 'c':
		// compact format - nothing to do
	case 'd':
		p.debugFormat = true
	default:
		Fatalf("importer: invalid encoding format in export data: got %q; want 'c' or 'd'", format)
	}

	// --- generic export data ---

	if v := p.string(); v != exportVersion {
		Fatalf("importer: unknown export data version: %s", v)
	}

	// populate typList with predeclared "known" types
	p.typList = append(p.typList, predeclared()...)

	// read package data
	p.pkg()
	if p.pkgList[0] != importpkg {
		Fatalf("importer: imported package not found in pkgList[0]")
	}

	// read compiler-specific flags
	importpkg.Safe = p.string() == "safe"

	// defer some type-checking until all types are read in completely
	// (go.y:import_there)
	tcok := typecheckok
	typecheckok = true
	defercheckwidth()

	// read consts
	for i := p.int(); i > 0; i-- {
		sym := p.localname()
		typ := p.typ()
		val := p.value(typ)
		importconst(sym, idealType(typ), nodlit(val))
	}

	// read vars
	for i := p.int(); i > 0; i-- {
		sym := p.localname()
		typ := p.typ()
		importvar(sym, typ)
	}

	// read funcs
	for i := p.int(); i > 0; i-- {
		// go.y:hidden_fndcl
		sym := p.localname()
		typ := p.typ()
		inl := p.int()

		importsym(sym, ONAME)
		if sym.Def != nil && sym.Def.Op == ONAME && !Eqtype(typ, sym.Def.Type) {
			Fatalf("importer: inconsistent definition for func %v during import\n\t%v\n\t%v", sym, sym.Def.Type, typ)
		}

		n := newfuncname(sym)
		n.Type = typ
		declare(n, PFUNC)
		funchdr(n)

		// go.y:hidden_import
		n.Func.Inl.Set(nil)
		if inl >= 0 {
			if inl != len(p.inlined) {
				panic("inlined body list inconsistent")
			}
			p.inlined = append(p.inlined, n.Func)
		}
		funcbody(n)
		importlist = append(importlist, n) // TODO(gri) do this only if body is inlineable?
	}

	// read types
	for i := p.int(); i > 0; i-- {
		// name is parsed as part of named type
		p.typ()
	}

	// --- compiler-specific export data ---

	// read inlined functions bodies
	n := p.int()
	for i := 0; i < n; i++ {
		body := p.nodeList()
		const hookup = false // TODO(gri) enable and remove this condition
		if hookup {
			p.inlined[i].Inl.Set(body)
		}
	}

	// --- end of export data ---

	typecheckok = tcok
	resumecheckwidth()

	testdclstack() // debugging only
}

func idealType(typ *Type) *Type {
	if isideal(typ) {
		// canonicalize ideal types
		typ = Types[TIDEAL]
	}
	return typ
}

type importer struct {
	in       *obj.Biobuf
	buf      []byte   // for reading strings
	bufarray [64]byte // initial underlying array for buf, large enough to avoid allocation when compiling std lib
	pkgList  []*Pkg
	typList  []*Type
	inlined  []*Func

	debugFormat bool
	read        int // bytes read
}

func (p *importer) pkg() *Pkg {
	// if the package was seen before, i is its index (>= 0)
	i := p.tagOrIndex()
	if i >= 0 {
		return p.pkgList[i]
	}

	// otherwise, i is the package tag (< 0)
	if i != packageTag {
		Fatalf("importer: expected package tag, found tag = %d", i)
	}

	// read package data
	name := p.string()
	path := p.string()

	// we should never see an empty package name
	if name == "" {
		Fatalf("importer: empty package name in import")
	}

	// we should never see a bad import path
	if isbadimport(path) {
		Fatalf("importer: bad path in import: %q", path)
	}

	// an empty path denotes the package we are currently importing
	pkg := importpkg
	if path != "" {
		pkg = mkpkg(path)
	}
	if pkg.Name == "" {
		pkg.Name = name
	} else if pkg.Name != name {
		Fatalf("importer: inconsistent package names: got %s; want %s (path = %s)", pkg.Name, name, path)
	}
	p.pkgList = append(p.pkgList, pkg)

	return pkg
}

func (p *importer) localname() *Sym {
	// go.y:hidden_importsym
	name := p.string()
	if name == "" {
		Fatalf("importer: unexpected anonymous name")
	}
	structpkg = importpkg // go.y:hidden_pkg_importsym
	return importpkg.Lookup(name)
}

func (p *importer) newtyp(etype EType) *Type {
	t := typ(etype)
	p.typList = append(p.typList, t)
	return t
}

func (p *importer) typ() *Type {
	// if the type was seen before, i is its index (>= 0)
	i := p.tagOrIndex()
	if i >= 0 {
		return p.typList[i]
	}

	// otherwise, i is the type tag (< 0)
	var t *Type
	switch i {
	case namedTag:
		// go.y:hidden_importsym
		tsym := p.qualifiedName()

		// go.y:hidden_pkgtype
		t = pkgtype(tsym)
		importsym(tsym, OTYPE)
		p.typList = append(p.typList, t)

		// read underlying type
		// go.y:hidden_type
		t0 := p.typ()
		importtype(t, t0) // go.y:hidden_import

		// interfaces don't have associated methods
		if t0.Etype == TINTER {
			break
		}

		// read associated methods
		for i := p.int(); i > 0; i-- {
			// go.y:hidden_fndcl
			name := p.string()
			recv := p.paramList() // TODO(gri) do we need a full param list for the receiver?
			params := p.paramList()
			result := p.paramList()
			inl := p.int()

			pkg := localpkg
			if !exportname(name) {
				pkg = tsym.Pkg
			}
			sym := pkg.Lookup(name)

			n := methodname1(newname(sym), recv.N.Right)
			n.Type = functype(recv.N, params, result)
			checkwidth(n.Type)
			// addmethod uses the global variable structpkg to verify consistency
			{
				saved := structpkg
				structpkg = tsym.Pkg
				addmethod(sym, n.Type, false, false)
				structpkg = saved
			}
			funchdr(n)

			// (comment from go.y)
			// inl.C's inlnode in on a dotmeth node expects to find the inlineable body as
			// (dotmeth's type).Nname.Inl, and dotmeth's type has been pulled
			// out by typecheck's lookdot as this $$.ttype. So by providing
			// this back link here we avoid special casing there.
			n.Type.Nname = n

			// go.y:hidden_import
			n.Func.Inl.Set(nil)
			if inl >= 0 {
				if inl != len(p.inlined) {
					panic("inlined body list inconsistent")
				}
				p.inlined = append(p.inlined, n.Func)
			}
			funcbody(n)
			importlist = append(importlist, n) // TODO(gri) do this only if body is inlineable?
		}

	case arrayTag, sliceTag:
		t = p.newtyp(TARRAY)
		t.Bound = -1
		if i == arrayTag {
			t.Bound = p.int64()
		}
		t.Type = p.typ()

	case dddTag:
		t = p.newtyp(T_old_DARRAY)
		t.Bound = -1
		t.Type = p.typ()

	case structTag:
		t = p.newtyp(TSTRUCT)
		tostruct0(t, p.fieldList())

	case pointerTag:
		t = p.newtyp(Tptr)
		t.Type = p.typ()

	case signatureTag:
		t = p.newtyp(TFUNC)
		params := p.paramList()
		result := p.paramList()
		functype0(t, nil, params, result)

	case interfaceTag:
		t = p.newtyp(TINTER)
		if p.int() != 0 {
			Fatalf("importer: unexpected embedded interface")
		}
		tointerface0(t, p.methodList())

	case mapTag:
		t = p.newtyp(TMAP)
		t.Down = p.typ() // key
		t.Type = p.typ() // val

	case chanTag:
		t = p.newtyp(TCHAN)
		t.Chan = uint8(p.int())
		t.Type = p.typ()

	default:
		Fatalf("importer: unexpected type (tag = %d)", i)
	}

	if t == nil {
		Fatalf("importer: nil type (type tag = %d)", i)
	}

	return t
}

func (p *importer) qualifiedName() *Sym {
	name := p.string()
	pkg := p.pkg()
	return pkg.Lookup(name)
}

// go.y:hidden_structdcl_list
func (p *importer) fieldList() *NodeList {
	i := p.int()
	if i == 0 {
		return nil
	}
	n := list1(p.field())
	for i--; i > 0; i-- {
		n = list(n, p.field())
	}
	return n
}

// go.y:hidden_structdcl
func (p *importer) field() *Node {
	sym := p.fieldName()
	typ := p.typ()
	note := p.note()

	var n *Node
	if sym.Name != "" {
		n = Nod(ODCLFIELD, newname(sym), typenod(typ))
	} else {
		// anonymous field - typ must be T or *T and T must be a type name
		s := typ.Sym
		if s == nil && Isptr[typ.Etype] {
			s = typ.Type.Sym // deref
		}
		pkg := importpkg
		if sym != nil {
			pkg = sym.Pkg
		}
		n = embedded(s, pkg)
		n.Right = typenod(typ)
	}
	n.SetVal(note)

	return n
}

func (p *importer) note() (v Val) {
	if s := p.string(); s != "" {
		v.U = s
	}
	return
}

// go.y:hidden_interfacedcl_list
func (p *importer) methodList() *NodeList {
	i := p.int()
	if i == 0 {
		return nil
	}
	n := list1(p.method())
	for i--; i > 0; i-- {
		n = list(n, p.method())
	}
	return n
}

// go.y:hidden_interfacedcl
func (p *importer) method() *Node {
	sym := p.fieldName()
	params := p.paramList()
	result := p.paramList()
	return Nod(ODCLFIELD, newname(sym), typenod(functype(fakethis(), params, result)))
}

// go.y:sym,hidden_importsym
func (p *importer) fieldName() *Sym {
	name := p.string()
	pkg := localpkg
	if name == "_" {
		// During imports, unqualified non-exported identifiers are from builtinpkg
		// (see go.y:sym). The binary exporter only exports blank as a non-exported
		// identifier without qualification.
		pkg = builtinpkg
	} else if name == "?" || name != "" && !exportname(name) {
		if name == "?" {
			name = ""
		}
		pkg = p.pkg()
	}
	return pkg.Lookup(name)
}

// go.y:ohidden_funarg_list
func (p *importer) paramList() *NodeList {
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
	n := list1(p.param(named))
	i--
	for ; i > 0; i-- {
		n = list(n, p.param(named))
	}
	return n
}

// go.y:hidden_funarg
func (p *importer) param(named bool) *Node {
	typ := p.typ()

	isddd := false
	if typ.Etype == T_old_DARRAY {
		// T_old_DARRAY indicates ... type
		typ.Etype = TARRAY
		isddd = true
	}

	n := Nod(ODCLFIELD, nil, typenod(typ))
	n.Isddd = isddd

	if named {
		name := p.string()
		if name == "" {
			Fatalf("importer: expected named parameter")
		}
		// The parameter package doesn't matter; it's never consulted.
		// We use the builtinpkg per go.y:sym (line 1181).
		n.Left = newname(builtinpkg.Lookup(name))
	}

	// TODO(gri) This is compiler-specific (escape info).
	// Move into compiler-specific section eventually?
	n.SetVal(p.note())

	return n
}

func (p *importer) value(typ *Type) (x Val) {
	switch tag := p.tagOrIndex(); tag {
	case falseTag:
		x.U = false

	case trueTag:
		x.U = true

	case int64Tag:
		u := new(Mpint)
		Mpmovecfix(u, p.int64())
		u.Rune = typ == idealrune
		x.U = u

	case floatTag:
		f := newMpflt()
		p.float(f)
		if typ == idealint || Isint[typ.Etype] {
			// uncommon case: large int encoded as float
			u := new(Mpint)
			mpmovefltfix(u, f)
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

	case nilTag:
		x.U = new(NilVal)

	default:
		Fatalf("importer: unexpected value tag %d", tag)
	}

	// verify ideal type
	if isideal(typ) && untype(x.Ctype()) != typ {
		Fatalf("importer: value %v and type %v don't match", x, typ)
	}

	return
}

func (p *importer) float(x *Mpflt) {
	sign := p.int()
	if sign == 0 {
		Mpmovecflt(x, 0)
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

// go.y:stmt_list
func (p *importer) nodeList() []*Node {
	c := p.int()
	s := make([]*Node, 0, c)
	for i := 0; i < c; i++ {
		s = append(s, p.node())
	}
	return s
}

func (p *importer) node() *Node {
	// TODO(gri) eventually we may need to allocate in each branch
	n := Nod(p.op(), nil, nil)

	switch n.Op {
	// names
	case ONAME, OPACK, ONONAME:
		name := mkname(p.sym())
		// TODO(gri) decide what to do here (this code throws away n)
		/*
			if name.Op != n.Op {
				Fatalf("importer: got node op = %s; want %s", opnames[name.Op], opnames[n.Op])
			}
		*/
		n = name

	case OTYPE:
		if p.bool() {
			n.Sym = p.sym()
		} else {
			n.Type = p.typ()
		}

	case OLITERAL:
		typ := p.typ()
		n.Type = idealType(typ)
		n.SetVal(p.value(typ))

	// expressions
	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if p.bool() {
			setNodeSeq(&n.List, p.nodeList())
		}
		n.Left, n.Right = p.nodesOrNil()
		n.Type = p.typ()

	case OPLUS, OMINUS, OADDR, OCOM, OIND, ONOT, ORECV:
		n.Left = p.node()

	case OADD, OAND, OANDAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE, OLT,
		OLSH, OMOD, OMUL, ONE, OOR, OOROR, ORSH, OSEND,
		OSUB, OXOR:
		n.Left = p.node()
		n.Right = p.node()

	case OADDSTR:
		setNodeSeq(&n.List, p.nodeList())

	case OPTRLIT:
		n.Left = p.node()

	case OSTRUCTLIT:
		n.Type = p.typ()
		setNodeSeq(&n.List, p.nodeList())
		n.Implicit = p.bool()

	case OARRAYLIT, OMAPLIT:
		n.Type = p.typ()
		setNodeSeq(&n.List, p.nodeList())
		n.Implicit = p.bool()

	case OKEY:
		n.Left, n.Right = p.nodesOrNil()

	case OCOPY, OCOMPLEX:
		n.Left = p.node()
		n.Right = p.node()

	case OCONV, OCONVIFACE, OCONVNOP, OARRAYBYTESTR, OARRAYRUNESTR, OSTRARRAYBYTE, OSTRARRAYRUNE, ORUNESTR:
		// n.Type = p.typ()
		// if p.bool() {
		// 	n.Left = p.node()
		// } else {
		// 	setNodeSeq(&n.List, p.nodeList())
		// }
		x := Nod(OCALL, p.typ().Nod, nil)
		if p.bool() {
			setNodeSeq(&x.List, []*Node{p.node()})
		} else {
			setNodeSeq(&x.List, p.nodeList())
		}
		return x

	case ODOT, ODOTPTR, ODOTMETH, ODOTINTER, OXDOT:
		// see parser.new_dotname
		obj := p.node()
		sel := p.sym()
		if obj.Op == OPACK {
			s := restrictlookup(sel.Name, obj.Name.Pkg)
			obj.Used = true
			return oldname(s)
		}
		return Nod(OXDOT, obj, newname(sel))

	case ODOTTYPE, ODOTTYPE2:
		n.Left = p.node()
		if p.bool() {
			n.Right = p.node()
		} else {
			n.Type = p.typ()
		}

	case OINDEX, OINDEXMAP, OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		n.Left = p.node()
		n.Right = p.node()

	case OREAL, OIMAG, OAPPEND, OCAP, OCLOSE, ODELETE, OLEN, OMAKE, ONEW, OPANIC,
		ORECOVER, OPRINT, OPRINTN:
		n.Left, _ = p.nodesOrNil()
		setNodeSeq(&n.List, p.nodeList())
		n.Isddd = p.bool()

	case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER, OGETG:
		n.Left = p.node()
		setNodeSeq(&n.List, p.nodeList())
		n.Isddd = p.bool()

	case OCMPSTR, OCMPIFACE:
		n.Left = p.node()
		n.Right = p.node()
		n.Etype = EType(p.int())

	case OPAREN:
		n.Left = p.node()

	// statements
	case ODCL:
		n.Left = p.node() // TODO(gri) compare with fmt code
		n.Left.Type = p.typ()

	case OAS:
		n.Left, n.Right = p.nodesOrNil()
		n.Colas = p.bool() // TODO(gri) what about complexinit?

	case OASOP:
		n.Left = p.node()
		n.Right = p.node()
		n.Etype = EType(p.int())

	case OAS2, OASWB:
		setNodeSeq(&n.List, p.nodeList())
		setNodeSeq(&n.Rlist, p.nodeList())

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		setNodeSeq(&n.List, p.nodeList())
		setNodeSeq(&n.Rlist, p.nodeList())

	case ORETURN:
		setNodeSeq(&n.List, p.nodeList())

	case OPROC, ODEFER:
		n.Left = p.node()

	case OIF:
		setNodeSeq(&n.Ninit, p.nodeList())
		n.Left = p.node()
		n.Nbody.Set(p.nodeList())
		setNodeSeq(&n.Rlist, p.nodeList())

	case OFOR:
		setNodeSeq(&n.Ninit, p.nodeList())
		n.Left, n.Right = p.nodesOrNil()
		n.Nbody.Set(p.nodeList())

	case ORANGE:
		if p.bool() {
			setNodeSeq(&n.List, p.nodeList())
		}
		n.Right = p.node()
		n.Nbody.Set(p.nodeList())

	case OSELECT, OSWITCH:
		setNodeSeq(&n.Ninit, p.nodeList())
		n.Left, _ = p.nodesOrNil()
		setNodeSeq(&n.List, p.nodeList())

	case OCASE, OXCASE:
		if p.bool() {
			setNodeSeq(&n.List, p.nodeList())
		}
		n.Nbody.Set(p.nodeList())

	case OBREAK, OCONTINUE, OGOTO, OFALL, OXFALL:
		n.Left, _ = p.nodesOrNil()

	case OEMPTY:
		// nothing to do

	case OLABEL:
		n.Left = p.node()

	default:
		panic(fmt.Sprintf("importer: %s (%d) node not yet supported", opnames[n.Op], n.Op))
	}

	return n
}

func (p *importer) nodesOrNil() (a, b *Node) {
	ab := p.int()
	if ab&1 != 0 {
		a = p.node()
	}
	if ab&2 != 0 {
		b = p.node()
	}
	return
}

func (p *importer) sym() *Sym {
	return p.fieldName()
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
		Fatalf("importer: exported integer too large")
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

	if n := int(p.rawInt64()); n > 0 {
		if cap(p.buf) < n {
			p.buf = make([]byte, n)
		} else {
			p.buf = p.buf[:n]
		}
		for i := 0; i < n; i++ {
			p.buf[i] = p.byte()
		}
		return string(p.buf)
	}

	return ""
}

func (p *importer) marker(want byte) {
	if got := p.byte(); got != want {
		Fatalf("importer: incorrect marker: got %c; want %c (pos = %d)", got, want, p.read)
	}

	pos := p.read
	if n := int(p.rawInt64()); n != pos {
		Fatalf("importer: incorrect position: got %d; want %d", n, pos)
	}
}

// rawInt64 should only be used by low-level decoders
func (p *importer) rawInt64() int64 {
	i, err := binary.ReadVarint(p)
	if err != nil {
		Fatalf("importer: read error: %v", err)
	}
	return i
}

// needed for binary.ReadVarint in rawInt64
func (p *importer) ReadByte() (byte, error) {
	return p.byte(), nil
}

// byte is the bottleneck interface for reading from p.in.
// It unescapes '|' 'S' to '$' and '|' '|' to '|'.
func (p *importer) byte() byte {
	c := obj.Bgetc(p.in)
	p.read++
	if c < 0 {
		Fatalf("importer: read error")
	}
	if c == '|' {
		c = obj.Bgetc(p.in)
		p.read++
		if c < 0 {
			Fatalf("importer: read error")
		}
		switch c {
		case 'S':
			c = '$'
		case '|':
			// nothing to do
		default:
			Fatalf("importer: unexpected escape sequence in export data")
		}
	}
	return byte(c)
}
