// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"sort"
	"unicode"
	"unicode/utf8"
)

var asmlist *NodeList

// Mark n's symbol as exported
func exportsym(n *Node) {
	if n == nil || n.Sym == nil {
		return
	}
	if n.Sym.Flags&(SymExport|SymPackage) != 0 {
		if n.Sym.Flags&SymPackage != 0 {
			Yyerror("export/package mismatch: %v", Sconv(n.Sym, 0))
		}
		return
	}

	n.Sym.Flags |= SymExport

	if Debug['E'] != 0 {
		fmt.Printf("export symbol %v\n", Sconv(n.Sym, 0))
	}
	exportlist = list(exportlist, n)
}

func exportname(s string) bool {
	if s[0] < utf8.RuneSelf {
		return 'A' <= s[0] && s[0] <= 'Z'
	}
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsUpper(r)
}

func initname(s string) bool {
	return s == "init"
}

// exportedsym reports whether a symbol will be visible
// to files that import our package.
func exportedsym(sym *Sym) bool {
	// Builtins are visible everywhere.
	if sym.Pkg == builtinpkg || sym.Origpkg == builtinpkg {
		return true
	}

	return sym.Pkg == localpkg && exportname(sym.Name)
}

func autoexport(n *Node, ctxt int) {
	if n == nil || n.Sym == nil {
		return
	}
	if (ctxt != PEXTERN && ctxt != PFUNC) || dclcontext != PEXTERN {
		return
	}
	if n.Ntype != nil && n.Ntype.Op == OTFUNC && n.Ntype.Left != nil { // method
		return
	}

	// -A is for cmd/gc/mkbuiltin script, so export everything
	if Debug['A'] != 0 || exportname(n.Sym.Name) || initname(n.Sym.Name) {
		exportsym(n)
	}
	if asmhdr != "" && n.Sym.Pkg == localpkg && n.Sym.Flags&SymAsm == 0 {
		n.Sym.Flags |= SymAsm
		asmlist = list(asmlist, n)
	}
}

func dumppkg(p *Pkg) {
	if p == nil || p == localpkg || p.Exported != 0 || p == builtinpkg {
		return
	}
	p.Exported = 1
	suffix := ""
	if p.Direct == 0 {
		suffix = " // indirect"
	}
	fmt.Fprintf(bout, "\timport %s \"%v\"%s\n", p.Name, Zconv(p.Path, 0), suffix)
}

// Look for anything we need for the inline body
func reexportdeplist(ll *NodeList) {
	for ; ll != nil; ll = ll.Next {
		reexportdep(ll.N)
	}
}

func reexportdep(n *Node) {
	if n == nil {
		return
	}

	//print("reexportdep %+hN\n", n);
	switch n.Op {
	case ONAME:
		switch n.Class &^ PHEAP {
		// methods will be printed along with their type
		// nodes for T.Method expressions
		case PFUNC:
			if n.Left != nil && n.Left.Op == OTYPE {
				break
			}

			// nodes for method calls.
			if n.Type == nil || n.Type.Thistuple > 0 {
				break
			}
			fallthrough

			// fallthrough
		case PEXTERN:
			if n.Sym != nil && !exportedsym(n.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport name %v\n", Sconv(n.Sym, 0))
				}
				exportlist = list(exportlist, n)
			}
		}

		// Local variables in the bodies need their type.
	case ODCL:
		t := n.Left.Type

		if t != Types[t.Etype] && t != idealbool && t != idealstring {
			if Isptr[t.Etype] != 0 {
				t = t.Type
			}
			if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport type %v from declaration\n", Sconv(t.Sym, 0))
				}
				exportlist = list(exportlist, t.Sym.Def)
			}
		}

	case OLITERAL:
		t := n.Type
		if t != Types[n.Type.Etype] && t != idealbool && t != idealstring {
			if Isptr[t.Etype] != 0 {
				t = t.Type
			}
			if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport literal type %v\n", Sconv(t.Sym, 0))
				}
				exportlist = list(exportlist, t.Sym.Def)
			}
		}
		fallthrough

		// fallthrough
	case OTYPE:
		if n.Sym != nil && !exportedsym(n.Sym) {
			if Debug['E'] != 0 {
				fmt.Printf("reexport literal/type %v\n", Sconv(n.Sym, 0))
			}
			exportlist = list(exportlist, n)
		}

		// for operations that need a type when rendered, put the type on the export list.
	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		ORUNESTR,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		ODOTTYPE,
		ODOTTYPE2,
		OSTRUCTLIT,
		OARRAYLIT,
		OPTRLIT,
		OMAKEMAP,
		OMAKESLICE,
		OMAKECHAN:
		t := n.Type

		if t.Sym == nil && t.Type != nil {
			t = t.Type
		}
		if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
			if Debug['E'] != 0 {
				fmt.Printf("reexport type for expression %v\n", Sconv(t.Sym, 0))
			}
			exportlist = list(exportlist, t.Sym.Def)
		}
	}

	reexportdep(n.Left)
	reexportdep(n.Right)
	reexportdeplist(n.List)
	reexportdeplist(n.Rlist)
	reexportdeplist(n.Ninit)
	reexportdep(n.Ntest)
	reexportdep(n.Nincr)
	reexportdeplist(n.Nbody)
	reexportdeplist(n.Nelse)
}

func dumpexportconst(s *Sym) {
	n := s.Def
	typecheck(&n, Erv)
	if n == nil || n.Op != OLITERAL {
		Fatal("dumpexportconst: oconst nil: %v", Sconv(s, 0))
	}

	t := n.Type // may or may not be specified
	dumpexporttype(t)

	if t != nil && !isideal(t) {
		fmt.Fprintf(bout, "\tconst %v %v = %v\n", Sconv(s, obj.FmtSharp), Tconv(t, obj.FmtSharp), Vconv(&n.Val, obj.FmtSharp))
	} else {
		fmt.Fprintf(bout, "\tconst %v = %v\n", Sconv(s, obj.FmtSharp), Vconv(&n.Val, obj.FmtSharp))
	}
}

func dumpexportvar(s *Sym) {
	n := s.Def
	typecheck(&n, Erv|Ecall)
	if n == nil || n.Type == nil {
		Yyerror("variable exported but not defined: %v", Sconv(s, 0))
		return
	}

	t := n.Type
	dumpexporttype(t)

	if t.Etype == TFUNC && n.Class == PFUNC {
		if n.Inl != nil {
			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if Debug['l'] < 2 {
				typecheckinl(n)
			}

			// NOTE: The space after %#S here is necessary for ld's export data parser.
			fmt.Fprintf(bout, "\tfunc %v %v { %v }\n", Sconv(s, obj.FmtSharp), Tconv(t, obj.FmtShort|obj.FmtSharp), Hconv(n.Inl, obj.FmtSharp))

			reexportdeplist(n.Inl)
		} else {
			fmt.Fprintf(bout, "\tfunc %v %v\n", Sconv(s, obj.FmtSharp), Tconv(t, obj.FmtShort|obj.FmtSharp))
		}
	} else {
		fmt.Fprintf(bout, "\tvar %v %v\n", Sconv(s, obj.FmtSharp), Tconv(t, obj.FmtSharp))
	}
}

type methodbyname []*Type

func (x methodbyname) Len() int {
	return len(x)
}

func (x methodbyname) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

func (x methodbyname) Less(i, j int) bool {
	a := x[i]
	b := x[j]
	return stringsCompare(a.Sym.Name, b.Sym.Name) < 0
}

func dumpexporttype(t *Type) {
	if t == nil {
		return
	}
	if t.Printed != 0 || t == Types[t.Etype] || t == bytetype || t == runetype || t == errortype {
		return
	}
	t.Printed = 1

	if t.Sym != nil && t.Etype != TFIELD {
		dumppkg(t.Sym.Pkg)
	}

	dumpexporttype(t.Type)
	dumpexporttype(t.Down)

	if t.Sym == nil || t.Etype == TFIELD {
		return
	}

	n := 0
	for f := t.Method; f != nil; f = f.Down {
		dumpexporttype(f)
		n++
	}

	m := make([]*Type, n)
	i := 0
	for f := t.Method; f != nil; f = f.Down {
		m[i] = f
		i++
	}
	sort.Sort(methodbyname(m[:n]))

	fmt.Fprintf(bout, "\ttype %v %v\n", Sconv(t.Sym, obj.FmtSharp), Tconv(t, obj.FmtSharp|obj.FmtLong))
	var f *Type
	for i := 0; i < n; i++ {
		f = m[i]
		if f.Nointerface {
			fmt.Fprintf(bout, "\t//go:nointerface\n")
		}
		if f.Type.Nname != nil && f.Type.Nname.Inl != nil { // nname was set by caninl

			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if Debug['l'] < 2 {
				typecheckinl(f.Type.Nname)
			}
			fmt.Fprintf(bout, "\tfunc (%v) %v %v { %v }\n", Tconv(getthisx(f.Type).Type, obj.FmtSharp), Sconv(f.Sym, obj.FmtShort|obj.FmtByte|obj.FmtSharp), Tconv(f.Type, obj.FmtShort|obj.FmtSharp), Hconv(f.Type.Nname.Inl, obj.FmtSharp))
			reexportdeplist(f.Type.Nname.Inl)
		} else {
			fmt.Fprintf(bout, "\tfunc (%v) %v %v\n", Tconv(getthisx(f.Type).Type, obj.FmtSharp), Sconv(f.Sym, obj.FmtShort|obj.FmtByte|obj.FmtSharp), Tconv(f.Type, obj.FmtShort|obj.FmtSharp))
		}
	}
}

func dumpsym(s *Sym) {
	if s.Flags&SymExported != 0 {
		return
	}
	s.Flags |= SymExported

	if s.Def == nil {
		Yyerror("unknown export symbol: %v", Sconv(s, 0))
		return
	}

	//	print("dumpsym %O %+S\n", s->def->op, s);
	dumppkg(s.Pkg)

	switch s.Def.Op {
	default:
		Yyerror("unexpected export symbol: %v %v", Oconv(int(s.Def.Op), 0), Sconv(s, 0))

	case OLITERAL:
		dumpexportconst(s)

	case OTYPE:
		if s.Def.Type.Etype == TFORW {
			Yyerror("export of incomplete type %v", Sconv(s, 0))
		} else {
			dumpexporttype(s.Def.Type)
		}

	case ONAME:
		dumpexportvar(s)
	}
}

func dumpexport() {
	lno := lineno

	fmt.Fprintf(bout, "\n$$\npackage %s", localpkg.Name)
	if safemode != 0 {
		fmt.Fprintf(bout, " safe")
	}
	fmt.Fprintf(bout, "\n")

	var p *Pkg
	for i := int32(0); i < int32(len(phash)); i++ {
		for p = phash[i]; p != nil; p = p.Link {
			if p.Direct != 0 {
				dumppkg(p)
			}
		}
	}

	for l := exportlist; l != nil; l = l.Next {
		lineno = l.N.Lineno
		dumpsym(l.N.Sym)
	}

	fmt.Fprintf(bout, "\n$$\n")
	lineno = lno
}

/*
 * import
 */

/*
 * return the sym for ss, which should match lexical
 */
func importsym(s *Sym, op int) *Sym {
	if s.Def != nil && int(s.Def.Op) != op {
		pkgstr := fmt.Sprintf("during import \"%v\"", Zconv(importpkg.Path, 0))
		redeclare(s, pkgstr)
	}

	// mark the symbol so it is not reexported
	if s.Def == nil {
		if exportname(s.Name) || initname(s.Name) {
			s.Flags |= SymExport
		} else {
			s.Flags |= SymPackage // package scope
		}
	}

	return s
}

/*
 * return the type pkg.name, forward declaring if needed
 */
func pkgtype(s *Sym) *Type {
	importsym(s, OTYPE)
	if s.Def == nil || s.Def.Op != OTYPE {
		t := typ(TFORW)
		t.Sym = s
		s.Def = typenod(t)
	}

	if s.Def.Type == nil {
		Yyerror("pkgtype %v", Sconv(s, 0))
	}
	return s.Def.Type
}

func importimport(s *Sym, z *Strlit) {
	// Informational: record package name
	// associated with import path, for use in
	// human-readable messages.

	if isbadimport(z) {
		errorexit()
	}
	p := mkpkg(z)
	if p.Name == "" {
		p.Name = s.Name
		Pkglookup(s.Name, nil).Npkg++
	} else if p.Name != s.Name {
		Yyerror("conflicting names %s and %s for package \"%v\"", p.Name, s.Name, Zconv(p.Path, 0))
	}

	if incannedimport == 0 && myimportpath != "" && z.S == myimportpath {
		Yyerror("import \"%v\": package depends on \"%v\" (import cycle)", Zconv(importpkg.Path, 0), Zconv(z, 0))
		errorexit()
	}
}

func importconst(s *Sym, t *Type, n *Node) {
	importsym(s, OLITERAL)
	Convlit(&n, t)

	if s.Def != nil { // TODO: check if already the same.
		return
	}

	if n.Op != OLITERAL {
		Yyerror("expression must be a constant")
		return
	}

	if n.Sym != nil {
		n1 := Nod(OXXX, nil, nil)
		*n1 = *n
		n = n1
	}

	n.Orig = newname(s)
	n.Sym = s
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import const %v\n", Sconv(s, 0))
	}
}

func importvar(s *Sym, t *Type) {
	importsym(s, ONAME)
	if s.Def != nil && s.Def.Op == ONAME {
		if Eqtype(t, s.Def.Type) {
			return
		}
		Yyerror("inconsistent definition for var %v during import\n\t%v (in \"%v\")\n\t%v (in \"%v\")", Sconv(s, 0), Tconv(s.Def.Type, 0), Zconv(s.Importdef.Path, 0), Tconv(t, 0), Zconv(importpkg.Path, 0))
	}

	n := newname(s)
	s.Importdef = importpkg
	n.Type = t
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import var %v %v\n", Sconv(s, 0), Tconv(t, obj.FmtLong))
	}
}

func importtype(pt *Type, t *Type) {
	// override declaration in unsafe.go for Pointer.
	// there is no way in Go code to define unsafe.Pointer
	// so we have to supply it.
	if incannedimport != 0 && importpkg.Name == "unsafe" && pt.Nod.Sym.Name == "Pointer" {
		t = Types[TUNSAFEPTR]
	}

	if pt.Etype == TFORW {
		n := pt.Nod
		copytype(pt.Nod, t)
		pt.Nod = n // unzero nod
		pt.Sym.Importdef = importpkg
		pt.Sym.Lastlineno = int32(parserline())
		declare(n, PEXTERN)
		checkwidth(pt)
	} else if !Eqtype(pt.Orig, t) {
		Yyerror("inconsistent definition for type %v during import\n\t%v (in \"%v\")\n\t%v (in \"%v\")", Sconv(pt.Sym, 0), Tconv(pt, obj.FmtLong), Zconv(pt.Sym.Importdef.Path, 0), Tconv(t, obj.FmtLong), Zconv(importpkg.Path, 0))
	}

	if Debug['E'] != 0 {
		fmt.Printf("import type %v %v\n", Tconv(pt, 0), Tconv(t, obj.FmtLong))
	}
}

func dumpasmhdr() {
	var b *obj.Biobuf

	b, err := obj.Bopenw(asmhdr)
	if err != nil {
		Fatal("%v", err)
	}
	fmt.Fprintf(b, "// generated by %cg -asmhdr from package %s\n\n", Thearch.Thechar, localpkg.Name)
	var n *Node
	var t *Type
	for l := asmlist; l != nil; l = l.Next {
		n = l.N
		if isblanksym(n.Sym) {
			continue
		}
		switch n.Op {
		case OLITERAL:
			fmt.Fprintf(b, "#define const_%s %v\n", n.Sym.Name, Vconv(&n.Val, obj.FmtSharp))

		case OTYPE:
			t = n.Type
			if t.Etype != TSTRUCT || t.Map != nil || t.Funarg != 0 {
				break
			}
			fmt.Fprintf(b, "#define %s__size %d\n", t.Sym.Name, int(t.Width))
			for t = t.Type; t != nil; t = t.Down {
				if !isblanksym(t.Sym) {
					fmt.Fprintf(b, "#define %s_%s %d\n", n.Sym.Name, t.Sym.Name, int(t.Width))
				}
			}
		}
	}

	obj.Bterm(b)
}
