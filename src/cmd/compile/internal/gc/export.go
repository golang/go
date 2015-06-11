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
			Yyerror("export/package mismatch: %v", n.Sym)
		}
		return
	}

	n.Sym.Flags |= SymExport

	if Debug['E'] != 0 {
		fmt.Printf("export symbol %v\n", n.Sym)
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

func autoexport(n *Node, ctxt uint8) {
	if n == nil || n.Sym == nil {
		return
	}
	if (ctxt != PEXTERN && ctxt != PFUNC) || dclcontext != PEXTERN {
		return
	}
	if n.Name.Param != nil && n.Name.Param.Ntype != nil && n.Name.Param.Ntype.Op == OTFUNC && n.Name.Param.Ntype.Left != nil { // method
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
	fmt.Fprintf(bout, "\timport %s %q%s\n", p.Name, p.Path, suffix)
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

		case PEXTERN:
			if n.Sym != nil && !exportedsym(n.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport name %v\n", n.Sym)
				}
				exportlist = list(exportlist, n)
			}
		}

		// Local variables in the bodies need their type.
	case ODCL:
		t := n.Left.Type

		if t != Types[t.Etype] && t != idealbool && t != idealstring {
			if Isptr[t.Etype] {
				t = t.Type
			}
			if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport type %v from declaration\n", t.Sym)
				}
				exportlist = list(exportlist, t.Sym.Def)
			}
		}

	case OLITERAL:
		t := n.Type
		if t != Types[n.Type.Etype] && t != idealbool && t != idealstring {
			if Isptr[t.Etype] {
				t = t.Type
			}
			if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport literal type %v\n", t.Sym)
				}
				exportlist = list(exportlist, t.Sym.Def)
			}
		}
		fallthrough

	case OTYPE:
		if n.Sym != nil && !exportedsym(n.Sym) {
			if Debug['E'] != 0 {
				fmt.Printf("reexport literal/type %v\n", n.Sym)
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
				fmt.Printf("reexport type for expression %v\n", t.Sym)
			}
			exportlist = list(exportlist, t.Sym.Def)
		}
	}

	reexportdep(n.Left)
	reexportdep(n.Right)
	reexportdeplist(n.List)
	reexportdeplist(n.Rlist)
	reexportdeplist(n.Ninit)
	reexportdeplist(n.Nbody)
}

func dumpexportconst(s *Sym) {
	n := s.Def
	typecheck(&n, Erv)
	if n == nil || n.Op != OLITERAL {
		Fatal("dumpexportconst: oconst nil: %v", s)
	}

	t := n.Type // may or may not be specified
	dumpexporttype(t)

	if t != nil && !isideal(t) {
		fmt.Fprintf(bout, "\tconst %v %v = %v\n", Sconv(s, obj.FmtSharp), Tconv(t, obj.FmtSharp), Vconv(n.Val(), obj.FmtSharp))
	} else {
		fmt.Fprintf(bout, "\tconst %v = %v\n", Sconv(s, obj.FmtSharp), Vconv(n.Val(), obj.FmtSharp))
	}
}

func dumpexportvar(s *Sym) {
	n := s.Def
	typecheck(&n, Erv|Ecall)
	if n == nil || n.Type == nil {
		Yyerror("variable exported but not defined: %v", s)
		return
	}

	t := n.Type
	dumpexporttype(t)

	if t.Etype == TFUNC && n.Class == PFUNC {
		if n.Func != nil && n.Func.Inl != nil {
			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if Debug['l'] < 2 {
				typecheckinl(n)
			}

			// NOTE: The space after %#S here is necessary for ld's export data parser.
			fmt.Fprintf(bout, "\tfunc %v %v { %v }\n", Sconv(s, obj.FmtSharp), Tconv(t, obj.FmtShort|obj.FmtSharp), Hconv(n.Func.Inl, obj.FmtSharp))

			reexportdeplist(n.Func.Inl)
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
		if f.Type.Nname != nil && f.Type.Nname.Func.Inl != nil { // nname was set by caninl

			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if Debug['l'] < 2 {
				typecheckinl(f.Type.Nname)
			}
			fmt.Fprintf(bout, "\tfunc (%v) %v %v { %v }\n", Tconv(getthisx(f.Type).Type, obj.FmtSharp), Sconv(f.Sym, obj.FmtShort|obj.FmtByte|obj.FmtSharp), Tconv(f.Type, obj.FmtShort|obj.FmtSharp), Hconv(f.Type.Nname.Func.Inl, obj.FmtSharp))
			reexportdeplist(f.Type.Nname.Func.Inl)
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
		Yyerror("unknown export symbol: %v", s)
		return
	}

	//	print("dumpsym %O %+S\n", s->def->op, s);
	dumppkg(s.Pkg)

	switch s.Def.Op {
	default:
		Yyerror("unexpected export symbol: %v %v", Oconv(int(s.Def.Op), 0), s)

	case OLITERAL:
		dumpexportconst(s)

	case OTYPE:
		if s.Def.Type.Etype == TFORW {
			Yyerror("export of incomplete type %v", s)
		} else {
			dumpexporttype(s.Def.Type)
		}

	case ONAME:
		dumpexportvar(s)
	}
}

func dumpexport() {
	lno := lineno

	if buildid != "" {
		fmt.Fprintf(bout, "build id %q\n", buildid)
	}
	fmt.Fprintf(bout, "\n$$\npackage %s", localpkg.Name)
	if safemode != 0 {
		fmt.Fprintf(bout, " safe")
	}
	fmt.Fprintf(bout, "\n")

	for _, p := range pkgs {
		if p.Direct != 0 {
			dumppkg(p)
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
		pkgstr := fmt.Sprintf("during import %q", importpkg.Path)
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
		s.Def.Name = new(Name)
	}

	if s.Def.Type == nil {
		Yyerror("pkgtype %v", s)
	}
	return s.Def.Type
}

var numImport = make(map[string]int)

func importimport(s *Sym, path string) {
	// Informational: record package name
	// associated with import path, for use in
	// human-readable messages.

	if isbadimport(path) {
		errorexit()
	}
	p := mkpkg(path)
	if p.Name == "" {
		p.Name = s.Name
		numImport[s.Name]++
	} else if p.Name != s.Name {
		Yyerror("conflicting names %s and %s for package %q", p.Name, s.Name, p.Path)
	}

	if incannedimport == 0 && myimportpath != "" && path == myimportpath {
		Yyerror("import %q: package depends on %q (import cycle)", importpkg.Path, path)
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
		fmt.Printf("import const %v\n", s)
	}
}

func importvar(s *Sym, t *Type) {
	importsym(s, ONAME)
	if s.Def != nil && s.Def.Op == ONAME {
		if Eqtype(t, s.Def.Type) {
			return
		}
		Yyerror("inconsistent definition for var %v during import\n\t%v (in %q)\n\t%v (in %q)", s, s.Def.Type, s.Importdef.Path, t, importpkg.Path)
	}

	n := newname(s)
	s.Importdef = importpkg
	n.Type = t
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import var %v %v\n", s, Tconv(t, obj.FmtLong))
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
		Yyerror("inconsistent definition for type %v during import\n\t%v (in %q)\n\t%v (in %q)", pt.Sym, Tconv(pt, obj.FmtLong), pt.Sym.Importdef.Path, Tconv(t, obj.FmtLong), importpkg.Path)
	}

	if Debug['E'] != 0 {
		fmt.Printf("import type %v %v\n", pt, Tconv(t, obj.FmtLong))
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
			fmt.Fprintf(b, "#define const_%s %v\n", n.Sym.Name, Vconv(n.Val(), obj.FmtSharp))

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
