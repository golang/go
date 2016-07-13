// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"bytes"
	"cmd/internal/bio"
	"fmt"
	"sort"
	"unicode"
	"unicode/utf8"
)

var (
	newexport    bool // if set, use new export format
	Debug_export int  // if set, print debugging information about export data
	exportsize   int
)

func exportf(format string, args ...interface{}) {
	n, _ := fmt.Fprintf(bout, format, args...)
	exportsize += n
	if Debug_export != 0 {
		fmt.Printf(format, args...)
	}
}

var asmlist []*Node

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
	exportlist = append(exportlist, n)
}

func exportname(s string) bool {
	if r := s[0]; r < utf8.RuneSelf {
		return 'A' <= r && r <= 'Z'
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

func autoexport(n *Node, ctxt Class) {
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
		asmlist = append(asmlist, n)
	}
}

func dumppkg(p *Pkg) {
	if p == nil || p == localpkg || p.Exported || p == builtinpkg {
		return
	}
	p.Exported = true
	suffix := ""
	if !p.Direct {
		suffix = " // indirect"
	}
	exportf("\timport %s %q%s\n", p.Name, p.Path, suffix)
}

// Look for anything we need for the inline body
func reexportdeplist(ll Nodes) {
	for _, n := range ll.Slice() {
		reexportdep(n)
	}
}

func reexportdep(n *Node) {
	if n == nil {
		return
	}

	//print("reexportdep %+hN\n", n);
	switch n.Op {
	case ONAME:
		switch n.Class {
		// methods will be printed along with their type
		// nodes for T.Method expressions
		case PFUNC:
			if n.Left != nil && n.Left.Op == OTYPE {
				break
			}

			// nodes for method calls.
			if n.Type == nil || n.Type.Recv() != nil {
				break
			}
			fallthrough

		case PEXTERN:
			if n.Sym != nil && !exportedsym(n.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport name %v\n", n.Sym)
				}
				exportlist = append(exportlist, n)
			}
		}

	// Local variables in the bodies need their type.
	case ODCL:
		t := n.Left.Type

		if t != Types[t.Etype] && t != idealbool && t != idealstring {
			if t.IsPtr() {
				t = t.Elem()
			}
			if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport type %v from declaration\n", t.Sym)
				}
				exportlist = append(exportlist, t.Sym.Def)
			}
		}

	case OLITERAL:
		t := n.Type
		if t != Types[n.Type.Etype] && t != idealbool && t != idealstring {
			if t.IsPtr() {
				t = t.Elem()
			}
			if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
				if Debug['E'] != 0 {
					fmt.Printf("reexport literal type %v\n", t.Sym)
				}
				exportlist = append(exportlist, t.Sym.Def)
			}
		}
		fallthrough

	case OTYPE:
		if n.Sym != nil && n.Sym.Def != nil && !exportedsym(n.Sym) {
			if Debug['E'] != 0 {
				fmt.Printf("reexport literal/type %v\n", n.Sym)
			}
			exportlist = append(exportlist, n)
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

		switch t.Etype {
		case TARRAY, TCHAN, TPTR32, TPTR64, TSLICE:
			if t.Sym == nil {
				t = t.Elem()
			}
		}
		if t != nil && t.Sym != nil && t.Sym.Def != nil && !exportedsym(t.Sym) {
			if Debug['E'] != 0 {
				fmt.Printf("reexport type for expression %v\n", t.Sym)
			}
			exportlist = append(exportlist, t.Sym.Def)
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
	n := typecheck(s.Def, Erv)
	if n == nil || n.Op != OLITERAL {
		Fatalf("dumpexportconst: oconst nil: %v", s)
	}

	t := n.Type // may or may not be specified
	dumpexporttype(t)

	if t != nil && !t.IsUntyped() {
		exportf("\tconst %v %v = %v\n", sconv(s, FmtSharp), Tconv(t, FmtSharp), vconv(n.Val(), FmtSharp))
	} else {
		exportf("\tconst %v = %v\n", sconv(s, FmtSharp), vconv(n.Val(), FmtSharp))
	}
}

func dumpexportvar(s *Sym) {
	n := s.Def
	n = typecheck(n, Erv|Ecall)
	if n == nil || n.Type == nil {
		Yyerror("variable exported but not defined: %v", s)
		return
	}

	t := n.Type
	dumpexporttype(t)

	if t.Etype == TFUNC && n.Class == PFUNC {
		if n.Func != nil && n.Func.Inl.Len() != 0 {
			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if Debug['l'] < 2 {
				typecheckinl(n)
			}

			// NOTE: The space after %#S here is necessary for ld's export data parser.
			exportf("\tfunc %v %v { %v }\n", sconv(s, FmtSharp), Tconv(t, FmtShort|FmtSharp), hconv(n.Func.Inl, FmtSharp|FmtBody))

			reexportdeplist(n.Func.Inl)
		} else {
			exportf("\tfunc %v %v\n", sconv(s, FmtSharp), Tconv(t, FmtShort|FmtSharp))
		}
	} else {
		exportf("\tvar %v %v\n", sconv(s, FmtSharp), Tconv(t, FmtSharp))
	}
}

// methodbyname sorts types by symbol name.
type methodbyname []*Field

func (x methodbyname) Len() int           { return len(x) }
func (x methodbyname) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x methodbyname) Less(i, j int) bool { return x[i].Sym.Name < x[j].Sym.Name }

func dumpexporttype(t *Type) {
	if t == nil {
		return
	}
	if t.Printed || t == Types[t.Etype] || t == bytetype || t == runetype || t == errortype {
		return
	}
	t.Printed = true

	if t.Sym != nil {
		dumppkg(t.Sym.Pkg)
	}

	switch t.Etype {
	case TSTRUCT, TINTER:
		for _, f := range t.Fields().Slice() {
			dumpexporttype(f.Type)
		}
	case TFUNC:
		dumpexporttype(t.Recvs())
		dumpexporttype(t.Results())
		dumpexporttype(t.Params())
	case TMAP:
		dumpexporttype(t.Val())
		dumpexporttype(t.Key())
	case TARRAY, TCHAN, TPTR32, TPTR64, TSLICE:
		dumpexporttype(t.Elem())
	}

	if t.Sym == nil {
		return
	}

	var m []*Field
	for _, f := range t.Methods().Slice() {
		dumpexporttype(f.Type)
		m = append(m, f)
	}
	sort.Sort(methodbyname(m))

	exportf("\ttype %v %v\n", sconv(t.Sym, FmtSharp), Tconv(t, FmtSharp|FmtLong))
	for _, f := range m {
		if f.Nointerface {
			exportf("\t//go:nointerface\n")
		}
		if f.Type.Nname() != nil && f.Type.Nname().Func.Inl.Len() != 0 { // nname was set by caninl

			// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
			// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
			if Debug['l'] < 2 {
				typecheckinl(f.Type.Nname())
			}
			exportf("\tfunc %v %v %v { %v }\n", Tconv(f.Type.Recvs(), FmtSharp), sconv(f.Sym, FmtShort|FmtByte|FmtSharp), Tconv(f.Type, FmtShort|FmtSharp), hconv(f.Type.Nname().Func.Inl, FmtSharp|FmtBody))
			reexportdeplist(f.Type.Nname().Func.Inl)
		} else {
			exportf("\tfunc %v %v %v\n", Tconv(f.Type.Recvs(), FmtSharp), sconv(f.Sym, FmtShort|FmtByte|FmtSharp), Tconv(f.Type, FmtShort|FmtSharp))
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
		Yyerror("unexpected export symbol: %v %v", s.Def.Op, s)

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
	if buildid != "" {
		exportf("build id %q\n", buildid)
	}

	size := 0 // size of export section without enclosing markers
	if newexport {
		// binary export
		// The linker also looks for the $$ marker - use char after $$ to distinguish format.
		exportf("\n$$B\n") // indicate binary format
		if debugFormat {
			// save a copy of the export data
			var copy bytes.Buffer
			bcopy := bufio.NewWriter(&copy)
			size = export(bcopy, Debug_export != 0)
			bcopy.Flush() // flushing to bytes.Buffer cannot fail
			if n, err := bout.Write(copy.Bytes()); n != size || err != nil {
				Fatalf("error writing export data: got %d bytes, want %d bytes, err = %v", n, size, err)
			}
			// export data must contain no '$' so that we can find the end by searching for "$$"
			if bytes.IndexByte(copy.Bytes(), '$') >= 0 {
				Fatalf("export data contains $")
			}

			// verify that we can read the copied export data back in
			// (use empty package map to avoid collisions)
			savedPkgMap := pkgMap
			savedPkgs := pkgs
			pkgMap = make(map[string]*Pkg)
			pkgs = nil
			importpkg = mkpkg("")
			Import(bufio.NewReader(&copy)) // must not die
			importpkg = nil
			pkgs = savedPkgs
			pkgMap = savedPkgMap
		} else {
			size = export(bout.Writer, Debug_export != 0)
		}
		exportf("\n$$\n")
	} else {
		// textual export
		lno := lineno

		exportf("\n$$\n") // indicate textual format
		exportsize = 0
		exportf("package %s", localpkg.Name)
		if safemode {
			exportf(" safe")
		}
		exportf("\n")

		for _, p := range pkgs {
			if p.Direct {
				dumppkg(p)
			}
		}

		// exportlist grows during iteration - cannot use range
		for i := 0; i < len(exportlist); i++ {
			n := exportlist[i]
			lineno = n.Lineno
			dumpsym(n.Sym)
		}

		size = exportsize
		exportf("\n$$\n")
		lineno = lno
	}

	if Debug_export != 0 {
		fmt.Printf("export data size = %d bytes\n", size)
	}
}

// importsym declares symbol s as an imported object representable by op.
func importsym(s *Sym, op Op) {
	if s.Def != nil && s.Def.Op != op {
		pkgstr := fmt.Sprintf("during import %q", importpkg.Path)
		redeclare(s, pkgstr)
	}

	// mark the symbol so it is not reexported
	if s.Def == nil {
		if Debug['A'] != 0 || exportname(s.Name) || initname(s.Name) {
			s.Flags |= SymExport
		} else {
			s.Flags |= SymPackage // package scope
		}
	}
}

// pkgtype returns the named type declared by symbol s.
// If no such type has been declared yet, a forward declaration is returned.
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

// numImport tracks how often a package with a given name is imported.
// It is used to provide a better error message (by using the package
// path to disambiguate) if a package that appears multiple times with
// the same name appears in an error message.
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

// importconst declares symbol s as an imported constant with type t and value n.
func importconst(s *Sym, t *Type, n *Node) {
	importsym(s, OLITERAL)
	n = convlit(n, t)

	if s.Def != nil { // TODO: check if already the same.
		return
	}

	if n.Op != OLITERAL {
		Yyerror("expression must be a constant")
		return
	}

	if n.Sym != nil {
		n1 := *n
		n = &n1
	}

	n.Orig = newname(s)
	n.Sym = s
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import const %v\n", s)
	}
}

// importvar declares symbol s as an imported variable with type t.
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
		fmt.Printf("import var %v %v\n", s, Tconv(t, FmtLong))
	}
}

// importtype and importer.importtype (bimport.go) need to remain in sync.
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
		pt.Sym.Lastlineno = lineno
		declare(n, PEXTERN)
		checkwidth(pt)
	} else if !Eqtype(pt.Orig, t) {
		Yyerror("inconsistent definition for type %v during import\n\t%v (in %q)\n\t%v (in %q)", pt.Sym, Tconv(pt, FmtLong), pt.Sym.Importdef.Path, Tconv(t, FmtLong), importpkg.Path)
	}

	if Debug['E'] != 0 {
		fmt.Printf("import type %v %v\n", pt, Tconv(t, FmtLong))
	}
}

func dumpasmhdr() {
	b, err := bio.Create(asmhdr)
	if err != nil {
		Fatalf("%v", err)
	}
	fmt.Fprintf(b, "// generated by compile -asmhdr from package %s\n\n", localpkg.Name)
	for _, n := range asmlist {
		if isblanksym(n.Sym) {
			continue
		}
		switch n.Op {
		case OLITERAL:
			fmt.Fprintf(b, "#define const_%s %v\n", n.Sym.Name, vconv(n.Val(), FmtSharp))

		case OTYPE:
			t := n.Type
			if !t.IsStruct() || t.StructType().Map != nil || t.IsFuncArgStruct() {
				break
			}
			fmt.Fprintf(b, "#define %s__size %d\n", t.Sym.Name, int(t.Width))
			for _, t := range t.Fields().Slice() {
				if !isblanksym(t.Sym) {
					fmt.Fprintf(b, "#define %s_%s %d\n", n.Sym.Name, t.Sym.Name, int(t.Offset))
				}
			}
		}
	}

	b.Close()
}
