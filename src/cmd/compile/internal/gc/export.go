// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"bytes"
	"cmd/internal/bio"
	"fmt"
	"unicode"
	"unicode/utf8"
)

var (
	Debug_export int // if set, print debugging information about export data
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
			yyerror("export/package mismatch: %v", n.Sym)
		}
		return
	}

	n.Sym.Flags |= SymExport
	if Debug['E'] != 0 {
		fmt.Printf("export symbol %v\n", n.Sym)
	}

	// Ensure original object is on exportlist before aliases.
	if n.Sym.Flags&SymAlias != 0 {
		exportlist = append(exportlist, n.Sym.Def)
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

	if exportname(n.Sym.Name) || initname(n.Sym.Name) {
		exportsym(n)
	}
	if asmhdr != "" && n.Sym.Pkg == localpkg && n.Sym.Flags&SymAsm == 0 {
		n.Sym.Flags |= SymAsm
		asmlist = append(asmlist, n)
	}
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
			if n.Type == nil || n.IsMethod() {
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
		OSLICELIT,
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

// methodbyname sorts types by symbol name.
type methodbyname []*Field

func (x methodbyname) Len() int           { return len(x) }
func (x methodbyname) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x methodbyname) Less(i, j int) bool { return x[i].Sym.Name < x[j].Sym.Name }

func dumpexport() {
	if buildid != "" {
		exportf("build id %q\n", buildid)
	}

	size := 0 // size of export section without enclosing markers
	// The linker also looks for the $$ marker - use char after $$ to distinguish format.
	exportf("\n$$B\n") // indicate binary export format
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
		// TODO(gri) is this still needed?
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
		if exportname(s.Name) || initname(s.Name) {
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
		yyerror("pkgtype %v", s)
	}
	return s.Def.Type
}

// importconst declares symbol s as an imported constant with type t and value n.
func importconst(s *Sym, t *Type, n *Node) {
	importsym(s, OLITERAL)
	n = convlit(n, t)

	if s.Def != nil { // TODO: check if already the same.
		return
	}

	if n.Op != OLITERAL {
		yyerror("expression must be a constant")
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
		if eqtype(t, s.Def.Type) {
			return
		}
		yyerror("inconsistent definition for var %v during import\n\t%v (in %q)\n\t%v (in %q)", s, s.Def.Type, s.Importdef.Path, t, importpkg.Path)
	}

	n := newname(s)
	s.Importdef = importpkg
	n.Type = t
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import var %v %L\n", s, t)
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
			fmt.Fprintf(b, "#define const_%s %#v\n", n.Sym.Name, n.Val())

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
