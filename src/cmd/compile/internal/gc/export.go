// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/src"
	"fmt"
)

var (
	Debug_export int // if set, print debugging information about export data
)

func exportf(bout *bio.Writer, format string, args ...interface{}) {
	fmt.Fprintf(bout, format, args...)
	if Debug_export != 0 {
		fmt.Printf(format, args...)
	}
}

var asmlist []*Node

// exportsym marks n for export (or reexport).
func exportsym(n *Node) {
	if n.Sym.OnExportList() {
		return
	}
	n.Sym.SetOnExportList(true)

	if Debug['E'] != 0 {
		fmt.Printf("export symbol %v\n", n.Sym)
	}

	exportlist = append(exportlist, n)
}

func initname(s string) bool {
	return s == "init"
}

func autoexport(n *Node, ctxt Class) {
	if n.Sym.Pkg != localpkg {
		return
	}
	if (ctxt != PEXTERN && ctxt != PFUNC) || dclcontext != PEXTERN {
		return
	}
	if n.Type != nil && n.Type.IsKind(TFUNC) && n.IsMethod() {
		return
	}

	if types.IsExported(n.Sym.Name) || initname(n.Sym.Name) {
		exportsym(n)
	}
	if asmhdr != "" && !n.Sym.Asm() {
		n.Sym.SetAsm(true)
		asmlist = append(asmlist, n)
	}
}

// methodbyname sorts types by symbol name.
type methodbyname []*types.Field

func (x methodbyname) Len() int           { return len(x) }
func (x methodbyname) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x methodbyname) Less(i, j int) bool { return x[i].Sym.Name < x[j].Sym.Name }

func dumpexport(bout *bio.Writer) {
	// The linker also looks for the $$ marker - use char after $$ to distinguish format.
	exportf(bout, "\n$$B\n") // indicate binary export format
	off := bout.Offset()
	iexport(bout.Writer)
	size := bout.Offset() - off
	exportf(bout, "\n$$\n")

	if Debug_export != 0 {
		fmt.Printf("export data size = %d bytes\n", size)
	}
}

func importsym(ipkg *types.Pkg, s *types.Sym, op Op) *Node {
	n := asNode(s.PkgDef())
	if n == nil {
		// iimport should have created a stub ONONAME
		// declaration for all imported symbols. The exception
		// is declarations for Runtimepkg, which are populated
		// by loadsys instead.
		if s.Pkg != Runtimepkg {
			Fatalf("missing ONONAME for %v\n", s)
		}

		n = dclname(s)
		s.SetPkgDef(asTypesNode(n))
		s.Importdef = ipkg
	}
	if n.Op != ONONAME && n.Op != op {
		redeclare(lineno, s, fmt.Sprintf("during import %q", ipkg.Path))
	}
	return n
}

// pkgtype returns the named type declared by symbol s.
// If no such type has been declared yet, a forward declaration is returned.
// ipkg is the package being imported
func importtype(ipkg *types.Pkg, pos src.XPos, s *types.Sym) *types.Type {
	n := importsym(ipkg, s, OTYPE)
	if n.Op != OTYPE {
		t := types.New(TFORW)
		t.Sym = s
		t.Nod = asTypesNode(n)

		n.Op = OTYPE
		n.Pos = pos
		n.Type = t
		n.SetClass(PEXTERN)
	}

	t := n.Type
	if t == nil {
		Fatalf("importtype %v", s)
	}
	return t
}

// importobj declares symbol s as an imported object representable by op.
// ipkg is the package being imported
func importobj(ipkg *types.Pkg, pos src.XPos, s *types.Sym, op Op, ctxt Class, t *types.Type) *Node {
	n := importsym(ipkg, s, op)
	if n.Op != ONONAME {
		if n.Op == op && (n.Class() != ctxt || !types.Identical(n.Type, t)) {
			redeclare(lineno, s, fmt.Sprintf("during import %q", ipkg.Path))
		}
		return nil
	}

	n.Op = op
	n.Pos = pos
	n.SetClass(ctxt)
	if ctxt == PFUNC {
		n.Sym.SetFunc(true)
	}
	n.Type = t
	return n
}

// importconst declares symbol s as an imported constant with type t and value val.
// ipkg is the package being imported
func importconst(ipkg *types.Pkg, pos src.XPos, s *types.Sym, t *types.Type, val Val) {
	n := importobj(ipkg, pos, s, OLITERAL, PEXTERN, t)
	if n == nil { // TODO: Check that value matches.
		return
	}

	n.SetVal(val)

	if Debug['E'] != 0 {
		fmt.Printf("import const %v %L = %v\n", s, t, val)
	}
}

// importfunc declares symbol s as an imported function with type t.
// ipkg is the package being imported
func importfunc(ipkg *types.Pkg, pos src.XPos, s *types.Sym, t *types.Type) {
	n := importobj(ipkg, pos, s, ONAME, PFUNC, t)
	if n == nil {
		return
	}

	n.Func = new(Func)
	t.SetNname(asTypesNode(n))

	if Debug['E'] != 0 {
		fmt.Printf("import func %v%S\n", s, t)
	}
}

// importvar declares symbol s as an imported variable with type t.
// ipkg is the package being imported
func importvar(ipkg *types.Pkg, pos src.XPos, s *types.Sym, t *types.Type) {
	n := importobj(ipkg, pos, s, ONAME, PEXTERN, t)
	if n == nil {
		return
	}

	if Debug['E'] != 0 {
		fmt.Printf("import var %v %L\n", s, t)
	}
}

// importalias declares symbol s as an imported type alias with type t.
// ipkg is the package being imported
func importalias(ipkg *types.Pkg, pos src.XPos, s *types.Sym, t *types.Type) {
	n := importobj(ipkg, pos, s, OTYPE, PEXTERN, t)
	if n == nil {
		return
	}

	if Debug['E'] != 0 {
		fmt.Printf("import type %v = %L\n", s, t)
	}
}

func dumpasmhdr() {
	b, err := bio.Create(asmhdr)
	if err != nil {
		Fatalf("%v", err)
	}
	fmt.Fprintf(b, "// generated by compile -asmhdr from package %s\n\n", localpkg.Name)
	for _, n := range asmlist {
		if n.Sym.IsBlank() {
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
			fmt.Fprintf(b, "#define %s__size %d\n", n.Sym.Name, int(t.Width))
			for _, f := range t.Fields().Slice() {
				if !f.Sym.IsBlank() {
					fmt.Fprintf(b, "#define %s_%s %d\n", n.Sym.Name, f.Sym.Name, int(f.Offset))
				}
			}
		}
	}

	b.Close()
}
