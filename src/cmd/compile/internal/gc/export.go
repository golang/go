// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"bytes"
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
	if buildid != "" {
		exportf(bout, "build id %q\n", buildid)
	}

	size := 0 // size of export section without enclosing markers
	// The linker also looks for the $$ marker - use char after $$ to distinguish format.
	exportf(bout, "\n$$B\n") // indicate binary export format
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
		types.CleanroomDo(func() {
			Import(types.NewPkg("", ""), bufio.NewReader(&copy)) // must not die
		})
	} else {
		size = export(bout.Writer, Debug_export != 0)
	}
	exportf(bout, "\n$$\n")

	if Debug_export != 0 {
		fmt.Printf("export data size = %d bytes\n", size)
	}
}

// importsym declares symbol s as an imported object representable by op.
// pkg is the package being imported
func importsym(pkg *types.Pkg, s *types.Sym, op Op) {
	if asNode(s.Def) != nil && asNode(s.Def).Op != op {
		pkgstr := fmt.Sprintf("during import %q", pkg.Path)
		redeclare(lineno, s, pkgstr)
	}
}

// pkgtype returns the named type declared by symbol s.
// If no such type has been declared yet, a forward declaration is returned.
// pkg is the package being imported
func pkgtype(pos src.XPos, pkg *types.Pkg, s *types.Sym) *types.Type {
	importsym(pkg, s, OTYPE)
	if asNode(s.Def) == nil || asNode(s.Def).Op != OTYPE {
		t := types.New(TFORW)
		t.Sym = s
		s.Def = asTypesNode(typenodl(pos, t))
		asNode(s.Def).Name = new(Name)
	}

	if asNode(s.Def).Type == nil {
		Fatalf("pkgtype %v", s)
	}
	return asNode(s.Def).Type
}

// importconst declares symbol s as an imported constant with type t and value val.
// pkg is the package being imported
func importconst(pos src.XPos, pkg *types.Pkg, s *types.Sym, t *types.Type, val Val) {
	importsym(pkg, s, OLITERAL)
	if asNode(s.Def) != nil { // TODO: check if already the same.
		return
	}

	n := npos(pos, nodlit(val))
	n = convlit1(n, t, false, reuseOK)
	n.Sym = s
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import const %v\n", s)
	}
}

// importvar declares symbol s as an imported variable with type t.
// pkg is the package being imported
func importvar(pos src.XPos, pkg *types.Pkg, s *types.Sym, t *types.Type) {
	importsym(pkg, s, ONAME)
	if asNode(s.Def) != nil && asNode(s.Def).Op == ONAME {
		if eqtype(t, asNode(s.Def).Type) {
			return
		}
		yyerror("inconsistent definition for var %v during import\n\t%v (in %q)\n\t%v (in %q)", s, asNode(s.Def).Type, s.Importdef.Path, t, pkg.Path)
	}

	n := newnamel(pos, s)
	s.Importdef = pkg
	n.Type = t
	declare(n, PEXTERN)

	if Debug['E'] != 0 {
		fmt.Printf("import var %v %L\n", s, t)
	}
}

// importalias declares symbol s as an imported type alias with type t.
// pkg is the package being imported
func importalias(pos src.XPos, pkg *types.Pkg, s *types.Sym, t *types.Type) {
	importsym(pkg, s, OTYPE)
	if asNode(s.Def) != nil && asNode(s.Def).Op == OTYPE {
		if eqtype(t, asNode(s.Def).Type) {
			return
		}
		yyerror("inconsistent definition for type alias %v during import\n\t%v (in %q)\n\t%v (in %q)", s, asNode(s.Def).Type, s.Importdef.Path, t, pkg.Path)
	}

	n := newnamel(pos, s)
	n.Op = OTYPE
	s.Importdef = pkg
	n.Type = t
	declare(n, PEXTERN)

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
