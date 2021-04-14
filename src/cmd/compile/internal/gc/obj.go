// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
)

// architecture-independent object file output
const ArhdrSize = 60

func formathdr(arhdr []byte, name string, size int64) {
	copy(arhdr[:], fmt.Sprintf("%-16s%-12d%-6d%-6d%-8o%-10d`\n", name, 0, 0, 0, 0644, size))
}

// These modes say which kind of object file to generate.
// The default use of the toolchain is to set both bits,
// generating a combined compiler+linker object, one that
// serves to describe the package to both the compiler and the linker.
// In fact the compiler and linker read nearly disjoint sections of
// that file, though, so in a distributed build setting it can be more
// efficient to split the output into two files, supplying the compiler
// object only to future compilations and the linker object only to
// future links.
//
// By default a combined object is written, but if -linkobj is specified
// on the command line then the default -o output is a compiler object
// and the -linkobj output is a linker object.
const (
	modeCompilerObj = 1 << iota
	modeLinkerObj
)

func dumpobj() {
	if linkobj == "" {
		dumpobj1(outfile, modeCompilerObj|modeLinkerObj)
		return
	}
	dumpobj1(outfile, modeCompilerObj)
	dumpobj1(linkobj, modeLinkerObj)
}

func dumpobj1(outfile string, mode int) {
	bout, err := bio.Create(outfile)
	if err != nil {
		flusherrors()
		fmt.Printf("can't create %s: %v\n", outfile, err)
		errorexit()
	}
	defer bout.Close()
	bout.WriteString("!<arch>\n")

	if mode&modeCompilerObj != 0 {
		start := startArchiveEntry(bout)
		dumpCompilerObj(bout)
		finishArchiveEntry(bout, start, "__.PKGDEF")
	}
	if mode&modeLinkerObj != 0 {
		start := startArchiveEntry(bout)
		dumpLinkerObj(bout)
		finishArchiveEntry(bout, start, "_go_.o")
	}
}

func printObjHeader(bout *bio.Writer) {
	fmt.Fprintf(bout, "go object %s %s %s %s\n", objabi.GOOS, objabi.GOARCH, objabi.Version, objabi.Expstring())
	if buildid != "" {
		fmt.Fprintf(bout, "build id %q\n", buildid)
	}
	if localpkg.Name == "main" {
		fmt.Fprintf(bout, "main\n")
	}
	fmt.Fprintf(bout, "\n") // header ends with blank line
}

func startArchiveEntry(bout *bio.Writer) int64 {
	var arhdr [ArhdrSize]byte
	bout.Write(arhdr[:])
	return bout.Offset()
}

func finishArchiveEntry(bout *bio.Writer, start int64, name string) {
	bout.Flush()
	size := bout.Offset() - start
	if size&1 != 0 {
		bout.WriteByte(0)
	}
	bout.MustSeek(start-ArhdrSize, 0)

	var arhdr [ArhdrSize]byte
	formathdr(arhdr[:], name, size)
	bout.Write(arhdr[:])
	bout.Flush()
	bout.MustSeek(start+size+(size&1), 0)
}

func dumpCompilerObj(bout *bio.Writer) {
	printObjHeader(bout)
	dumpexport(bout)
}

func dumpdata() {
	externs := len(externdcl)

	dumpglobls()
	addptabs()
	addsignats(externdcl)
	dumpsignats()
	dumptabs()
	dumpimportstrings()
	dumpbasictypes()

	// Calls to dumpsignats can generate functions,
	// like method wrappers and hash and equality routines.
	// Compile any generated functions, process any new resulting types, repeat.
	// This can't loop forever, because there is no way to generate an infinite
	// number of types in a finite amount of code.
	// In the typical case, we loop 0 or 1 times.
	// It was not until issue 24761 that we found any code that required a loop at all.
	for len(compilequeue) > 0 {
		compileFunctions()
		dumpsignats()
	}

	// Dump extra globals.
	tmp := externdcl

	if externdcl != nil {
		externdcl = externdcl[externs:]
	}
	dumpglobls()
	externdcl = tmp

	if zerosize > 0 {
		zero := mappkg.Lookup("zero")
		ggloblsym(zero.Linksym(), int32(zerosize), obj.DUPOK|obj.RODATA)
	}

	addGCLocals()
}

func dumpLinkerObj(bout *bio.Writer) {
	printObjHeader(bout)

	if len(pragcgobuf) != 0 {
		// write empty export section; must be before cgo section
		fmt.Fprintf(bout, "\n$$\n\n$$\n\n")
		fmt.Fprintf(bout, "\n$$  // cgo\n")
		if err := json.NewEncoder(bout).Encode(pragcgobuf); err != nil {
			Fatalf("serializing pragcgobuf: %v", err)
		}
		fmt.Fprintf(bout, "\n$$\n\n")
	}

	fmt.Fprintf(bout, "\n!\n")

	obj.WriteObjFile(Ctxt, bout, myimportpath)
}

func addptabs() {
	if !Ctxt.Flag_dynlink || localpkg.Name != "main" {
		return
	}
	for _, exportn := range exportlist {
		s := exportn.Sym
		n := asNode(s.Def)
		if n == nil {
			continue
		}
		if n.Op != ONAME {
			continue
		}
		if !types.IsExported(s.Name) {
			continue
		}
		if s.Pkg.Name != "main" {
			continue
		}
		if n.Type.Etype == TFUNC && n.Class() == PFUNC {
			// function
			ptabs = append(ptabs, ptabEntry{s: s, t: asNode(s.Def).Type})
		} else {
			// variable
			ptabs = append(ptabs, ptabEntry{s: s, t: types.NewPtr(asNode(s.Def).Type)})
		}
	}
}

func dumpGlobal(n *Node) {
	if n.Type == nil {
		Fatalf("external %v nil type\n", n)
	}
	if n.Class() == PFUNC {
		return
	}
	if n.Sym.Pkg != localpkg {
		return
	}
	dowidth(n.Type)
	ggloblnod(n)
}

func dumpGlobalConst(n *Node) {
	// only export typed constants
	t := n.Type
	if t == nil {
		return
	}
	if n.Sym.Pkg != localpkg {
		return
	}
	// only export integer constants for now
	switch t.Etype {
	case TINT8:
	case TINT16:
	case TINT32:
	case TINT64:
	case TINT:
	case TUINT8:
	case TUINT16:
	case TUINT32:
	case TUINT64:
	case TUINT:
	case TUINTPTR:
		// ok
	case TIDEAL:
		if !Isconst(n, CTINT) {
			return
		}
		x := n.Val().U.(*Mpint)
		if x.Cmp(minintval[TINT]) < 0 || x.Cmp(maxintval[TINT]) > 0 {
			return
		}
		// Ideal integers we export as int (if they fit).
		t = types.Types[TINT]
	default:
		return
	}
	Ctxt.DwarfIntConst(myimportpath, n.Sym.Name, typesymname(t), n.Int64())
}

func dumpglobls() {
	// add globals
	for _, n := range externdcl {
		switch n.Op {
		case ONAME:
			dumpGlobal(n)
		case OLITERAL:
			dumpGlobalConst(n)
		}
	}

	sort.Slice(funcsyms, func(i, j int) bool {
		return funcsyms[i].LinksymName() < funcsyms[j].LinksymName()
	})
	for _, s := range funcsyms {
		sf := s.Pkg.Lookup(funcsymname(s)).Linksym()
		dsymptr(sf, 0, s.Linksym(), 0)
		ggloblsym(sf, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}

	// Do not reprocess funcsyms on next dumpglobls call.
	funcsyms = nil
}

// addGCLocals adds gcargs, gclocals, gcregs, and stack object symbols to Ctxt.Data.
//
// This is done during the sequential phase after compilation, since
// global symbols can't be declared during parallel compilation.
func addGCLocals() {
	for _, s := range Ctxt.Text {
		if s.Func == nil {
			continue
		}
		for _, gcsym := range []*obj.LSym{s.Func.GCArgs, s.Func.GCLocals, s.Func.GCRegs} {
			if gcsym != nil && !gcsym.OnList() {
				ggloblsym(gcsym, int32(len(gcsym.P)), obj.RODATA|obj.DUPOK)
			}
		}
		if x := s.Func.StackObjects; x != nil {
			attr := int16(obj.RODATA)
			if s.DuplicateOK() {
				attr |= obj.DUPOK
			}
			ggloblsym(x, int32(len(x.P)), attr)
		}
		if x := s.Func.OpenCodedDeferInfo; x != nil {
			ggloblsym(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
		}
	}
}

func duintxx(s *obj.LSym, off int, v uint64, wid int) int {
	if off&(wid-1) != 0 {
		Fatalf("duintxxLSym: misaligned: v=%d wid=%d off=%d", v, wid, off)
	}
	s.WriteInt(Ctxt, int64(off), wid, int64(v))
	return off + wid
}

func duint8(s *obj.LSym, off int, v uint8) int {
	return duintxx(s, off, uint64(v), 1)
}

func duint16(s *obj.LSym, off int, v uint16) int {
	return duintxx(s, off, uint64(v), 2)
}

func duint32(s *obj.LSym, off int, v uint32) int {
	return duintxx(s, off, uint64(v), 4)
}

func duintptr(s *obj.LSym, off int, v uint64) int {
	return duintxx(s, off, v, Widthptr)
}

func dbvec(s *obj.LSym, off int, bv bvec) int {
	// Runtime reads the bitmaps as byte arrays. Oblige.
	for j := 0; int32(j) < bv.n; j += 8 {
		word := bv.b[j/32]
		off = duint8(s, off, uint8(word>>(uint(j)%32)))
	}
	return off
}

func stringsym(pos src.XPos, s string) (data *obj.LSym) {
	var symname string
	if len(s) > 100 {
		// Huge strings are hashed to avoid long names in object files.
		// Indulge in some paranoia by writing the length of s, too,
		// as protection against length extension attacks.
		h := sha256.New()
		io.WriteString(h, s)
		symname = fmt.Sprintf(".gostring.%d.%x", len(s), h.Sum(nil))
	} else {
		// Small strings get named directly by their contents.
		symname = strconv.Quote(s)
	}

	const prefix = "go.string."
	symdataname := prefix + symname

	symdata := Ctxt.Lookup(symdataname)

	if !symdata.SeenGlobl() {
		// string data
		off := dsname(symdata, 0, s, pos, "string")
		ggloblsym(symdata, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)
	}

	return symdata
}

var slicebytes_gen int

func slicebytes(nam *Node, s string) {
	slicebytes_gen++
	symname := fmt.Sprintf(".gobytes.%d", slicebytes_gen)
	sym := localpkg.Lookup(symname)
	symnode := newname(sym)
	sym.Def = asTypesNode(symnode)

	lsym := sym.Linksym()
	off := dsname(lsym, 0, s, nam.Pos, "slice")
	ggloblsym(lsym, int32(off), obj.NOPTR|obj.LOCAL)

	if nam.Op != ONAME {
		Fatalf("slicebytes %v", nam)
	}
	slicesym(nam, symnode, int64(len(s)))
}

func dsname(s *obj.LSym, off int, t string, pos src.XPos, what string) int {
	// Objects that are too large will cause the data section to overflow right away,
	// causing a cryptic error message by the linker. Check for oversize objects here
	// and provide a useful error message instead.
	if int64(len(t)) > 2e9 {
		yyerrorl(pos, "%v with length %v is too big", what, len(t))
		return 0
	}

	s.WriteString(Ctxt, int64(off), len(t), t)
	return off + len(t)
}

func dsymptr(s *obj.LSym, off int, x *obj.LSym, xoff int) int {
	off = int(Rnd(int64(off), int64(Widthptr)))
	s.WriteAddr(Ctxt, int64(off), Widthptr, x, int64(xoff))
	off += Widthptr
	return off
}

func dsymptrOff(s *obj.LSym, off int, x *obj.LSym) int {
	s.WriteOff(Ctxt, int64(off), x, 0)
	off += 4
	return off
}

func dsymptrWeakOff(s *obj.LSym, off int, x *obj.LSym) int {
	s.WriteWeakOff(Ctxt, int64(off), x, 0)
	off += 4
	return off
}

// slicesym writes a static slice symbol {&arr, lencap, lencap} to n.
// arr must be an ONAME. slicesym does not modify n.
func slicesym(n, arr *Node, lencap int64) {
	s := n.Sym.Linksym()
	base := n.Xoffset
	if arr.Op != ONAME {
		Fatalf("slicesym non-name arr %v", arr)
	}
	s.WriteAddr(Ctxt, base, Widthptr, arr.Sym.Linksym(), arr.Xoffset)
	s.WriteInt(Ctxt, base+sliceLenOffset, Widthptr, lencap)
	s.WriteInt(Ctxt, base+sliceCapOffset, Widthptr, lencap)
}

// addrsym writes the static address of a to n. a must be an ONAME.
// Neither n nor a is modified.
func addrsym(n, a *Node) {
	if n.Op != ONAME {
		Fatalf("addrsym n op %v", n.Op)
	}
	if n.Sym == nil {
		Fatalf("addrsym nil n sym")
	}
	if a.Op != ONAME {
		Fatalf("addrsym a op %v", a.Op)
	}
	s := n.Sym.Linksym()
	s.WriteAddr(Ctxt, n.Xoffset, Widthptr, a.Sym.Linksym(), a.Xoffset)
}

// pfuncsym writes the static address of f to n. f must be a global function.
// Neither n nor f is modified.
func pfuncsym(n, f *Node) {
	if n.Op != ONAME {
		Fatalf("pfuncsym n op %v", n.Op)
	}
	if n.Sym == nil {
		Fatalf("pfuncsym nil n sym")
	}
	if f.Class() != PFUNC {
		Fatalf("pfuncsym class not PFUNC %d", f.Class())
	}
	s := n.Sym.Linksym()
	s.WriteAddr(Ctxt, n.Xoffset, Widthptr, funcsym(f.Sym).Linksym(), f.Xoffset)
}

// litsym writes the static literal c to n.
// Neither n nor c is modified.
func litsym(n, c *Node, wid int) {
	if n.Op != ONAME {
		Fatalf("litsym n op %v", n.Op)
	}
	if c.Op != OLITERAL {
		Fatalf("litsym c op %v", c.Op)
	}
	if n.Sym == nil {
		Fatalf("litsym nil n sym")
	}
	s := n.Sym.Linksym()
	switch u := c.Val().U.(type) {
	case bool:
		i := int64(obj.Bool2int(u))
		s.WriteInt(Ctxt, n.Xoffset, wid, i)

	case *Mpint:
		s.WriteInt(Ctxt, n.Xoffset, wid, u.Int64())

	case *Mpflt:
		f := u.Float64()
		switch n.Type.Etype {
		case TFLOAT32:
			s.WriteFloat32(Ctxt, n.Xoffset, float32(f))
		case TFLOAT64:
			s.WriteFloat64(Ctxt, n.Xoffset, f)
		}

	case *Mpcplx:
		r := u.Real.Float64()
		i := u.Imag.Float64()
		switch n.Type.Etype {
		case TCOMPLEX64:
			s.WriteFloat32(Ctxt, n.Xoffset, float32(r))
			s.WriteFloat32(Ctxt, n.Xoffset+4, float32(i))
		case TCOMPLEX128:
			s.WriteFloat64(Ctxt, n.Xoffset, r)
			s.WriteFloat64(Ctxt, n.Xoffset+8, i)
		}

	case string:
		symdata := stringsym(n.Pos, u)
		s.WriteAddr(Ctxt, n.Xoffset, Widthptr, symdata, 0)
		s.WriteInt(Ctxt, n.Xoffset+int64(Widthptr), Widthptr, int64(len(u)))

	default:
		Fatalf("litsym unhandled OLITERAL %v", c)
	}
}
