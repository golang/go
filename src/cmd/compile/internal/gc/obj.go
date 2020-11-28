// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"go/constant"
	"io"
	"io/ioutil"
	"os"
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
	if base.Flag.LinkObj == "" {
		dumpobj1(base.Flag.LowerO, modeCompilerObj|modeLinkerObj)
		return
	}
	dumpobj1(base.Flag.LowerO, modeCompilerObj)
	dumpobj1(base.Flag.LinkObj, modeLinkerObj)
}

func dumpobj1(outfile string, mode int) {
	bout, err := bio.Create(outfile)
	if err != nil {
		base.FlushErrors()
		fmt.Printf("can't create %s: %v\n", outfile, err)
		base.ErrorExit()
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
	if base.Flag.BuildID != "" {
		fmt.Fprintf(bout, "build id %q\n", base.Flag.BuildID)
	}
	if ir.LocalPkg.Name == "main" {
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
	xtops := len(xtop)

	dumpglobls()
	addptabs()
	exportlistLen := len(exportlist)
	addsignats(externdcl)
	dumpsignats()
	dumptabs()
	ptabsLen := len(ptabs)
	itabsLen := len(itabs)
	dumpimportstrings()
	dumpbasictypes()
	dumpembeds()

	// Calls to dumpsignats can generate functions,
	// like method wrappers and hash and equality routines.
	// Compile any generated functions, process any new resulting types, repeat.
	// This can't loop forever, because there is no way to generate an infinite
	// number of types in a finite amount of code.
	// In the typical case, we loop 0 or 1 times.
	// It was not until issue 24761 that we found any code that required a loop at all.
	for {
		for i := xtops; i < len(xtop); i++ {
			n := xtop[i]
			if n.Op() == ir.ODCLFUNC {
				funccompile(n.(*ir.Func))
			}
		}
		xtops = len(xtop)
		compileFunctions()
		dumpsignats()
		if xtops == len(xtop) {
			break
		}
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

	if exportlistLen != len(exportlist) {
		base.Fatalf("exportlist changed after compile functions loop")
	}
	if ptabsLen != len(ptabs) {
		base.Fatalf("ptabs changed after compile functions loop")
	}
	if itabsLen != len(itabs) {
		base.Fatalf("itabs changed after compile functions loop")
	}
}

func dumpLinkerObj(bout *bio.Writer) {
	printObjHeader(bout)

	if len(pragcgobuf) != 0 {
		// write empty export section; must be before cgo section
		fmt.Fprintf(bout, "\n$$\n\n$$\n\n")
		fmt.Fprintf(bout, "\n$$  // cgo\n")
		if err := json.NewEncoder(bout).Encode(pragcgobuf); err != nil {
			base.Fatalf("serializing pragcgobuf: %v", err)
		}
		fmt.Fprintf(bout, "\n$$\n\n")
	}

	fmt.Fprintf(bout, "\n!\n")

	obj.WriteObjFile(base.Ctxt, bout)
}

func addptabs() {
	if !base.Ctxt.Flag_dynlink || ir.LocalPkg.Name != "main" {
		return
	}
	for _, exportn := range exportlist {
		s := exportn.Sym()
		n := ir.AsNode(s.Def)
		if n == nil {
			continue
		}
		if n.Op() != ir.ONAME {
			continue
		}
		if !types.IsExported(s.Name) {
			continue
		}
		if s.Pkg.Name != "main" {
			continue
		}
		if n.Type().Etype == types.TFUNC && n.Class() == ir.PFUNC {
			// function
			ptabs = append(ptabs, ptabEntry{s: s, t: ir.AsNode(s.Def).Type()})
		} else {
			// variable
			ptabs = append(ptabs, ptabEntry{s: s, t: types.NewPtr(ir.AsNode(s.Def).Type())})
		}
	}
}

func dumpGlobal(n ir.Node) {
	if n.Type() == nil {
		base.Fatalf("external %v nil type\n", n)
	}
	if n.Class() == ir.PFUNC {
		return
	}
	if n.Sym().Pkg != ir.LocalPkg {
		return
	}
	dowidth(n.Type())
	ggloblnod(n)
}

func dumpGlobalConst(n ir.Node) {
	// only export typed constants
	t := n.Type()
	if t == nil {
		return
	}
	if n.Sym().Pkg != ir.LocalPkg {
		return
	}
	// only export integer constants for now
	if !t.IsInteger() {
		return
	}
	v := n.Val()
	if t.IsUntyped() {
		// Export untyped integers as int (if they fit).
		t = types.Types[types.TINT]
		if doesoverflow(v, t) {
			return
		}
	}
	base.Ctxt.DwarfIntConst(base.Ctxt.Pkgpath, n.Sym().Name, typesymname(t), ir.Int64Val(t, v))
}

func dumpglobls() {
	// add globals
	for _, n := range externdcl {
		switch n.Op() {
		case ir.ONAME:
			dumpGlobal(n)
		case ir.OLITERAL:
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
	for _, s := range base.Ctxt.Text {
		fn := s.Func()
		if fn == nil {
			continue
		}
		for _, gcsym := range []*obj.LSym{fn.GCArgs, fn.GCLocals} {
			if gcsym != nil && !gcsym.OnList() {
				ggloblsym(gcsym, int32(len(gcsym.P)), obj.RODATA|obj.DUPOK)
			}
		}
		if x := fn.StackObjects; x != nil {
			attr := int16(obj.RODATA)
			ggloblsym(x, int32(len(x.P)), attr)
			x.Set(obj.AttrStatic, true)
		}
		if x := fn.OpenCodedDeferInfo; x != nil {
			ggloblsym(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
		}
	}
}

func duintxx(s *obj.LSym, off int, v uint64, wid int) int {
	if off&(wid-1) != 0 {
		base.Fatalf("duintxxLSym: misaligned: v=%d wid=%d off=%d", v, wid, off)
	}
	s.WriteInt(base.Ctxt, int64(off), wid, int64(v))
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

const (
	stringSymPrefix  = "go.string."
	stringSymPattern = ".gostring.%d.%x"
)

// stringsym returns a symbol containing the string s.
// The symbol contains the string data, not a string header.
func stringsym(pos src.XPos, s string) (data *obj.LSym) {
	var symname string
	if len(s) > 100 {
		// Huge strings are hashed to avoid long names in object files.
		// Indulge in some paranoia by writing the length of s, too,
		// as protection against length extension attacks.
		// Same pattern is known to fileStringSym below.
		h := sha256.New()
		io.WriteString(h, s)
		symname = fmt.Sprintf(stringSymPattern, len(s), h.Sum(nil))
	} else {
		// Small strings get named directly by their contents.
		symname = strconv.Quote(s)
	}

	symdata := base.Ctxt.Lookup(stringSymPrefix + symname)
	if !symdata.OnList() {
		off := dstringdata(symdata, 0, s, pos, "string")
		ggloblsym(symdata, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)
		symdata.Set(obj.AttrContentAddressable, true)
	}

	return symdata
}

// fileStringSym returns a symbol for the contents and the size of file.
// If readonly is true, the symbol shares storage with any literal string
// or other file with the same content and is placed in a read-only section.
// If readonly is false, the symbol is a read-write copy separate from any other,
// for use as the backing store of a []byte.
// The content hash of file is copied into hash. (If hash is nil, nothing is copied.)
// The returned symbol contains the data itself, not a string header.
func fileStringSym(pos src.XPos, file string, readonly bool, hash []byte) (*obj.LSym, int64, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	if !info.Mode().IsRegular() {
		return nil, 0, fmt.Errorf("not a regular file")
	}
	size := info.Size()
	if size <= 1*1024 {
		data, err := ioutil.ReadAll(f)
		if err != nil {
			return nil, 0, err
		}
		if int64(len(data)) != size {
			return nil, 0, fmt.Errorf("file changed between reads")
		}
		var sym *obj.LSym
		if readonly {
			sym = stringsym(pos, string(data))
		} else {
			sym = slicedata(pos, string(data)).Sym().Linksym()
		}
		if len(hash) > 0 {
			sum := sha256.Sum256(data)
			copy(hash, sum[:])
		}
		return sym, size, nil
	}
	if size > 2e9 {
		// ggloblsym takes an int32,
		// and probably the rest of the toolchain
		// can't handle such big symbols either.
		// See golang.org/issue/9862.
		return nil, 0, fmt.Errorf("file too large")
	}

	// File is too big to read and keep in memory.
	// Compute hash if needed for read-only content hashing or if the caller wants it.
	var sum []byte
	if readonly || len(hash) > 0 {
		h := sha256.New()
		n, err := io.Copy(h, f)
		if err != nil {
			return nil, 0, err
		}
		if n != size {
			return nil, 0, fmt.Errorf("file changed between reads")
		}
		sum = h.Sum(nil)
		copy(hash, sum)
	}

	var symdata *obj.LSym
	if readonly {
		symname := fmt.Sprintf(stringSymPattern, size, sum)
		symdata = base.Ctxt.Lookup(stringSymPrefix + symname)
		if !symdata.OnList() {
			info := symdata.NewFileInfo()
			info.Name = file
			info.Size = size
			ggloblsym(symdata, int32(size), obj.DUPOK|obj.RODATA|obj.LOCAL)
			// Note: AttrContentAddressable cannot be set here,
			// because the content-addressable-handling code
			// does not know about file symbols.
		}
	} else {
		// Emit a zero-length data symbol
		// and then fix up length and content to use file.
		symdata = slicedata(pos, "").Sym().Linksym()
		symdata.Size = size
		symdata.Type = objabi.SNOPTRDATA
		info := symdata.NewFileInfo()
		info.Name = file
		info.Size = size
	}

	return symdata, size, nil
}

var slicedataGen int

func slicedata(pos src.XPos, s string) ir.Node {
	slicedataGen++
	symname := fmt.Sprintf(".gobytes.%d", slicedataGen)
	sym := ir.LocalPkg.Lookup(symname)
	symnode := NewName(sym)
	sym.Def = symnode

	lsym := sym.Linksym()
	off := dstringdata(lsym, 0, s, pos, "slice")
	ggloblsym(lsym, int32(off), obj.NOPTR|obj.LOCAL)

	return symnode
}

func slicebytes(nam ir.Node, s string) {
	if nam.Op() != ir.ONAME {
		base.Fatalf("slicebytes %v", nam)
	}
	slicesym(nam, slicedata(nam.Pos(), s), int64(len(s)))
}

func dstringdata(s *obj.LSym, off int, t string, pos src.XPos, what string) int {
	// Objects that are too large will cause the data section to overflow right away,
	// causing a cryptic error message by the linker. Check for oversize objects here
	// and provide a useful error message instead.
	if int64(len(t)) > 2e9 {
		base.ErrorfAt(pos, "%v with length %v is too big", what, len(t))
		return 0
	}

	s.WriteString(base.Ctxt, int64(off), len(t), t)
	return off + len(t)
}

func dsymptr(s *obj.LSym, off int, x *obj.LSym, xoff int) int {
	off = int(Rnd(int64(off), int64(Widthptr)))
	s.WriteAddr(base.Ctxt, int64(off), Widthptr, x, int64(xoff))
	off += Widthptr
	return off
}

func dsymptrOff(s *obj.LSym, off int, x *obj.LSym) int {
	s.WriteOff(base.Ctxt, int64(off), x, 0)
	off += 4
	return off
}

func dsymptrWeakOff(s *obj.LSym, off int, x *obj.LSym) int {
	s.WriteWeakOff(base.Ctxt, int64(off), x, 0)
	off += 4
	return off
}

// slicesym writes a static slice symbol {&arr, lencap, lencap} to n.
// arr must be an ONAME. slicesym does not modify n.
func slicesym(n, arr ir.Node, lencap int64) {
	s := n.Sym().Linksym()
	off := n.Offset()
	if arr.Op() != ir.ONAME {
		base.Fatalf("slicesym non-name arr %v", arr)
	}
	s.WriteAddr(base.Ctxt, off, Widthptr, arr.Sym().Linksym(), arr.Offset())
	s.WriteInt(base.Ctxt, off+sliceLenOffset, Widthptr, lencap)
	s.WriteInt(base.Ctxt, off+sliceCapOffset, Widthptr, lencap)
}

// addrsym writes the static address of a to n. a must be an ONAME.
// Neither n nor a is modified.
func addrsym(n, a ir.Node) {
	if n.Op() != ir.ONAME {
		base.Fatalf("addrsym n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("addrsym nil n sym")
	}
	if a.Op() != ir.ONAME {
		base.Fatalf("addrsym a op %v", a.Op())
	}
	s := n.Sym().Linksym()
	s.WriteAddr(base.Ctxt, n.Offset(), Widthptr, a.Sym().Linksym(), a.Offset())
}

// pfuncsym writes the static address of f to n. f must be a global function.
// Neither n nor f is modified.
func pfuncsym(n, f ir.Node) {
	if n.Op() != ir.ONAME {
		base.Fatalf("pfuncsym n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("pfuncsym nil n sym")
	}
	if f.Class() != ir.PFUNC {
		base.Fatalf("pfuncsym class not PFUNC %d", f.Class())
	}
	s := n.Sym().Linksym()
	s.WriteAddr(base.Ctxt, n.Offset(), Widthptr, funcsym(f.Sym()).Linksym(), f.Offset())
}

// litsym writes the static literal c to n.
// Neither n nor c is modified.
func litsym(n, c ir.Node, wid int) {
	if n.Op() != ir.ONAME {
		base.Fatalf("litsym n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("litsym nil n sym")
	}
	if !types.Identical(n.Type(), c.Type()) {
		base.Fatalf("litsym: type mismatch: %v has type %v, but %v has type %v", n, n.Type(), c, c.Type())
	}
	if c.Op() == ir.ONIL {
		return
	}
	if c.Op() != ir.OLITERAL {
		base.Fatalf("litsym c op %v", c.Op())
	}
	s := n.Sym().Linksym()
	switch u := c.Val(); u.Kind() {
	case constant.Bool:
		i := int64(obj.Bool2int(constant.BoolVal(u)))
		s.WriteInt(base.Ctxt, n.Offset(), wid, i)

	case constant.Int:
		s.WriteInt(base.Ctxt, n.Offset(), wid, ir.Int64Val(n.Type(), u))

	case constant.Float:
		f, _ := constant.Float64Val(u)
		switch n.Type().Etype {
		case types.TFLOAT32:
			s.WriteFloat32(base.Ctxt, n.Offset(), float32(f))
		case types.TFLOAT64:
			s.WriteFloat64(base.Ctxt, n.Offset(), f)
		}

	case constant.Complex:
		re, _ := constant.Float64Val(constant.Real(u))
		im, _ := constant.Float64Val(constant.Imag(u))
		switch n.Type().Etype {
		case types.TCOMPLEX64:
			s.WriteFloat32(base.Ctxt, n.Offset(), float32(re))
			s.WriteFloat32(base.Ctxt, n.Offset()+4, float32(im))
		case types.TCOMPLEX128:
			s.WriteFloat64(base.Ctxt, n.Offset(), re)
			s.WriteFloat64(base.Ctxt, n.Offset()+8, im)
		}

	case constant.String:
		i := constant.StringVal(u)
		symdata := stringsym(n.Pos(), i)
		s.WriteAddr(base.Ctxt, n.Offset(), Widthptr, symdata, 0)
		s.WriteInt(base.Ctxt, n.Offset()+int64(Widthptr), Widthptr, int64(len(i)))

	default:
		base.Fatalf("litsym unhandled OLITERAL %v", c)
	}
}
