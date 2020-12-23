// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/archive"
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
	if types.LocalPkg.Name == "main" {
		fmt.Fprintf(bout, "main\n")
	}
	fmt.Fprintf(bout, "\n") // header ends with blank line
}

func startArchiveEntry(bout *bio.Writer) int64 {
	var arhdr [archive.HeaderSize]byte
	bout.Write(arhdr[:])
	return bout.Offset()
}

func finishArchiveEntry(bout *bio.Writer, start int64, name string) {
	bout.Flush()
	size := bout.Offset() - start
	if size&1 != 0 {
		bout.WriteByte(0)
	}
	bout.MustSeek(start-archive.HeaderSize, 0)

	var arhdr [archive.HeaderSize]byte
	archive.FormatHeader(arhdr[:], name, size)
	bout.Write(arhdr[:])
	bout.Flush()
	bout.MustSeek(start+size+(size&1), 0)
}

func dumpCompilerObj(bout *bio.Writer) {
	printObjHeader(bout)
	dumpexport(bout)
}

func dumpdata() {
	numExterns := len(typecheck.Target.Externs)
	numDecls := len(typecheck.Target.Decls)

	dumpglobls(typecheck.Target.Externs)
	dumpfuncsyms()
	addptabs()
	numExports := len(typecheck.Target.Exports)
	addsignats(typecheck.Target.Externs)
	dumpsignats()
	dumptabs()
	numPTabs, numITabs := CountTabs()
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
		for i := numDecls; i < len(typecheck.Target.Decls); i++ {
			n := typecheck.Target.Decls[i]
			if n.Op() == ir.ODCLFUNC {
				funccompile(n.(*ir.Func))
			}
		}
		numDecls = len(typecheck.Target.Decls)
		compileFunctions()
		dumpsignats()
		if numDecls == len(typecheck.Target.Decls) {
			break
		}
	}

	// Dump extra globals.
	dumpglobls(typecheck.Target.Externs[numExterns:])

	if zerosize > 0 {
		zero := ir.Pkgs.Map.Lookup("zero")
		objw.Global(zero.Linksym(), int32(zerosize), obj.DUPOK|obj.RODATA)
	}

	addGCLocals()

	if numExports != len(typecheck.Target.Exports) {
		base.Fatalf("Target.Exports changed after compile functions loop")
	}
	newNumPTabs, newNumITabs := CountTabs()
	if newNumPTabs != numPTabs {
		base.Fatalf("ptabs changed after compile functions loop")
	}
	if newNumITabs != numITabs {
		base.Fatalf("itabs changed after compile functions loop")
	}
}

func dumpLinkerObj(bout *bio.Writer) {
	printObjHeader(bout)

	if len(typecheck.Target.CgoPragmas) != 0 {
		// write empty export section; must be before cgo section
		fmt.Fprintf(bout, "\n$$\n\n$$\n\n")
		fmt.Fprintf(bout, "\n$$  // cgo\n")
		if err := json.NewEncoder(bout).Encode(typecheck.Target.CgoPragmas); err != nil {
			base.Fatalf("serializing pragcgobuf: %v", err)
		}
		fmt.Fprintf(bout, "\n$$\n\n")
	}

	fmt.Fprintf(bout, "\n!\n")

	obj.WriteObjFile(base.Ctxt, bout)
}

func addptabs() {
	if !base.Ctxt.Flag_dynlink || types.LocalPkg.Name != "main" {
		return
	}
	for _, exportn := range typecheck.Target.Exports {
		s := exportn.Sym()
		nn := ir.AsNode(s.Def)
		if nn == nil {
			continue
		}
		if nn.Op() != ir.ONAME {
			continue
		}
		n := nn.(*ir.Name)
		if !types.IsExported(s.Name) {
			continue
		}
		if s.Pkg.Name != "main" {
			continue
		}
		if n.Type().Kind() == types.TFUNC && n.Class_ == ir.PFUNC {
			// function
			ptabs = append(ptabs, ptabEntry{s: s, t: s.Def.Type()})
		} else {
			// variable
			ptabs = append(ptabs, ptabEntry{s: s, t: types.NewPtr(s.Def.Type())})
		}
	}
}

func dumpGlobal(n *ir.Name) {
	if n.Type() == nil {
		base.Fatalf("external %v nil type\n", n)
	}
	if n.Class_ == ir.PFUNC {
		return
	}
	if n.Sym().Pkg != types.LocalPkg {
		return
	}
	types.CalcSize(n.Type())
	ggloblnod(n)
}

func dumpGlobalConst(n ir.Node) {
	// only export typed constants
	t := n.Type()
	if t == nil {
		return
	}
	if n.Sym().Pkg != types.LocalPkg {
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
		if ir.ConstOverflow(v, t) {
			return
		}
	}
	base.Ctxt.DwarfIntConst(base.Ctxt.Pkgpath, n.Sym().Name, types.TypeSymName(t), ir.IntVal(t, v))
}

func dumpglobls(externs []ir.Node) {
	// add globals
	for _, n := range externs {
		switch n.Op() {
		case ir.ONAME:
			dumpGlobal(n.(*ir.Name))
		case ir.OLITERAL:
			dumpGlobalConst(n)
		}
	}
}

func dumpfuncsyms() {
	sort.Slice(funcsyms, func(i, j int) bool {
		return funcsyms[i].LinksymName() < funcsyms[j].LinksymName()
	})
	for _, s := range funcsyms {
		sf := s.Pkg.Lookup(ir.FuncSymName(s)).Linksym()
		objw.SymPtr(sf, 0, s.Linksym(), 0)
		objw.Global(sf, int32(types.PtrSize), obj.DUPOK|obj.RODATA)
	}
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
				objw.Global(gcsym, int32(len(gcsym.P)), obj.RODATA|obj.DUPOK)
			}
		}
		if x := fn.StackObjects; x != nil {
			attr := int16(obj.RODATA)
			objw.Global(x, int32(len(x.P)), attr)
			x.Set(obj.AttrStatic, true)
		}
		if x := fn.OpenCodedDeferInfo; x != nil {
			objw.Global(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
		}
	}
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
		objw.Global(symdata, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)
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
			objw.Global(symdata, int32(size), obj.DUPOK|obj.RODATA|obj.LOCAL)
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

func slicedata(pos src.XPos, s string) *ir.Name {
	slicedataGen++
	symname := fmt.Sprintf(".gobytes.%d", slicedataGen)
	sym := types.LocalPkg.Lookup(symname)
	symnode := typecheck.NewName(sym)
	sym.Def = symnode

	lsym := sym.Linksym()
	off := dstringdata(lsym, 0, s, pos, "slice")
	objw.Global(lsym, int32(off), obj.NOPTR|obj.LOCAL)

	return symnode
}

func slicebytes(nam *ir.Name, off int64, s string) {
	if nam.Op() != ir.ONAME {
		base.Fatalf("slicebytes %v", nam)
	}
	slicesym(nam, off, slicedata(nam.Pos(), s), int64(len(s)))
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

// slicesym writes a static slice symbol {&arr, lencap, lencap} to n+noff.
// slicesym does not modify n.
func slicesym(n *ir.Name, noff int64, arr *ir.Name, lencap int64) {
	s := n.Sym().Linksym()
	if arr.Op() != ir.ONAME {
		base.Fatalf("slicesym non-name arr %v", arr)
	}
	s.WriteAddr(base.Ctxt, noff, types.PtrSize, arr.Sym().Linksym(), 0)
	s.WriteInt(base.Ctxt, noff+types.SliceLenOffset, types.PtrSize, lencap)
	s.WriteInt(base.Ctxt, noff+types.SliceCapOffset, types.PtrSize, lencap)
}

// addrsym writes the static address of a to n. a must be an ONAME.
// Neither n nor a is modified.
func addrsym(n *ir.Name, noff int64, a *ir.Name, aoff int64) {
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
	s.WriteAddr(base.Ctxt, noff, types.PtrSize, a.Sym().Linksym(), aoff)
}

// pfuncsym writes the static address of f to n. f must be a global function.
// Neither n nor f is modified.
func pfuncsym(n *ir.Name, noff int64, f *ir.Name) {
	if n.Op() != ir.ONAME {
		base.Fatalf("pfuncsym n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("pfuncsym nil n sym")
	}
	if f.Class_ != ir.PFUNC {
		base.Fatalf("pfuncsym class not PFUNC %d", f.Class_)
	}
	s := n.Sym().Linksym()
	s.WriteAddr(base.Ctxt, noff, types.PtrSize, funcsym(f.Sym()).Linksym(), 0)
}

// litsym writes the static literal c to n.
// Neither n nor c is modified.
func litsym(n *ir.Name, noff int64, c ir.Node, wid int) {
	if n.Op() != ir.ONAME {
		base.Fatalf("litsym n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("litsym nil n sym")
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
		s.WriteInt(base.Ctxt, noff, wid, i)

	case constant.Int:
		s.WriteInt(base.Ctxt, noff, wid, ir.IntVal(c.Type(), u))

	case constant.Float:
		f, _ := constant.Float64Val(u)
		switch c.Type().Kind() {
		case types.TFLOAT32:
			s.WriteFloat32(base.Ctxt, noff, float32(f))
		case types.TFLOAT64:
			s.WriteFloat64(base.Ctxt, noff, f)
		}

	case constant.Complex:
		re, _ := constant.Float64Val(constant.Real(u))
		im, _ := constant.Float64Val(constant.Imag(u))
		switch c.Type().Kind() {
		case types.TCOMPLEX64:
			s.WriteFloat32(base.Ctxt, noff, float32(re))
			s.WriteFloat32(base.Ctxt, noff+4, float32(im))
		case types.TCOMPLEX128:
			s.WriteFloat64(base.Ctxt, noff, re)
			s.WriteFloat64(base.Ctxt, noff+8, im)
		}

	case constant.String:
		i := constant.StringVal(u)
		symdata := stringsym(n.Pos(), i)
		s.WriteAddr(base.Ctxt, noff, types.PtrSize, symdata, 0)
		s.WriteInt(base.Ctxt, noff+int64(types.PtrSize), types.PtrSize, int64(len(i)))

	default:
		base.Fatalf("litsym unhandled OLITERAL %v", c)
	}
}

func ggloblnod(nam ir.Node) {
	s := nam.Sym().Linksym()
	s.Gotype = ngotype(nam).Linksym()
	flags := 0
	if nam.Name().Readonly() {
		flags = obj.RODATA
	}
	if nam.Type() != nil && !nam.Type().HasPointers() {
		flags |= obj.NOPTR
	}
	base.Ctxt.Globl(s, nam.Type().Width, flags)
	if nam.Name().LibfuzzerExtraCounter() {
		s.Type = objabi.SLIBFUZZER_EXTRA_COUNTER
	}
	if nam.Sym().Linkname != "" {
		// Make sure linkname'd symbol is non-package. When a symbol is
		// both imported and linkname'd, s.Pkg may not set to "_" in
		// types.Sym.Linksym because LSym already exists. Set it here.
		s.Pkg = "_"
	}
}
