// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/noder"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/staticdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/archive"
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"encoding/json"
	"fmt"
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
	bout.WriteString(objabi.HeaderString())
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
	noder.WriteExports(bout)
}

func dumpdata() {
	numExterns := len(typecheck.Target.Externs)
	numDecls := len(typecheck.Target.Decls)

	dumpglobls(typecheck.Target.Externs)
	reflectdata.CollectPTabs()
	numExports := len(typecheck.Target.Exports)
	addsignats(typecheck.Target.Externs)
	reflectdata.WriteRuntimeTypes()
	reflectdata.WriteTabs()
	numPTabs := reflectdata.CountPTabs()
	reflectdata.WriteImportStrings()
	reflectdata.WriteBasicTypes()
	dumpembeds()

	// Calls to WriteRuntimeTypes can generate functions,
	// like method wrappers and hash and equality routines.
	// Compile any generated functions, process any new resulting types, repeat.
	// This can't loop forever, because there is no way to generate an infinite
	// number of types in a finite amount of code.
	// In the typical case, we loop 0 or 1 times.
	// It was not until issue 24761 that we found any code that required a loop at all.
	for {
		for i := numDecls; i < len(typecheck.Target.Decls); i++ {
			if n, ok := typecheck.Target.Decls[i].(*ir.Func); ok {
				enqueueFunc(n)
			}
		}
		numDecls = len(typecheck.Target.Decls)
		compileFunctions()
		reflectdata.WriteRuntimeTypes()
		if numDecls == len(typecheck.Target.Decls) {
			break
		}
	}

	// Dump extra globals.
	dumpglobls(typecheck.Target.Externs[numExterns:])

	if reflectdata.ZeroSize > 0 {
		zero := base.PkgLinksym("go.map", "zero", obj.ABI0)
		objw.Global(zero, int32(reflectdata.ZeroSize), obj.DUPOK|obj.RODATA)
		zero.Set(obj.AttrStatic, true)
	}

	staticdata.WriteFuncSyms()
	addGCLocals()

	if numExports != len(typecheck.Target.Exports) {
		base.Fatalf("Target.Exports changed after compile functions loop")
	}
	newNumPTabs := reflectdata.CountPTabs()
	if newNumPTabs != numPTabs {
		base.Fatalf("ptabs changed after compile functions loop")
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

func dumpGlobal(n *ir.Name) {
	if n.Type() == nil {
		base.Fatalf("external %v nil type\n", n)
	}
	if n.Class == ir.PFUNC {
		return
	}
	if n.Sym().Pkg != types.LocalPkg {
		return
	}
	types.CalcSize(n.Type())
	ggloblnod(n)
	base.Ctxt.DwarfGlobal(base.Ctxt.Pkgpath, types.TypeSymName(n.Type()), n.Linksym())
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
	} else {
		// If the type of the constant is an instantiated generic, we need to emit
		// that type so the linker knows about it. See issue 51245.
		_ = reflectdata.TypeLinksym(t)
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
			objw.Global(x, int32(len(x.P)), obj.RODATA)
			x.Set(obj.AttrStatic, true)
		}
		if x := fn.OpenCodedDeferInfo; x != nil {
			objw.Global(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
		}
		if x := fn.ArgInfo; x != nil {
			objw.Global(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
			x.Set(obj.AttrStatic, true)
		}
		if x := fn.ArgLiveInfo; x != nil {
			objw.Global(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
			x.Set(obj.AttrStatic, true)
		}
		if x := fn.WrapInfo; x != nil && !x.OnList() {
			objw.Global(x, int32(len(x.P)), obj.RODATA|obj.DUPOK)
			x.Set(obj.AttrStatic, true)
		}
	}
}

func ggloblnod(nam *ir.Name) {
	s := nam.Linksym()
	s.Gotype = reflectdata.TypeLinksym(nam.Type())
	flags := 0
	if nam.Readonly() {
		flags = obj.RODATA
	}
	if nam.Type() != nil && !nam.Type().HasPointers() {
		flags |= obj.NOPTR
	}
	base.Ctxt.Globl(s, nam.Type().Size(), flags)
	if nam.LibfuzzerExtraCounter() {
		s.Type = objabi.SLIBFUZZER_EXTRA_COUNTER
	}
	if nam.Sym().Linkname != "" {
		// Make sure linkname'd symbol is non-package. When a symbol is
		// both imported and linkname'd, s.Pkg may not set to "_" in
		// types.Sym.Linksym because LSym already exists. Set it here.
		s.Pkg = "_"
	}
}

func dumpembeds() {
	for _, v := range typecheck.Target.Embeds {
		staticdata.WriteEmbed(v)
	}
}

func addsignats(dcls []ir.Node) {
	// copy types from dcl list to signatset
	for _, n := range dcls {
		if n.Op() == ir.OTYPE {
			reflectdata.NeedRuntimeType(n.Type())
		}
	}
}
