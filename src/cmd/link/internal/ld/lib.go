// Inferno utils/8l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/8l/asm.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package ld

import (
	"bufio"
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"crypto/sha1"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Data layout and relocation.

// Derived from Inferno utils/6l/l.h
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

type Arch struct {
	Funcalign        int
	Maxalign         int
	Minalign         int
	Dwarfregsp       int
	Dwarfreglr       int
	Linuxdynld       string
	Freebsddynld     string
	Netbsddynld      string
	Openbsddynld     string
	Dragonflydynld   string
	Solarisdynld     string
	Adddynrel        func(*Link, *Symbol, *Reloc) bool
	Archinit         func(*Link)
	Archreloc        func(*Link, *Reloc, *Symbol, *int64) int
	Archrelocvariant func(*Link, *Reloc, *Symbol, int64) int64
	Trampoline       func(*Link, *Reloc, *Symbol)
	Asmb             func(*Link)
	Elfreloc1        func(*Link, *Reloc, int64) int
	Elfsetupplt      func(*Link)
	Gentext          func(*Link)
	Machoreloc1      func(*Symbol, *Reloc, int64) int
	PEreloc1         func(*Symbol, *Reloc, int64) bool
	Wput             func(uint16)
	Lput             func(uint32)
	Vput             func(uint64)
	Append16         func(b []byte, v uint16) []byte
	Append32         func(b []byte, v uint32) []byte
	Append64         func(b []byte, v uint64) []byte

	// TLSIEtoLE converts a TLS Initial Executable relocation to
	// a TLS Local Executable relocation.
	//
	// This is possible when a TLS IE relocation refers to a local
	// symbol in an executable, which is typical when internally
	// linking PIE binaries.
	TLSIEtoLE func(s *Symbol, off, size int)
}

var (
	Thearch Arch
	Lcsize  int32
	rpath   Rpath
	Spsize  int32
	Symsize int32
)

// Terrible but standard terminology.
// A segment describes a block of file to load into memory.
// A section further describes the pieces of that block for
// use in debuggers and such.

const (
	MINFUNC = 16 // minimum size for a function
)

type Segment struct {
	Rwx      uint8  // permission as usual unix bits (5 = r-x etc)
	Vaddr    uint64 // virtual address
	Length   uint64 // length in memory
	Fileoff  uint64 // file offset
	Filelen  uint64 // length on disk
	Sections []*Section
}

type Section struct {
	Rwx     uint8
	Extnum  int16
	Align   int32
	Name    string
	Vaddr   uint64
	Length  uint64
	Seg     *Segment
	Elfsect *ElfShdr
	Reloff  uint64
	Rellen  uint64
}

// DynlinkingGo returns whether we are producing Go code that can live
// in separate shared libraries linked together at runtime.
func (ctxt *Link) DynlinkingGo() bool {
	if !ctxt.Loaded {
		panic("DynlinkingGo called before all symbols loaded")
	}
	canUsePlugins := ctxt.Syms.ROLookup("plugin.Open", 0) != nil
	return Buildmode == BuildmodeShared || *FlagLinkshared || Buildmode == BuildmodePlugin || canUsePlugins
}

// UseRelro returns whether to make use of "read only relocations" aka
// relro.
func UseRelro() bool {
	switch Buildmode {
	case BuildmodeCArchive, BuildmodeCShared, BuildmodeShared, BuildmodePIE, BuildmodePlugin:
		return Iself
	default:
		return *FlagLinkshared
	}
}

var (
	SysArch         *sys.Arch
	dynexp          []*Symbol
	dynlib          []string
	ldflag          []string
	havedynamic     int
	Funcalign       int
	iscgo           bool
	elfglobalsymndx int
	interpreter     string

	debug_s  bool // backup old value of debug['s']
	HEADR    int32
	Headtype objabi.HeadType

	nerrors  int
	liveness int64
)

var (
	Segtext      Segment
	Segrodata    Segment
	Segrelrodata Segment
	Segdata      Segment
	Segdwarf     Segment
)

/* whence for ldpkg */
const (
	FileObj = 0 + iota
	ArchiveObj
	Pkgdef
)

// TODO(dfc) outBuf duplicates bio.Writer
type outBuf struct {
	w   *bufio.Writer
	f   *os.File
	off int64
}

func (w *outBuf) Write(p []byte) (n int, err error) {
	n, err = w.w.Write(p)
	w.off += int64(n)
	return n, err
}

func (w *outBuf) WriteString(s string) (n int, err error) {
	n, err = coutbuf.w.WriteString(s)
	w.off += int64(n)
	return n, err
}

func (w *outBuf) Offset() int64 {
	return w.off
}

var coutbuf outBuf

const pkgname = "__.PKGDEF"

var (
	// Set if we see an object compiled by the host compiler that is not
	// from a package that is known to support internal linking mode.
	externalobj = false
	theline     string
)

func Lflag(ctxt *Link, arg string) {
	ctxt.Libdir = append(ctxt.Libdir, arg)
}

/*
 * Unix doesn't like it when we write to a running (or, sometimes,
 * recently run) binary, so remove the output file before writing it.
 * On Windows 7, remove() can force a subsequent create() to fail.
 * S_ISREG() does not exist on Plan 9.
 */
func mayberemoveoutfile() {
	if fi, err := os.Lstat(*flagOutfile); err == nil && !fi.Mode().IsRegular() {
		return
	}
	os.Remove(*flagOutfile)
}

func libinit(ctxt *Link) {
	Funcalign = Thearch.Funcalign

	// add goroot to the end of the libdir list.
	suffix := ""

	suffixsep := ""
	if *flagInstallSuffix != "" {
		suffixsep = "_"
		suffix = *flagInstallSuffix
	} else if *flagRace {
		suffixsep = "_"
		suffix = "race"
	} else if *flagMsan {
		suffixsep = "_"
		suffix = "msan"
	}

	Lflag(ctxt, filepath.Join(objabi.GOROOT, "pkg", fmt.Sprintf("%s_%s%s%s", objabi.GOOS, objabi.GOARCH, suffixsep, suffix)))

	mayberemoveoutfile()
	f, err := os.OpenFile(*flagOutfile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0775)
	if err != nil {
		Exitf("cannot create %s: %v", *flagOutfile, err)
	}

	coutbuf.w = bufio.NewWriter(f)
	coutbuf.f = f

	if *flagEntrySymbol == "" {
		switch Buildmode {
		case BuildmodeCShared, BuildmodeCArchive:
			*flagEntrySymbol = fmt.Sprintf("_rt0_%s_%s_lib", objabi.GOARCH, objabi.GOOS)
		case BuildmodeExe, BuildmodePIE:
			*flagEntrySymbol = fmt.Sprintf("_rt0_%s_%s", objabi.GOARCH, objabi.GOOS)
		case BuildmodeShared, BuildmodePlugin:
			// No *flagEntrySymbol for -buildmode=shared and plugin
		default:
			Errorf(nil, "unknown *flagEntrySymbol for buildmode %v", Buildmode)
		}
	}
}

func errorexit() {
	if coutbuf.f != nil {
		if nerrors != 0 {
			Cflush()
		}
		// For rmtemp run at atexit time on Windows.
		if err := coutbuf.f.Close(); err != nil {
			Exitf("close: %v", err)
		}
	}

	if nerrors != 0 {
		if coutbuf.f != nil {
			mayberemoveoutfile()
		}
		Exit(2)
	}

	Exit(0)
}

func loadinternal(ctxt *Link, name string) *Library {
	if *FlagLinkshared && ctxt.PackageShlib != nil {
		if shlibname := ctxt.PackageShlib[name]; shlibname != "" {
			return addlibpath(ctxt, "internal", "internal", "", name, shlibname)
		}
	}
	if ctxt.PackageFile != nil {
		if pname := ctxt.PackageFile[name]; pname != "" {
			return addlibpath(ctxt, "internal", "internal", pname, name, "")
		}
		ctxt.Logf("loadinternal: cannot find %s\n", name)
		return nil
	}

	for i := 0; i < len(ctxt.Libdir); i++ {
		if *FlagLinkshared {
			shlibname := filepath.Join(ctxt.Libdir[i], name+".shlibname")
			if ctxt.Debugvlog != 0 {
				ctxt.Logf("searching for %s.a in %s\n", name, shlibname)
			}
			if _, err := os.Stat(shlibname); err == nil {
				return addlibpath(ctxt, "internal", "internal", "", name, shlibname)
			}
		}
		pname := filepath.Join(ctxt.Libdir[i], name+".a")
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("searching for %s.a in %s\n", name, pname)
		}
		if _, err := os.Stat(pname); err == nil {
			return addlibpath(ctxt, "internal", "internal", pname, name, "")
		}
	}

	ctxt.Logf("warning: unable to find %s.a\n", name)
	return nil
}

// findLibPathCmd uses cmd command to find gcc library libname.
// It returns library full path if found, or "none" if not found.
func (ctxt *Link) findLibPathCmd(cmd, libname string) string {
	if *flagExtld == "" {
		*flagExtld = "gcc"
	}
	args := hostlinkArchArgs()
	args = append(args, cmd)
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%s %v\n", *flagExtld, args)
	}
	out, err := exec.Command(*flagExtld, args...).Output()
	if err != nil {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("not using a %s file because compiler failed\n%v\n%s\n", libname, err, out)
		}
		return "none"
	}
	return strings.TrimSpace(string(out))
}

// findLibPath searches for library libname.
// It returns library full path if found, or "none" if not found.
func (ctxt *Link) findLibPath(libname string) string {
	return ctxt.findLibPathCmd("--print-file-name="+libname, libname)
}

func (ctxt *Link) loadlib() {
	switch Buildmode {
	case BuildmodeCShared, BuildmodePlugin:
		s := ctxt.Syms.Lookup("runtime.islibrary", 0)
		s.Attr |= AttrDuplicateOK
		Adduint8(ctxt, s, 1)
	case BuildmodeCArchive:
		s := ctxt.Syms.Lookup("runtime.isarchive", 0)
		s.Attr |= AttrDuplicateOK
		Adduint8(ctxt, s, 1)
	}

	loadinternal(ctxt, "runtime")
	if SysArch.Family == sys.ARM {
		loadinternal(ctxt, "math")
	}
	if *flagRace {
		loadinternal(ctxt, "runtime/race")
	}
	if *flagMsan {
		loadinternal(ctxt, "runtime/msan")
	}

	var i int
	for i = 0; i < len(ctxt.Library); i++ {
		iscgo = iscgo || ctxt.Library[i].Pkg == "runtime/cgo"
		if ctxt.Library[i].Shlib == "" {
			if ctxt.Debugvlog > 1 {
				ctxt.Logf("%5.2f autolib: %s (from %s)\n", Cputime(), ctxt.Library[i].File, ctxt.Library[i].Objref)
			}
			objfile(ctxt, ctxt.Library[i])
		}
	}

	for i = 0; i < len(ctxt.Library); i++ {
		if ctxt.Library[i].Shlib != "" {
			if ctxt.Debugvlog > 1 {
				ctxt.Logf("%5.2f autolib: %s (from %s)\n", Cputime(), ctxt.Library[i].Shlib, ctxt.Library[i].Objref)
			}
			ldshlibsyms(ctxt, ctxt.Library[i].Shlib)
		}
	}

	// We now have enough information to determine the link mode.
	determineLinkMode(ctxt)

	// Recalculate pe parameters now that we have Linkmode set.
	if Headtype == objabi.Hwindows {
		Peinit(ctxt)
	}

	if Headtype == objabi.Hdarwin && Linkmode == LinkExternal {
		*FlagTextAddr = 0
	}

	if Linkmode == LinkExternal && SysArch.Family == sys.PPC64 {
		toc := ctxt.Syms.Lookup(".TOC.", 0)
		toc.Type = SDYNIMPORT
	}

	if Linkmode == LinkExternal && !iscgo {
		// This indicates a user requested -linkmode=external.
		// The startup code uses an import of runtime/cgo to decide
		// whether to initialize the TLS.  So give it one. This could
		// be handled differently but it's an unusual case.
		loadinternal(ctxt, "runtime/cgo")

		if i < len(ctxt.Library) {
			if ctxt.Library[i].Shlib != "" {
				ldshlibsyms(ctxt, ctxt.Library[i].Shlib)
			} else {
				if Buildmode == BuildmodeShared || *FlagLinkshared {
					Exitf("cannot implicitly include runtime/cgo in a shared library")
				}
				objfile(ctxt, ctxt.Library[i])
			}
		}
	}

	if Linkmode == LinkInternal {
		// Drop all the cgo_import_static declarations.
		// Turns out we won't be needing them.
		for _, s := range ctxt.Syms.Allsym {
			if s.Type == SHOSTOBJ {
				// If a symbol was marked both
				// cgo_import_static and cgo_import_dynamic,
				// then we want to make it cgo_import_dynamic
				// now.
				if s.Extname != "" && s.Dynimplib != "" && !s.Attr.CgoExport() {
					s.Type = SDYNIMPORT
				} else {
					s.Type = 0
				}
			}
		}
	}

	tlsg := ctxt.Syms.Lookup("runtime.tlsg", 0)

	// runtime.tlsg is used for external linking on platforms that do not define
	// a variable to hold g in assembly (currently only intel).
	if tlsg.Type == 0 {
		tlsg.Type = STLSBSS
		tlsg.Size = int64(SysArch.PtrSize)
	} else if tlsg.Type != SDYNIMPORT {
		Errorf(nil, "runtime declared tlsg variable %v", tlsg.Type)
	}
	tlsg.Attr |= AttrReachable
	ctxt.Tlsg = tlsg

	var moduledata *Symbol
	if Buildmode == BuildmodePlugin {
		moduledata = ctxt.Syms.Lookup("local.pluginmoduledata", 0)
		moduledata.Attr |= AttrLocal
	} else {
		moduledata = ctxt.Syms.Lookup("runtime.firstmoduledata", 0)
	}
	if moduledata.Type != 0 && moduledata.Type != SDYNIMPORT {
		// If the module (toolchain-speak for "executable or shared
		// library") we are linking contains the runtime package, it
		// will define the runtime.firstmoduledata symbol and we
		// truncate it back to 0 bytes so we can define its entire
		// contents in symtab.go:symtab().
		moduledata.Size = 0

		// In addition, on ARM, the runtime depends on the linker
		// recording the value of GOARM.
		if SysArch.Family == sys.ARM {
			s := ctxt.Syms.Lookup("runtime.goarm", 0)
			s.Type = SRODATA
			s.Size = 0
			Adduint8(ctxt, s, uint8(objabi.GOARM))
		}

		if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
			s := ctxt.Syms.Lookup("runtime.framepointer_enabled", 0)
			s.Type = SRODATA
			s.Size = 0
			Adduint8(ctxt, s, 1)
		}
	} else {
		// If OTOH the module does not contain the runtime package,
		// create a local symbol for the moduledata.
		moduledata = ctxt.Syms.Lookup("local.moduledata", 0)
		moduledata.Attr |= AttrLocal
	}
	// In all cases way we mark the moduledata as noptrdata to hide it from
	// the GC.
	moduledata.Type = SNOPTRDATA
	moduledata.Attr |= AttrReachable
	ctxt.Moduledata = moduledata

	// Now that we know the link mode, trim the dynexp list.
	x := AttrCgoExportDynamic

	if Linkmode == LinkExternal {
		x = AttrCgoExportStatic
	}
	w := 0
	for i := 0; i < len(dynexp); i++ {
		if dynexp[i].Attr&x != 0 {
			dynexp[w] = dynexp[i]
			w++
		}
	}
	dynexp = dynexp[:w]

	// In internal link mode, read the host object files.
	if Linkmode == LinkInternal {
		hostobjs(ctxt)

		// If we have any undefined symbols in external
		// objects, try to read them from the libgcc file.
		any := false
		for _, s := range ctxt.Syms.Allsym {
			for _, r := range s.R {
				if r.Sym != nil && r.Sym.Type&SMASK == SXREF && r.Sym.Name != ".got" {
					any = true
					break
				}
			}
		}
		if any {
			if *flagLibGCC == "" {
				*flagLibGCC = ctxt.findLibPathCmd("--print-libgcc-file-name", "libgcc")
			}
			if *flagLibGCC != "none" {
				hostArchive(ctxt, *flagLibGCC)
			}
			if Headtype == objabi.Hwindows {
				if p := ctxt.findLibPath("libmingwex.a"); p != "none" {
					hostArchive(ctxt, p)
				}
				if p := ctxt.findLibPath("libmingw32.a"); p != "none" {
					hostArchive(ctxt, p)
				}
				// TODO: maybe do something similar to peimporteddlls to collect all lib names
				// and try link them all to final exe just like libmingwex.a and libmingw32.a:
				/*
					for:
					#cgo windows LDFLAGS: -lmsvcrt -lm
					import:
					libmsvcrt.a libm.a
				*/
			}
		}
	} else {
		hostlinksetup()
	}

	// We've loaded all the code now.
	ctxt.Loaded = true

	// If there are no dynamic libraries needed, gcc disables dynamic linking.
	// Because of this, glibc's dynamic ELF loader occasionally (like in version 2.13)
	// assumes that a dynamic binary always refers to at least one dynamic library.
	// Rather than be a source of test cases for glibc, disable dynamic linking
	// the same way that gcc would.
	//
	// Exception: on OS X, programs such as Shark only work with dynamic
	// binaries, so leave it enabled on OS X (Mach-O) binaries.
	// Also leave it enabled on Solaris which doesn't support
	// statically linked binaries.
	if Buildmode == BuildmodeExe {
		if havedynamic == 0 && Headtype != objabi.Hdarwin && Headtype != objabi.Hsolaris {
			*FlagD = true
		}
	}

	// If package versioning is required, generate a hash of the
	// the packages used in the link.
	if Buildmode == BuildmodeShared || Buildmode == BuildmodePlugin || ctxt.Syms.ROLookup("plugin.Open", 0) != nil {
		for i = 0; i < len(ctxt.Library); i++ {
			if ctxt.Library[i].Shlib == "" {
				genhash(ctxt, ctxt.Library[i])
			}
		}
	}

	if SysArch == sys.Arch386 {
		if (Buildmode == BuildmodeCArchive && Iself) || Buildmode == BuildmodeCShared || Buildmode == BuildmodePIE || ctxt.DynlinkingGo() {
			got := ctxt.Syms.Lookup("_GLOBAL_OFFSET_TABLE_", 0)
			got.Type = SDYNIMPORT
			got.Attr |= AttrReachable
		}
	}

	importcycles()

	// put symbols into Textp
	// do it in postorder so that packages are laid down in dependency order
	// internal first, then everything else
	ctxt.Library = postorder(ctxt.Library)
	for _, doInternal := range [2]bool{true, false} {
		for _, lib := range ctxt.Library {
			if isRuntimeDepPkg(lib.Pkg) != doInternal {
				continue
			}
			ctxt.Textp = append(ctxt.Textp, lib.textp...)
			for _, s := range lib.dupTextSyms {
				if !s.Attr.OnList() {
					ctxt.Textp = append(ctxt.Textp, s)
					s.Attr |= AttrOnList
					// dupok symbols may be defined in multiple packages. its
					// associated package is chosen sort of arbitrarily (the
					// first containing package that the linker loads). canonicalize
					// it here to the package with which it will be laid down
					// in text.
					s.File = objabi.PathToPrefix(lib.Pkg)
				}
			}
		}
	}

	if len(ctxt.Shlibs) > 0 {
		// We might have overwritten some functions above (this tends to happen for the
		// autogenerated type equality/hashing functions) and we don't want to generated
		// pcln table entries for these any more so remove them from Textp.
		textp := make([]*Symbol, 0, len(ctxt.Textp))
		for _, s := range ctxt.Textp {
			if s.Type != SDYNIMPORT {
				textp = append(textp, s)
			}
		}
		ctxt.Textp = textp
	}
}

/*
 * look for the next file in an archive.
 * adapted from libmach.
 */
func nextar(bp *bio.Reader, off int64, a *ArHdr) int64 {
	if off&1 != 0 {
		off++
	}
	bp.Seek(off, 0)
	var buf [SAR_HDR]byte
	if n, err := io.ReadFull(bp, buf[:]); err != nil {
		if n == 0 && err != io.EOF {
			return -1
		}
		return 0
	}

	a.name = artrim(buf[0:16])
	a.date = artrim(buf[16:28])
	a.uid = artrim(buf[28:34])
	a.gid = artrim(buf[34:40])
	a.mode = artrim(buf[40:48])
	a.size = artrim(buf[48:58])
	a.fmag = artrim(buf[58:60])

	arsize := atolwhex(a.size)
	if arsize&1 != 0 {
		arsize++
	}
	return arsize + SAR_HDR
}

func genhash(ctxt *Link, lib *Library) {
	f, err := bio.Open(lib.File)
	if err != nil {
		Errorf(nil, "cannot open file %s for hash generation: %v", lib.File, err)
		return
	}
	defer f.Close()

	var arhdr ArHdr
	l := nextar(f, int64(len(ARMAG)), &arhdr)
	if l <= 0 {
		Errorf(nil, "%s: short read on archive file symbol header", lib.File)
		return
	}

	h := sha1.New()

	// To compute the hash of a package, we hash the first line of
	// __.PKGDEF (which contains the toolchain version and any
	// GOEXPERIMENT flags) and the export data (which is between
	// the first two occurences of "\n$$").

	pkgDefBytes := make([]byte, atolwhex(arhdr.size))
	_, err = io.ReadFull(f, pkgDefBytes)
	if err != nil {
		Errorf(nil, "%s: error reading package data: %v", lib.File, err)
		return
	}
	firstEOL := bytes.Index(pkgDefBytes, []byte("\n"))
	if firstEOL < 0 {
		Errorf(nil, "cannot parse package data of %s for hash generation, no newline found", lib.File)
		return
	}
	firstDoubleDollar := bytes.Index(pkgDefBytes, []byte("\n$$"))
	if firstDoubleDollar < 0 {
		Errorf(nil, "cannot parse package data of %s for hash generation, no \\n$$ found", lib.File)
		return
	}
	secondDoubleDollar := bytes.Index(pkgDefBytes[firstDoubleDollar+1:], []byte("\n$$"))
	if secondDoubleDollar < 0 {
		Errorf(nil, "cannot parse package data of %s for hash generation, only one \\n$$ found", lib.File)
		return
	}
	h.Write(pkgDefBytes[0:firstEOL])
	h.Write(pkgDefBytes[firstDoubleDollar : firstDoubleDollar+secondDoubleDollar])
	lib.hash = hex.EncodeToString(h.Sum(nil))
}

func objfile(ctxt *Link, lib *Library) {
	pkg := objabi.PathToPrefix(lib.Pkg)

	if ctxt.Debugvlog > 1 {
		ctxt.Logf("%5.2f ldobj: %s (%s)\n", Cputime(), lib.File, pkg)
	}
	f, err := bio.Open(lib.File)
	if err != nil {
		Exitf("cannot open file %s: %v", lib.File, err)
	}

	for i := 0; i < len(ARMAG); i++ {
		if c, err := f.ReadByte(); err == nil && c == ARMAG[i] {
			continue
		}

		/* load it as a regular file */
		l := f.Seek(0, 2)

		f.Seek(0, 0)
		ldobj(ctxt, f, lib, l, lib.File, lib.File, FileObj)
		f.Close()

		return
	}

	/* process __.PKGDEF */
	off := f.Offset()

	var arhdr ArHdr
	l := nextar(f, off, &arhdr)
	var pname string
	if l <= 0 {
		Errorf(nil, "%s: short read on archive file symbol header", lib.File)
		goto out
	}

	if !strings.HasPrefix(arhdr.name, pkgname) {
		Errorf(nil, "%s: cannot find package header", lib.File)
		goto out
	}

	off += l

	ldpkg(ctxt, f, pkg, atolwhex(arhdr.size), lib.File, Pkgdef)

	/*
	 * load all the object files from the archive now.
	 * this gives us sequential file access and keeps us
	 * from needing to come back later to pick up more
	 * objects.  it breaks the usual C archive model, but
	 * this is Go, not C.  the common case in Go is that
	 * we need to load all the objects, and then we throw away
	 * the individual symbols that are unused.
	 *
	 * loading every object will also make it possible to
	 * load foreign objects not referenced by __.PKGDEF.
	 */
	for {
		l = nextar(f, off, &arhdr)
		if l == 0 {
			break
		}
		if l < 0 {
			Exitf("%s: malformed archive", lib.File)
		}

		off += l

		pname = fmt.Sprintf("%s(%s)", lib.File, arhdr.name)
		l = atolwhex(arhdr.size)
		ldobj(ctxt, f, lib, l, pname, lib.File, ArchiveObj)
	}

out:
	f.Close()
}

type Hostobj struct {
	ld     func(*Link, *bio.Reader, string, int64, string)
	pkg    string
	pn     string
	file   string
	off    int64
	length int64
}

var hostobj []Hostobj

// These packages can use internal linking mode.
// Others trigger external mode.
var internalpkg = []string{
	"crypto/x509",
	"net",
	"os/user",
	"runtime/cgo",
	"runtime/race",
	"runtime/msan",
}

func ldhostobj(ld func(*Link, *bio.Reader, string, int64, string), f *bio.Reader, pkg string, length int64, pn string, file string) *Hostobj {
	isinternal := false
	for i := 0; i < len(internalpkg); i++ {
		if pkg == internalpkg[i] {
			isinternal = true
			break
		}
	}

	// DragonFly declares errno with __thread, which results in a symbol
	// type of R_386_TLS_GD or R_X86_64_TLSGD. The Go linker does not
	// currently know how to handle TLS relocations, hence we have to
	// force external linking for any libraries that link in code that
	// uses errno. This can be removed if the Go linker ever supports
	// these relocation types.
	if Headtype == objabi.Hdragonfly {
		if pkg == "net" || pkg == "os/user" {
			isinternal = false
		}
	}

	if !isinternal {
		externalobj = true
	}

	hostobj = append(hostobj, Hostobj{})
	h := &hostobj[len(hostobj)-1]
	h.ld = ld
	h.pkg = pkg
	h.pn = pn
	h.file = file
	h.off = f.Offset()
	h.length = length
	return h
}

func hostobjs(ctxt *Link) {
	var h *Hostobj

	for i := 0; i < len(hostobj); i++ {
		h = &hostobj[i]
		f, err := bio.Open(h.file)
		if err != nil {
			Exitf("cannot reopen %s: %v", h.pn, err)
		}

		f.Seek(h.off, 0)
		h.ld(ctxt, f, h.pkg, h.length, h.pn)
		f.Close()
	}
}

// provided by lib9

func rmtemp() {
	os.RemoveAll(*flagTmpdir)
}

func hostlinksetup() {
	if Linkmode != LinkExternal {
		return
	}

	// For external link, record that we need to tell the external linker -s,
	// and turn off -s internally: the external linker needs the symbol
	// information for its final link.
	debug_s = *FlagS
	*FlagS = false

	// create temporary directory and arrange cleanup
	if *flagTmpdir == "" {
		dir, err := ioutil.TempDir("", "go-link-")
		if err != nil {
			log.Fatal(err)
		}
		*flagTmpdir = dir
		AtExit(rmtemp)
	}

	// change our output to temporary object file
	coutbuf.f.Close()
	mayberemoveoutfile()

	p := filepath.Join(*flagTmpdir, "go.o")
	var err error
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0775)
	if err != nil {
		Exitf("cannot create %s: %v", p, err)
	}

	coutbuf.w = bufio.NewWriter(f)
	coutbuf.f = f
}

// hostobjCopy creates a copy of the object files in hostobj in a
// temporary directory.
func hostobjCopy() (paths []string) {
	var wg sync.WaitGroup
	sema := make(chan struct{}, runtime.NumCPU()) // limit open file descriptors
	for i, h := range hostobj {
		h := h
		dst := filepath.Join(*flagTmpdir, fmt.Sprintf("%06d.o", i))
		paths = append(paths, dst)

		wg.Add(1)
		go func() {
			sema <- struct{}{}
			defer func() {
				<-sema
				wg.Done()
			}()
			f, err := os.Open(h.file)
			if err != nil {
				Exitf("cannot reopen %s: %v", h.pn, err)
			}
			if _, err := f.Seek(h.off, 0); err != nil {
				Exitf("cannot seek %s: %v", h.pn, err)
			}

			w, err := os.Create(dst)
			if err != nil {
				Exitf("cannot create %s: %v", dst, err)
			}
			if _, err := io.CopyN(w, f, h.length); err != nil {
				Exitf("cannot write %s: %v", dst, err)
			}
			if err := w.Close(); err != nil {
				Exitf("cannot close %s: %v", dst, err)
			}
		}()
	}
	wg.Wait()
	return paths
}

// writeGDBLinkerScript creates gcc linker script file in temp
// directory. writeGDBLinkerScript returns created file path.
// The script is used to work around gcc bug
// (see https://golang.org/issue/20183 for details).
func writeGDBLinkerScript() string {
	name := "fix_debug_gdb_scripts.ld"
	path := filepath.Join(*flagTmpdir, name)
	src := `SECTIONS
{
  .debug_gdb_scripts BLOCK(__section_alignment__) (NOLOAD) :
  {
    *(.debug_gdb_scripts)
  }
}
INSERT AFTER .debug_types;
`
	err := ioutil.WriteFile(path, []byte(src), 0666)
	if err != nil {
		Errorf(nil, "WriteFile %s failed: %v", name, err)
	}
	return path
}

// archive builds a .a archive from the hostobj object files.
func (ctxt *Link) archive() {
	if Buildmode != BuildmodeCArchive {
		return
	}

	if *flagExtar == "" {
		*flagExtar = "ar"
	}

	mayberemoveoutfile()

	// Force the buffer to flush here so that external
	// tools will see a complete file.
	Cflush()
	if err := coutbuf.f.Close(); err != nil {
		Exitf("close: %v", err)
	}
	coutbuf.f = nil

	argv := []string{*flagExtar, "-q", "-c", "-s", *flagOutfile}
	argv = append(argv, filepath.Join(*flagTmpdir, "go.o"))
	argv = append(argv, hostobjCopy()...)

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("archive: %s\n", strings.Join(argv, " "))
	}

	if out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput(); err != nil {
		Exitf("running %s failed: %v\n%s", argv[0], err, out)
	}
}

func (l *Link) hostlink() {
	if Linkmode != LinkExternal || nerrors > 0 {
		return
	}
	if Buildmode == BuildmodeCArchive {
		return
	}

	if *flagExtld == "" {
		*flagExtld = "gcc"
	}

	var argv []string
	argv = append(argv, *flagExtld)
	argv = append(argv, hostlinkArchArgs()...)

	if !*FlagS && !debug_s {
		argv = append(argv, "-gdwarf-2")
	} else if Headtype == objabi.Hdarwin {
		// Recent versions of macOS print
		//	ld: warning: option -s is obsolete and being ignored
		// so do not pass any arguments.
	} else {
		argv = append(argv, "-s")
	}

	switch Headtype {
	case objabi.Hdarwin:
		argv = append(argv, "-Wl,-headerpad,1144")
		if l.DynlinkingGo() {
			argv = append(argv, "-Wl,-flat_namespace")
		} else if !SysArch.InFamily(sys.ARM64) {
			argv = append(argv, "-Wl,-no_pie")
		}
	case objabi.Hopenbsd:
		argv = append(argv, "-Wl,-nopie")
	case objabi.Hwindows:
		if windowsgui {
			argv = append(argv, "-mwindows")
		} else {
			argv = append(argv, "-mconsole")
		}
	}

	switch Buildmode {
	case BuildmodeExe:
		if Headtype == objabi.Hdarwin {
			argv = append(argv, "-Wl,-pagezero_size,4000000")
		}
	case BuildmodePIE:
		if UseRelro() {
			argv = append(argv, "-Wl,-z,relro")
		}
		argv = append(argv, "-pie")
	case BuildmodeCShared:
		if Headtype == objabi.Hdarwin {
			argv = append(argv, "-dynamiclib")
			if SysArch.Family != sys.AMD64 {
				argv = append(argv, "-Wl,-read_only_relocs,suppress")
			}
		} else {
			// ELF.
			argv = append(argv, "-Wl,-Bsymbolic")
			if UseRelro() {
				argv = append(argv, "-Wl,-z,relro")
			}
			// Pass -z nodelete to mark the shared library as
			// non-closeable: a dlclose will do nothing.
			argv = append(argv, "-shared", "-Wl,-z,nodelete")
		}
	case BuildmodeShared:
		if UseRelro() {
			argv = append(argv, "-Wl,-z,relro")
		}
		argv = append(argv, "-shared")
	case BuildmodePlugin:
		if Headtype == objabi.Hdarwin {
			argv = append(argv, "-dynamiclib")
		} else {
			if UseRelro() {
				argv = append(argv, "-Wl,-z,relro")
			}
			argv = append(argv, "-shared")
		}
	}

	if Iself && l.DynlinkingGo() {
		// We force all symbol resolution to be done at program startup
		// because lazy PLT resolution can use large amounts of stack at
		// times we cannot allow it to do so.
		argv = append(argv, "-Wl,-znow")

		// Do not let the host linker generate COPY relocations. These
		// can move symbols out of sections that rely on stable offsets
		// from the beginning of the section (like STYPE).
		argv = append(argv, "-Wl,-znocopyreloc")

		if SysArch.InFamily(sys.ARM, sys.ARM64) {
			// On ARM, the GNU linker will generate COPY relocations
			// even with -znocopyreloc set.
			// https://sourceware.org/bugzilla/show_bug.cgi?id=19962
			//
			// On ARM64, the GNU linker will fail instead of
			// generating COPY relocations.
			//
			// In both cases, switch to gold.
			argv = append(argv, "-fuse-ld=gold")

			// If gold is not installed, gcc will silently switch
			// back to ld.bfd. So we parse the version information
			// and provide a useful error if gold is missing.
			cmd := exec.Command(*flagExtld, "-fuse-ld=gold", "-Wl,--version")
			if out, err := cmd.CombinedOutput(); err == nil {
				if !bytes.Contains(out, []byte("GNU gold")) {
					log.Fatalf("ARM external linker must be gold (issue #15696), but is not: %s", out)
				}
			}
		}
	}

	if Iself && len(buildinfo) > 0 {
		argv = append(argv, fmt.Sprintf("-Wl,--build-id=0x%x", buildinfo))
	}

	// On Windows, given -o foo, GCC will append ".exe" to produce
	// "foo.exe".  We have decided that we want to honor the -o
	// option. To make this work, we append a '.' so that GCC
	// will decide that the file already has an extension. We
	// only want to do this when producing a Windows output file
	// on a Windows host.
	outopt := *flagOutfile
	if objabi.GOOS == "windows" && runtime.GOOS == "windows" && filepath.Ext(outopt) == "" {
		outopt += "."
	}
	argv = append(argv, "-o")
	argv = append(argv, outopt)

	if rpath.val != "" {
		argv = append(argv, fmt.Sprintf("-Wl,-rpath,%s", rpath.val))
	}

	// Force global symbols to be exported for dlopen, etc.
	if Iself {
		argv = append(argv, "-rdynamic")
	}

	if strings.Contains(argv[0], "clang") {
		argv = append(argv, "-Qunused-arguments")
	}

	argv = append(argv, filepath.Join(*flagTmpdir, "go.o"))
	argv = append(argv, hostobjCopy()...)

	if *FlagLinkshared {
		seenDirs := make(map[string]bool)
		seenLibs := make(map[string]bool)
		addshlib := func(path string) {
			dir, base := filepath.Split(path)
			if !seenDirs[dir] {
				argv = append(argv, "-L"+dir)
				if !rpath.set {
					argv = append(argv, "-Wl,-rpath="+dir)
				}
				seenDirs[dir] = true
			}
			base = strings.TrimSuffix(base, ".so")
			base = strings.TrimPrefix(base, "lib")
			if !seenLibs[base] {
				argv = append(argv, "-l"+base)
				seenLibs[base] = true
			}
		}
		for _, shlib := range l.Shlibs {
			addshlib(shlib.Path)
			for _, dep := range shlib.Deps {
				if dep == "" {
					continue
				}
				libpath := findshlib(l, dep)
				if libpath != "" {
					addshlib(libpath)
				}
			}
		}
	}

	argv = append(argv, ldflag...)

	// When building a program with the default -buildmode=exe the
	// gc compiler generates code requires DT_TEXTREL in a
	// position independent executable (PIE). On systems where the
	// toolchain creates PIEs by default, and where DT_TEXTREL
	// does not work, the resulting programs will not run. See
	// issue #17847. To avoid this problem pass -no-pie to the
	// toolchain if it is supported.
	if Buildmode == BuildmodeExe {
		src := filepath.Join(*flagTmpdir, "trivial.c")
		if err := ioutil.WriteFile(src, []byte("int main() { return 0; }"), 0666); err != nil {
			Errorf(nil, "WriteFile trivial.c failed: %v", err)
		}

		// GCC uses -no-pie, clang uses -nopie.
		for _, nopie := range []string{"-no-pie", "-nopie"} {
			cmd := exec.Command(argv[0], nopie, "trivial.c")
			cmd.Dir = *flagTmpdir
			cmd.Env = append([]string{"LC_ALL=C"}, os.Environ()...)
			out, err := cmd.CombinedOutput()
			// GCC says "unrecognized command line option ‘-no-pie’"
			// clang says "unknown argument: '-no-pie'"
			supported := err == nil && !bytes.Contains(out, []byte("unrecognized")) && !bytes.Contains(out, []byte("unknown"))
			if supported {
				argv = append(argv, nopie)
				break
			}
		}
	}

	for _, p := range strings.Fields(*flagExtldflags) {
		argv = append(argv, p)

		// clang, unlike GCC, passes -rdynamic to the linker
		// even when linking with -static, causing a linker
		// error when using GNU ld. So take out -rdynamic if
		// we added it. We do it in this order, rather than
		// only adding -rdynamic later, so that -*extldflags
		// can override -rdynamic without using -static.
		if Iself && p == "-static" {
			for i := range argv {
				if argv[i] == "-rdynamic" {
					argv[i] = "-static"
				}
			}
		}
	}
	if Headtype == objabi.Hwindows {
		// use gcc linker script to work around gcc bug
		// (see https://golang.org/issue/20183 for details).
		p := writeGDBLinkerScript()
		argv = append(argv, "-Wl,-T,"+p)
		// libmingw32 and libmingwex have some inter-dependencies,
		// so must use linker groups.
		argv = append(argv, "-Wl,--start-group", "-lmingwex", "-lmingw32", "-Wl,--end-group")
		argv = append(argv, peimporteddlls()...)
	}

	if l.Debugvlog != 0 {
		l.Logf("%5.2f host link:", Cputime())
		for _, v := range argv {
			l.Logf(" %q", v)
		}
		l.Logf("\n")
	}

	if out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput(); err != nil {
		Exitf("running %s failed: %v\n%s", argv[0], err, out)
	} else if len(out) > 0 {
		// always print external output even if the command is successful, so that we don't
		// swallow linker warnings (see https://golang.org/issue/17935).
		l.Logf("%s", out)
	}

	if !*FlagS && !*FlagW && !debug_s && Headtype == objabi.Hdarwin {
		// Skip combining dwarf on arm.
		if !SysArch.InFamily(sys.ARM, sys.ARM64) {
			dsym := filepath.Join(*flagTmpdir, "go.dwarf")
			if out, err := exec.Command("dsymutil", "-f", *flagOutfile, "-o", dsym).CombinedOutput(); err != nil {
				Exitf("%s: running dsymutil failed: %v\n%s", os.Args[0], err, out)
			}
			// Skip combining if `dsymutil` didn't generate a file. See #11994.
			if _, err := os.Stat(dsym); os.IsNotExist(err) {
				return
			}
			// For os.Rename to work reliably, must be in same directory as outfile.
			combinedOutput := *flagOutfile + "~"
			if err := machoCombineDwarf(*flagOutfile, dsym, combinedOutput); err != nil {
				Exitf("%s: combining dwarf failed: %v", os.Args[0], err)
			}
			os.Remove(*flagOutfile)
			if err := os.Rename(combinedOutput, *flagOutfile); err != nil {
				Exitf("%s: %v", os.Args[0], err)
			}
		}
	}
}

// hostlinkArchArgs returns arguments to pass to the external linker
// based on the architecture.
func hostlinkArchArgs() []string {
	switch SysArch.Family {
	case sys.I386:
		return []string{"-m32"}
	case sys.AMD64, sys.PPC64, sys.S390X:
		return []string{"-m64"}
	case sys.ARM:
		return []string{"-marm"}
	case sys.ARM64:
		// nothing needed
	case sys.MIPS64:
		return []string{"-mabi=64"}
	case sys.MIPS:
		return []string{"-mabi=32"}
	}
	return nil
}

// ldobj loads an input object. If it is a host object (an object
// compiled by a non-Go compiler) it returns the Hostobj pointer. If
// it is a Go object, it returns nil.
func ldobj(ctxt *Link, f *bio.Reader, lib *Library, length int64, pn string, file string, whence int) *Hostobj {
	pkg := objabi.PathToPrefix(lib.Pkg)

	eof := f.Offset() + length
	start := f.Offset()
	c1 := bgetc(f)
	c2 := bgetc(f)
	c3 := bgetc(f)
	c4 := bgetc(f)
	f.Seek(start, 0)

	magic := uint32(c1)<<24 | uint32(c2)<<16 | uint32(c3)<<8 | uint32(c4)
	if magic == 0x7f454c46 { // \x7F E L F
		return ldhostobj(ldelf, f, pkg, length, pn, file)
	}

	if magic&^1 == 0xfeedface || magic&^0x01000000 == 0xcefaedfe {
		return ldhostobj(ldmacho, f, pkg, length, pn, file)
	}

	if c1 == 0x4c && c2 == 0x01 || c1 == 0x64 && c2 == 0x86 {
		return ldhostobj(ldpe, f, pkg, length, pn, file)
	}

	/* check the header */
	line, err := f.ReadString('\n')
	if err != nil {
		Errorf(nil, "truncated object file: %s: %v", pn, err)
		return nil
	}

	if !strings.HasPrefix(line, "go object ") {
		if strings.HasSuffix(pn, ".go") {
			Exitf("%s: uncompiled .go source file", pn)
			return nil
		}

		if line == SysArch.Name {
			// old header format: just $GOOS
			Errorf(nil, "%s: stale object file", pn)
			return nil
		}

		Errorf(nil, "%s: not an object file", pn)
		return nil
	}

	// First, check that the basic GOOS, GOARCH, and Version match.
	t := fmt.Sprintf("%s %s %s ", objabi.GOOS, objabi.GOARCH, objabi.Version)

	line = strings.TrimRight(line, "\n")
	if !strings.HasPrefix(line[10:]+" ", t) && !*flagF {
		Errorf(nil, "%s: object is [%s] expected [%s]", pn, line[10:], t)
		return nil
	}

	// Second, check that longer lines match each other exactly,
	// so that the Go compiler and write additional information
	// that must be the same from run to run.
	if len(line) >= len(t)+10 {
		if theline == "" {
			theline = line[10:]
		} else if theline != line[10:] {
			Errorf(nil, "%s: object is [%s] expected [%s]", pn, line[10:], theline)
			return nil
		}
	}

	/* skip over exports and other info -- ends with \n!\n */
	import0 := f.Offset()

	c1 = '\n' // the last line ended in \n
	c2 = bgetc(f)
	c3 = bgetc(f)
	for c1 != '\n' || c2 != '!' || c3 != '\n' {
		c1 = c2
		c2 = c3
		c3 = bgetc(f)
		if c3 == -1 {
			Errorf(nil, "truncated object file: %s", pn)
			return nil
		}
	}

	import1 := f.Offset()

	f.Seek(import0, 0)
	ldpkg(ctxt, f, pkg, import1-import0-2, pn, whence) // -2 for !\n
	f.Seek(import1, 0)

	LoadObjFile(ctxt, f, lib, eof-f.Offset(), pn)
	return nil
}

func readelfsymboldata(ctxt *Link, f *elf.File, sym *elf.Symbol) []byte {
	data := make([]byte, sym.Size)
	sect := f.Sections[sym.Section]
	if sect.Type != elf.SHT_PROGBITS && sect.Type != elf.SHT_NOTE {
		Errorf(nil, "reading %s from non-data section", sym.Name)
	}
	n, err := sect.ReadAt(data, int64(sym.Value-sect.Addr))
	if uint64(n) != sym.Size {
		Errorf(nil, "reading contents of %s: %v", sym.Name, err)
	}
	return data
}

func readwithpad(r io.Reader, sz int32) ([]byte, error) {
	data := make([]byte, Rnd(int64(sz), 4))
	_, err := io.ReadFull(r, data)
	if err != nil {
		return nil, err
	}
	data = data[:sz]
	return data, nil
}

func readnote(f *elf.File, name []byte, typ int32) ([]byte, error) {
	for _, sect := range f.Sections {
		if sect.Type != elf.SHT_NOTE {
			continue
		}
		r := sect.Open()
		for {
			var namesize, descsize, noteType int32
			err := binary.Read(r, f.ByteOrder, &namesize)
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("read namesize failed: %v", err)
			}
			err = binary.Read(r, f.ByteOrder, &descsize)
			if err != nil {
				return nil, fmt.Errorf("read descsize failed: %v", err)
			}
			err = binary.Read(r, f.ByteOrder, &noteType)
			if err != nil {
				return nil, fmt.Errorf("read type failed: %v", err)
			}
			noteName, err := readwithpad(r, namesize)
			if err != nil {
				return nil, fmt.Errorf("read name failed: %v", err)
			}
			desc, err := readwithpad(r, descsize)
			if err != nil {
				return nil, fmt.Errorf("read desc failed: %v", err)
			}
			if string(name) == string(noteName) && typ == noteType {
				return desc, nil
			}
		}
	}
	return nil, nil
}

func findshlib(ctxt *Link, shlib string) string {
	for _, libdir := range ctxt.Libdir {
		libpath := filepath.Join(libdir, shlib)
		if _, err := os.Stat(libpath); err == nil {
			return libpath
		}
	}
	Errorf(nil, "cannot find shared library: %s", shlib)
	return ""
}

func ldshlibsyms(ctxt *Link, shlib string) {
	libpath := findshlib(ctxt, shlib)
	if libpath == "" {
		return
	}
	for _, processedlib := range ctxt.Shlibs {
		if processedlib.Path == libpath {
			return
		}
	}
	if ctxt.Debugvlog > 1 {
		ctxt.Logf("%5.2f ldshlibsyms: found library with name %s at %s\n", Cputime(), shlib, libpath)
	}

	f, err := elf.Open(libpath)
	if err != nil {
		Errorf(nil, "cannot open shared library: %s", libpath)
		return
	}

	hash, err := readnote(f, ELF_NOTE_GO_NAME, ELF_NOTE_GOABIHASH_TAG)
	if err != nil {
		Errorf(nil, "cannot read ABI hash from shared library %s: %v", libpath, err)
		return
	}

	depsbytes, err := readnote(f, ELF_NOTE_GO_NAME, ELF_NOTE_GODEPS_TAG)
	if err != nil {
		Errorf(nil, "cannot read dep list from shared library %s: %v", libpath, err)
		return
	}
	deps := strings.Split(string(depsbytes), "\n")

	syms, err := f.DynamicSymbols()
	if err != nil {
		Errorf(nil, "cannot read symbols from shared library: %s", libpath)
		return
	}
	gcdataLocations := make(map[uint64]*Symbol)
	for _, elfsym := range syms {
		if elf.ST_TYPE(elfsym.Info) == elf.STT_NOTYPE || elf.ST_TYPE(elfsym.Info) == elf.STT_SECTION {
			continue
		}
		lsym := ctxt.Syms.Lookup(elfsym.Name, 0)
		// Because loadlib above loads all .a files before loading any shared
		// libraries, any non-dynimport symbols we find that duplicate symbols
		// already loaded should be ignored (the symbols from the .a files
		// "win").
		if lsym.Type != 0 && lsym.Type != SDYNIMPORT {
			continue
		}
		lsym.Type = SDYNIMPORT
		lsym.ElfType = elf.ST_TYPE(elfsym.Info)
		lsym.Size = int64(elfsym.Size)
		if elfsym.Section != elf.SHN_UNDEF {
			// Set .File for the library that actually defines the symbol.
			lsym.File = libpath
			// The decodetype_* functions in decodetype.go need access to
			// the type data.
			if strings.HasPrefix(lsym.Name, "type.") && !strings.HasPrefix(lsym.Name, "type..") {
				lsym.P = readelfsymboldata(ctxt, f, &elfsym)
				gcdataLocations[elfsym.Value+2*uint64(SysArch.PtrSize)+8+1*uint64(SysArch.PtrSize)] = lsym
			}
		}
	}
	gcdataAddresses := make(map[*Symbol]uint64)
	if SysArch.Family == sys.ARM64 {
		for _, sect := range f.Sections {
			if sect.Type == elf.SHT_RELA {
				var rela elf.Rela64
				rdr := sect.Open()
				for {
					err := binary.Read(rdr, f.ByteOrder, &rela)
					if err == io.EOF {
						break
					} else if err != nil {
						Errorf(nil, "reading relocation failed %v", err)
						return
					}
					t := elf.R_AARCH64(rela.Info & 0xffff)
					if t != elf.R_AARCH64_RELATIVE {
						continue
					}
					if lsym, ok := gcdataLocations[rela.Off]; ok {
						gcdataAddresses[lsym] = uint64(rela.Addend)
					}
				}
			}
		}
	}

	ctxt.Shlibs = append(ctxt.Shlibs, Shlib{Path: libpath, Hash: hash, Deps: deps, File: f, gcdataAddresses: gcdataAddresses})
}

func addsection(seg *Segment, name string, rwx int) *Section {
	sect := new(Section)
	sect.Rwx = uint8(rwx)
	sect.Name = name
	sect.Seg = seg
	sect.Align = int32(SysArch.PtrSize) // everything is at least pointer-aligned
	seg.Sections = append(seg.Sections, sect)
	return sect
}

func Le16(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1])<<8
}

func Le32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func Le64(b []byte) uint64 {
	return uint64(Le32(b)) | uint64(Le32(b[4:]))<<32
}

func Be16(b []byte) uint16 {
	return uint16(b[0])<<8 | uint16(b[1])
}

func Be32(b []byte) uint32 {
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}

type chain struct {
	sym   *Symbol
	up    *chain
	limit int // limit on entry to sym
}

var morestack *Symbol

// TODO: Record enough information in new object files to
// allow stack checks here.

func haslinkregister(ctxt *Link) bool {
	return ctxt.FixedFrameSize() != 0
}

func callsize(ctxt *Link) int {
	if haslinkregister(ctxt) {
		return 0
	}
	return SysArch.RegSize
}

func (ctxt *Link) dostkcheck() {
	var ch chain

	morestack = ctxt.Syms.Lookup("runtime.morestack", 0)

	// Every splitting function ensures that there are at least StackLimit
	// bytes available below SP when the splitting prologue finishes.
	// If the splitting function calls F, then F begins execution with
	// at least StackLimit - callsize() bytes available.
	// Check that every function behaves correctly with this amount
	// of stack, following direct calls in order to piece together chains
	// of non-splitting functions.
	ch.up = nil

	ch.limit = objabi.StackLimit - callsize(ctxt)

	// Check every function, but do the nosplit functions in a first pass,
	// to make the printed failure chains as short as possible.
	for _, s := range ctxt.Textp {
		// runtime.racesymbolizethunk is called from gcc-compiled C
		// code running on the operating system thread stack.
		// It uses more than the usual amount of stack but that's okay.
		if s.Name == "runtime.racesymbolizethunk" {
			continue
		}

		if s.Attr.NoSplit() {
			ch.sym = s
			stkcheck(ctxt, &ch, 0)
		}
	}

	for _, s := range ctxt.Textp {
		if !s.Attr.NoSplit() {
			ch.sym = s
			stkcheck(ctxt, &ch, 0)
		}
	}
}

func stkcheck(ctxt *Link, up *chain, depth int) int {
	limit := up.limit
	s := up.sym

	// Don't duplicate work: only need to consider each
	// function at top of safe zone once.
	top := limit == objabi.StackLimit-callsize(ctxt)
	if top {
		if s.Attr.StackCheck() {
			return 0
		}
		s.Attr |= AttrStackCheck
	}

	if depth > 100 {
		Errorf(s, "nosplit stack check too deep")
		stkbroke(ctxt, up, 0)
		return -1
	}

	if s.Attr.External() || s.FuncInfo == nil {
		// external function.
		// should never be called directly.
		// onlyctxt.Diagnose the direct caller.
		// TODO(mwhudson): actually think about this.
		if depth == 1 && s.Type != SXREF && !ctxt.DynlinkingGo() &&
			Buildmode != BuildmodeCArchive && Buildmode != BuildmodePIE && Buildmode != BuildmodeCShared && Buildmode != BuildmodePlugin {

			Errorf(s, "call to external function")
		}
		return -1
	}

	if limit < 0 {
		stkbroke(ctxt, up, limit)
		return -1
	}

	// morestack looks like it calls functions,
	// but it switches the stack pointer first.
	if s == morestack {
		return 0
	}

	var ch chain
	ch.up = up

	if !s.Attr.NoSplit() {
		// Ensure we have enough stack to call morestack.
		ch.limit = limit - callsize(ctxt)
		ch.sym = morestack
		if stkcheck(ctxt, &ch, depth+1) < 0 {
			return -1
		}
		if !top {
			return 0
		}
		// Raise limit to allow frame.
		locals := int32(0)
		if s.FuncInfo != nil {
			locals = s.FuncInfo.Locals
		}
		limit = int(objabi.StackLimit+locals) + int(ctxt.FixedFrameSize())
	}

	// Walk through sp adjustments in function, consuming relocs.
	ri := 0

	endr := len(s.R)
	var ch1 chain
	var pcsp Pciter
	var r *Reloc
	for pciterinit(ctxt, &pcsp, &s.FuncInfo.Pcsp); pcsp.done == 0; pciternext(&pcsp) {
		// pcsp.value is in effect for [pcsp.pc, pcsp.nextpc).

		// Check stack size in effect for this span.
		if int32(limit)-pcsp.value < 0 {
			stkbroke(ctxt, up, int(int32(limit)-pcsp.value))
			return -1
		}

		// Process calls in this span.
		for ; ri < endr && uint32(s.R[ri].Off) < pcsp.nextpc; ri++ {
			r = &s.R[ri]
			switch r.Type {
			// Direct call.
			case objabi.R_CALL, objabi.R_CALLARM, objabi.R_CALLARM64, objabi.R_CALLPOWER, objabi.R_CALLMIPS:
				ch.limit = int(int32(limit) - pcsp.value - int32(callsize(ctxt)))
				ch.sym = r.Sym
				if stkcheck(ctxt, &ch, depth+1) < 0 {
					return -1
				}

			// Indirect call. Assume it is a call to a splitting function,
			// so we have to make sure it can call morestack.
			// Arrange the data structures to report both calls, so that
			// if there is an error, stkprint shows all the steps involved.
			case objabi.R_CALLIND:
				ch.limit = int(int32(limit) - pcsp.value - int32(callsize(ctxt)))

				ch.sym = nil
				ch1.limit = ch.limit - callsize(ctxt) // for morestack in called prologue
				ch1.up = &ch
				ch1.sym = morestack
				if stkcheck(ctxt, &ch1, depth+2) < 0 {
					return -1
				}
			}
		}
	}

	return 0
}

func stkbroke(ctxt *Link, ch *chain, limit int) {
	Errorf(ch.sym, "nosplit stack overflow")
	stkprint(ctxt, ch, limit)
}

func stkprint(ctxt *Link, ch *chain, limit int) {
	var name string

	if ch.sym != nil {
		name = ch.sym.Name
		if ch.sym.Attr.NoSplit() {
			name += " (nosplit)"
		}
	} else {
		name = "function pointer"
	}

	if ch.up == nil {
		// top of chain.  ch->sym != nil.
		if ch.sym.Attr.NoSplit() {
			fmt.Printf("\t%d\tassumed on entry to %s\n", ch.limit, name)
		} else {
			fmt.Printf("\t%d\tguaranteed after split check in %s\n", ch.limit, name)
		}
	} else {
		stkprint(ctxt, ch.up, ch.limit+callsize(ctxt))
		if !haslinkregister(ctxt) {
			fmt.Printf("\t%d\ton entry to %s\n", ch.limit, name)
		}
	}

	if ch.limit != limit {
		fmt.Printf("\t%d\tafter %s uses %d\n", limit, name, ch.limit-limit)
	}
}

func Cflush() {
	if err := coutbuf.w.Flush(); err != nil {
		Exitf("flushing %s: %v", coutbuf.f.Name(), err)
	}
}

func Cseek(p int64) {
	if p == coutbuf.off {
		return
	}
	Cflush()
	if _, err := coutbuf.f.Seek(p, 0); err != nil {
		Exitf("seeking in output [0, 1]: %v", err)
	}
	coutbuf.off = p
}

func Cwritestring(s string) {
	coutbuf.WriteString(s)
}

func Cwrite(p []byte) {
	coutbuf.Write(p)
}

func Cput(c uint8) {
	coutbuf.w.WriteByte(c)
	coutbuf.off++
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: link [options] main.o\n")
	objabi.Flagprint(2)
	Exit(2)
}

func doversion() {
	Exitf("version %s", objabi.Version)
}

type SymbolType int8

const (
	TextSym      SymbolType = 'T'
	DataSym                 = 'D'
	BSSSym                  = 'B'
	UndefinedSym            = 'U'
	TLSSym                  = 't'
	FileSym                 = 'f'
	FrameSym                = 'm'
	ParamSym                = 'p'
	AutoSym                 = 'a'
)

func genasmsym(ctxt *Link, put func(*Link, *Symbol, string, SymbolType, int64, *Symbol)) {
	// These symbols won't show up in the first loop below because we
	// skip STEXT symbols. Normal STEXT symbols are emitted by walking textp.
	s := ctxt.Syms.Lookup("runtime.text", 0)
	if s.Type == STEXT {
		put(ctxt, s, s.Name, TextSym, s.Value, nil)
	}

	n := 0

	// Generate base addresses for all text sections if there are multiple
	for _, sect := range Segtext.Sections {
		if n == 0 {
			n++
			continue
		}
		if sect.Name != ".text" {
			break
		}
		s = ctxt.Syms.ROLookup(fmt.Sprintf("runtime.text.%d", n), 0)
		if s == nil {
			break
		}
		if s.Type == STEXT {
			put(ctxt, s, s.Name, TextSym, s.Value, nil)
		}
		n++
	}

	s = ctxt.Syms.Lookup("runtime.etext", 0)
	if s.Type == STEXT {
		put(ctxt, s, s.Name, TextSym, s.Value, nil)
	}

	for _, s := range ctxt.Syms.Allsym {
		if s.Attr.NotInSymbolTable() {
			continue
		}
		if (s.Name == "" || s.Name[0] == '.') && s.Version == 0 && s.Name != ".rathole" && s.Name != ".TOC." {
			continue
		}
		switch s.Type & SMASK {
		case SCONST,
			SRODATA,
			SSYMTAB,
			SPCLNTAB,
			SINITARR,
			SDATA,
			SNOPTRDATA,
			SELFROSECT,
			SMACHOGOT,
			STYPE,
			SSTRING,
			SGOSTRING,
			SGOFUNC,
			SGCBITS,
			STYPERELRO,
			SSTRINGRELRO,
			SGOSTRINGRELRO,
			SGOFUNCRELRO,
			SGCBITSRELRO,
			SRODATARELRO,
			STYPELINK,
			SITABLINK,
			SWINDOWS:
			if !s.Attr.Reachable() {
				continue
			}
			put(ctxt, s, s.Name, DataSym, Symaddr(s), s.Gotype)

		case SBSS, SNOPTRBSS:
			if !s.Attr.Reachable() {
				continue
			}
			if len(s.P) > 0 {
				Errorf(s, "should not be bss (size=%d type=%v special=%v)", len(s.P), s.Type, s.Attr.Special())
			}
			put(ctxt, s, s.Name, BSSSym, Symaddr(s), s.Gotype)

		case SFILE:
			put(ctxt, nil, s.Name, FileSym, s.Value, nil)

		case SHOSTOBJ:
			if Headtype == objabi.Hwindows || Iself {
				put(ctxt, s, s.Name, UndefinedSym, s.Value, nil)
			}

		case SDYNIMPORT:
			if !s.Attr.Reachable() {
				continue
			}
			put(ctxt, s, s.Extname, UndefinedSym, 0, nil)

		case STLSBSS:
			if Linkmode == LinkExternal {
				put(ctxt, s, s.Name, TLSSym, Symaddr(s), s.Gotype)
			}
		}
	}

	var off int32
	for _, s := range ctxt.Textp {
		put(ctxt, s, s.Name, TextSym, s.Value, s.Gotype)

		locals := int32(0)
		if s.FuncInfo != nil {
			locals = s.FuncInfo.Locals
		}
		// NOTE(ality): acid can't produce a stack trace without .frame symbols
		put(ctxt, nil, ".frame", FrameSym, int64(locals)+int64(SysArch.PtrSize), nil)

		if s.FuncInfo == nil {
			continue
		}
		for _, a := range s.FuncInfo.Autom {
			// Emit a or p according to actual offset, even if label is wrong.
			// This avoids negative offsets, which cannot be encoded.
			if a.Name != objabi.A_AUTO && a.Name != objabi.A_PARAM {
				continue
			}

			// compute offset relative to FP
			if a.Name == objabi.A_PARAM {
				off = a.Aoffset
			} else {
				off = a.Aoffset - int32(SysArch.PtrSize)
			}

			// FP
			if off >= 0 {
				put(ctxt, nil, a.Asym.Name, ParamSym, int64(off), a.Gotype)
				continue
			}

			// SP
			if off <= int32(-SysArch.PtrSize) {
				put(ctxt, nil, a.Asym.Name, AutoSym, -(int64(off) + int64(SysArch.PtrSize)), a.Gotype)
				continue
			}
			// Otherwise, off is addressing the saved program counter.
			// Something underhanded is going on. Say nothing.
		}
	}

	if ctxt.Debugvlog != 0 || *flagN {
		ctxt.Logf("%5.2f symsize = %d\n", Cputime(), uint32(Symsize))
	}
}

func Symaddr(s *Symbol) int64 {
	if !s.Attr.Reachable() {
		Errorf(s, "unreachable symbol in symaddr")
	}
	return s.Value
}

func (ctxt *Link) xdefine(p string, t SymKind, v int64) {
	s := ctxt.Syms.Lookup(p, 0)
	s.Type = t
	s.Value = v
	s.Attr |= AttrReachable
	s.Attr |= AttrSpecial
	s.Attr |= AttrLocal
}

func datoff(s *Symbol, addr int64) int64 {
	if uint64(addr) >= Segdata.Vaddr {
		return int64(uint64(addr) - Segdata.Vaddr + Segdata.Fileoff)
	}
	if uint64(addr) >= Segtext.Vaddr {
		return int64(uint64(addr) - Segtext.Vaddr + Segtext.Fileoff)
	}
	Errorf(s, "invalid datoff %#x", addr)
	return 0
}

func Entryvalue(ctxt *Link) int64 {
	a := *flagEntrySymbol
	if a[0] >= '0' && a[0] <= '9' {
		return atolwhex(a)
	}
	s := ctxt.Syms.Lookup(a, 0)
	if s.Type == 0 {
		return *FlagTextAddr
	}
	if s.Type != STEXT {
		Errorf(s, "entry not text")
	}
	return s.Value
}

func undefsym(ctxt *Link, s *Symbol) {
	var r *Reloc

	for i := 0; i < len(s.R); i++ {
		r = &s.R[i]
		if r.Sym == nil { // happens for some external ARM relocs
			continue
		}
		if r.Sym.Type == Sxxx || r.Sym.Type == SXREF {
			Errorf(s, "undefined: %q", r.Sym.Name)
		}
		if !r.Sym.Attr.Reachable() && r.Type != objabi.R_WEAKADDROFF {
			Errorf(s, "relocation target %q", r.Sym.Name)
		}
	}
}

func (ctxt *Link) undef() {
	for _, s := range ctxt.Textp {
		undefsym(ctxt, s)
	}
	for _, s := range datap {
		undefsym(ctxt, s)
	}
	if nerrors > 0 {
		errorexit()
	}
}

func (ctxt *Link) callgraph() {
	if !*FlagC {
		return
	}

	var i int
	var r *Reloc
	for _, s := range ctxt.Textp {
		for i = 0; i < len(s.R); i++ {
			r = &s.R[i]
			if r.Sym == nil {
				continue
			}
			if (r.Type == objabi.R_CALL || r.Type == objabi.R_CALLARM || r.Type == objabi.R_CALLPOWER || r.Type == objabi.R_CALLMIPS) && r.Sym.Type == STEXT {
				ctxt.Logf("%s calls %s\n", s.Name, r.Sym.Name)
			}
		}
	}
}

func Rnd(v int64, r int64) int64 {
	if r <= 0 {
		return v
	}
	v += r - 1
	c := v % r
	if c < 0 {
		c += r
	}
	v -= c
	return v
}

func bgetc(r *bio.Reader) int {
	c, err := r.ReadByte()
	if err != nil {
		if err != io.EOF {
			log.Fatalf("reading input: %v", err)
		}
		return -1
	}
	return int(c)
}

type markKind uint8 // for postorder traversal
const (
	unvisited markKind = iota
	visiting
	visited
)

func postorder(libs []*Library) []*Library {
	order := make([]*Library, 0, len(libs)) // hold the result
	mark := make(map[*Library]markKind, len(libs))
	for _, lib := range libs {
		dfs(lib, mark, &order)
	}
	return order
}

func dfs(lib *Library, mark map[*Library]markKind, order *[]*Library) {
	if mark[lib] == visited {
		return
	}
	if mark[lib] == visiting {
		panic("found import cycle while visiting " + lib.Pkg)
	}
	mark[lib] = visiting
	for _, i := range lib.imports {
		dfs(i, mark, order)
	}
	mark[lib] = visited
	*order = append(*order, lib)
}
