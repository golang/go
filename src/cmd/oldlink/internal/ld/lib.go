// Inferno utils/8l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/8l/asm.c
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
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/oldlink/internal/loadelf"
	"cmd/oldlink/internal/loader"
	"cmd/oldlink/internal/loadmacho"
	"cmd/oldlink/internal/loadpe"
	"cmd/oldlink/internal/loadxcoff"
	"cmd/oldlink/internal/objfile"
	"cmd/oldlink/internal/sym"
	"crypto/sha1"
	"debug/elf"
	"debug/macho"
	"encoding/base64"
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
	"sort"
	"strings"
	"sync"
)

// Data layout and relocation.

// Derived from Inferno utils/6l/l.h
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/l.h
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
	Funcalign      int
	Maxalign       int
	Minalign       int
	Dwarfregsp     int
	Dwarfreglr     int
	Androiddynld   string
	Linuxdynld     string
	Freebsddynld   string
	Netbsddynld    string
	Openbsddynld   string
	Dragonflydynld string
	Solarisdynld   string
	Adddynrel      func(*Link, *sym.Symbol, *sym.Reloc) bool
	Archinit       func(*Link)
	// Archreloc is an arch-specific hook that assists in
	// relocation processing (invoked by 'relocsym'); it handles
	// target-specific relocation tasks. Here "rel" is the current
	// relocation being examined, "sym" is the symbol containing the
	// chunk of data to which the relocation applies, and "off" is the
	// contents of the to-be-relocated data item (from sym.P). Return
	// value is the appropriately relocated value (to be written back
	// to the same spot in sym.P) and a boolean indicating
	// success/failure (a failing value indicates a fatal error).
	Archreloc func(link *Link, rel *sym.Reloc, sym *sym.Symbol,
		offset int64) (relocatedOffset int64, success bool)
	// Archrelocvariant is a second arch-specific hook used for
	// relocation processing; it handles relocations where r.Type is
	// insufficient to describe the relocation (r.Variant !=
	// sym.RV_NONE). Here "rel" is the relocation being applied, "sym"
	// is the symbol containing the chunk of data to which the
	// relocation applies, and "off" is the contents of the
	// to-be-relocated data item (from sym.P). Return is an updated
	// offset value.
	Archrelocvariant func(link *Link, rel *sym.Reloc, sym *sym.Symbol,
		offset int64) (relocatedOffset int64)
	Trampoline func(*Link, *sym.Reloc, *sym.Symbol)

	// Asmb and Asmb2 are arch-specific routines that write the output
	// file. Typically, Asmb writes most of the content (sections and
	// segments), for which we have computed the size and offset. Asmb2
	// writes the rest.
	Asmb  func(*Link)
	Asmb2 func(*Link)

	Elfreloc1   func(*Link, *sym.Reloc, int64) bool
	Elfsetupplt func(*Link)
	Gentext     func(*Link)
	Machoreloc1 func(*sys.Arch, *OutBuf, *sym.Symbol, *sym.Reloc, int64) bool
	PEreloc1    func(*sys.Arch, *OutBuf, *sym.Symbol, *sym.Reloc, int64) bool
	Xcoffreloc1 func(*sys.Arch, *OutBuf, *sym.Symbol, *sym.Reloc, int64) bool

	// TLSIEtoLE converts a TLS Initial Executable relocation to
	// a TLS Local Executable relocation.
	//
	// This is possible when a TLS IE relocation refers to a local
	// symbol in an executable, which is typical when internally
	// linking PIE binaries.
	TLSIEtoLE func(s *sym.Symbol, off, size int)

	// optional override for assignAddress
	AssignAddress func(ctxt *Link, sect *sym.Section, n int, s *sym.Symbol, va uint64, isTramp bool) (*sym.Section, int, uint64)
}

var (
	thearch Arch
	Lcsize  int32
	rpath   Rpath
	Spsize  int32
	Symsize int32
)

const (
	MINFUNC = 16 // minimum size for a function
)

// DynlinkingGo reports whether we are producing Go code that can live
// in separate shared libraries linked together at runtime.
func (ctxt *Link) DynlinkingGo() bool {
	if !ctxt.Loaded {
		panic("DynlinkingGo called before all symbols loaded")
	}
	return ctxt.BuildMode == BuildModeShared || ctxt.linkShared || ctxt.BuildMode == BuildModePlugin || ctxt.canUsePlugins
}

// CanUsePlugins reports whether a plugins can be used
func (ctxt *Link) CanUsePlugins() bool {
	if !ctxt.Loaded {
		panic("CanUsePlugins called before all symbols loaded")
	}
	return ctxt.canUsePlugins
}

// UseRelro reports whether to make use of "read only relocations" aka
// relro.
func (ctxt *Link) UseRelro() bool {
	switch ctxt.BuildMode {
	case BuildModeCArchive, BuildModeCShared, BuildModeShared, BuildModePIE, BuildModePlugin:
		return ctxt.IsELF || ctxt.HeadType == objabi.Haix
	default:
		return ctxt.linkShared || (ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal)
	}
}

var (
	dynexp          []*sym.Symbol
	dynlib          []string
	ldflag          []string
	havedynamic     int
	Funcalign       int
	iscgo           bool
	elfglobalsymndx int
	interpreter     string

	debug_s bool // backup old value of debug['s']
	HEADR   int32

	nerrors  int
	liveness int64

	// See -strictdups command line flag.
	checkStrictDups   int // 0=off 1=warning 2=error
	strictDupMsgCount int
)

var (
	Segtext      sym.Segment
	Segrodata    sym.Segment
	Segrelrodata sym.Segment
	Segdata      sym.Segment
	Segdwarf     sym.Segment
)

const pkgdef = "__.PKGDEF"

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
	Funcalign = thearch.Funcalign

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
	f, err := os.OpenFile(*flagOutfile, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0775)
	if err != nil {
		Exitf("cannot create %s: %v", *flagOutfile, err)
	}

	ctxt.Out.w = bufio.NewWriter(f)
	ctxt.Out.f = f

	if *flagEntrySymbol == "" {
		switch ctxt.BuildMode {
		case BuildModeCShared, BuildModeCArchive:
			*flagEntrySymbol = fmt.Sprintf("_rt0_%s_%s_lib", objabi.GOARCH, objabi.GOOS)
		case BuildModeExe, BuildModePIE:
			*flagEntrySymbol = fmt.Sprintf("_rt0_%s_%s", objabi.GOARCH, objabi.GOOS)
		case BuildModeShared, BuildModePlugin:
			// No *flagEntrySymbol for -buildmode=shared and plugin
		default:
			Errorf(nil, "unknown *flagEntrySymbol for buildmode %v", ctxt.BuildMode)
		}
	}
}

func exitIfErrors() {
	if nerrors != 0 || checkStrictDups > 1 && strictDupMsgCount > 0 {
		mayberemoveoutfile()
		Exit(2)
	}

}

func errorexit() {
	exitIfErrors()
	Exit(0)
}

func loadinternal(ctxt *Link, name string) *sym.Library {
	if ctxt.linkShared && ctxt.PackageShlib != nil {
		if shlib := ctxt.PackageShlib[name]; shlib != "" {
			return addlibpath(ctxt, "internal", "internal", "", name, shlib)
		}
	}
	if ctxt.PackageFile != nil {
		if pname := ctxt.PackageFile[name]; pname != "" {
			return addlibpath(ctxt, "internal", "internal", pname, name, "")
		}
		ctxt.Logf("loadinternal: cannot find %s\n", name)
		return nil
	}

	for _, libdir := range ctxt.Libdir {
		if ctxt.linkShared {
			shlibname := filepath.Join(libdir, name+".shlibname")
			if ctxt.Debugvlog != 0 {
				ctxt.Logf("searching for %s.a in %s\n", name, shlibname)
			}
			if _, err := os.Stat(shlibname); err == nil {
				return addlibpath(ctxt, "internal", "internal", "", name, shlibname)
			}
		}
		pname := filepath.Join(libdir, name+".a")
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

// extld returns the current external linker.
func (ctxt *Link) extld() string {
	if *flagExtld == "" {
		*flagExtld = "gcc"
	}
	return *flagExtld
}

// findLibPathCmd uses cmd command to find gcc library libname.
// It returns library full path if found, or "none" if not found.
func (ctxt *Link) findLibPathCmd(cmd, libname string) string {
	extld := ctxt.extld()
	args := hostlinkArchArgs(ctxt.Arch)
	args = append(args, cmd)
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%s %v\n", extld, args)
	}
	out, err := exec.Command(extld, args...).Output()
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
	if *flagNewobj {
		var flags uint32
		switch *FlagStrictDups {
		case 0:
			// nothing to do
		case 1, 2:
			flags = loader.FlagStrictDups
		default:
			log.Fatalf("invalid -strictdups flag value %d", *FlagStrictDups)
		}
		ctxt.loader = loader.NewLoader(flags)
	}

	ctxt.cgo_export_static = make(map[string]bool)
	ctxt.cgo_export_dynamic = make(map[string]bool)

	// ctxt.Library grows during the loop, so not a range loop.
	i := 0
	for ; i < len(ctxt.Library); i++ {
		lib := ctxt.Library[i]
		if lib.Shlib == "" {
			if ctxt.Debugvlog > 1 {
				ctxt.Logf("autolib: %s (from %s)\n", lib.File, lib.Objref)
			}
			loadobjfile(ctxt, lib)
		}
	}

	// load internal packages, if not already
	if *flagRace {
		loadinternal(ctxt, "runtime/race")
	}
	if *flagMsan {
		loadinternal(ctxt, "runtime/msan")
	}
	loadinternal(ctxt, "runtime")
	for ; i < len(ctxt.Library); i++ {
		lib := ctxt.Library[i]
		if lib.Shlib == "" {
			loadobjfile(ctxt, lib)
		}
	}

	if *flagNewobj {
		iscgo = ctxt.loader.Lookup("x_cgo_init", 0) != 0
		ctxt.canUsePlugins = ctxt.loader.Lookup("plugin.Open", sym.SymVerABIInternal) != 0
	} else {
		iscgo = ctxt.Syms.ROLookup("x_cgo_init", 0) != nil
		ctxt.canUsePlugins = ctxt.Syms.ROLookup("plugin.Open", sym.SymVerABIInternal) != nil
	}

	// We now have enough information to determine the link mode.
	determineLinkMode(ctxt)

	if ctxt.LinkMode == LinkExternal && !iscgo && ctxt.LibraryByPkg["runtime/cgo"] == nil && !(objabi.GOOS == "darwin" && ctxt.BuildMode != BuildModePlugin && ctxt.Arch.Family == sys.AMD64) {
		// This indicates a user requested -linkmode=external.
		// The startup code uses an import of runtime/cgo to decide
		// whether to initialize the TLS.  So give it one. This could
		// be handled differently but it's an unusual case.
		if lib := loadinternal(ctxt, "runtime/cgo"); lib != nil {
			if lib.Shlib != "" {
				ldshlibsyms(ctxt, lib.Shlib)
			} else {
				if ctxt.BuildMode == BuildModeShared || ctxt.linkShared {
					Exitf("cannot implicitly include runtime/cgo in a shared library")
				}
				loadobjfile(ctxt, lib)
			}
		}
	}

	for _, lib := range ctxt.Library {
		if lib.Shlib != "" {
			if ctxt.Debugvlog > 1 {
				ctxt.Logf("autolib: %s (from %s)\n", lib.Shlib, lib.Objref)
			}
			ldshlibsyms(ctxt, lib.Shlib)
		}
	}

	if ctxt.LinkMode == LinkInternal && len(hostobj) != 0 {
		if *flagNewobj {
			// In newobj mode, we typically create sym.Symbols later therefore
			// also set cgo attributes later. However, for internal cgo linking,
			// the host object loaders still work with sym.Symbols (for now),
			// and they need cgo attributes set to work properly. So process
			// them now.
			lookup := func(name string, ver int) *sym.Symbol { return ctxt.loader.LookupOrCreate(name, ver, ctxt.Syms) }
			for _, d := range ctxt.cgodata {
				setCgoAttr(ctxt, lookup, d.file, d.pkg, d.directives)
			}
			ctxt.cgodata = nil
		}

		// Drop all the cgo_import_static declarations.
		// Turns out we won't be needing them.
		for _, s := range ctxt.Syms.Allsym {
			if s.Type == sym.SHOSTOBJ {
				// If a symbol was marked both
				// cgo_import_static and cgo_import_dynamic,
				// then we want to make it cgo_import_dynamic
				// now.
				if s.Extname() != "" && s.Dynimplib() != "" && !s.Attr.CgoExport() {
					s.Type = sym.SDYNIMPORT
				} else {
					s.Type = 0
				}
			}
		}
	}

	// Conditionally load host objects, or setup for external linking.
	hostobjs(ctxt)
	hostlinksetup(ctxt)

	if *flagNewobj {
		// Add references of externally defined symbols.
		ctxt.loader.LoadRefs(ctxt.Arch, ctxt.Syms)
	}

	// Now that we know the link mode, set the dynexp list.
	if !*flagNewobj { // set this later in newobj mode
		setupdynexp(ctxt)
	}

	if ctxt.LinkMode == LinkInternal && len(hostobj) != 0 {
		// If we have any undefined symbols in external
		// objects, try to read them from the libgcc file.
		any := false
		for _, s := range ctxt.Syms.Allsym {
			for i := range s.R {
				r := &s.R[i] // Copying sym.Reloc has measurable impact on performance
				if r.Sym != nil && r.Sym.Type == sym.SXREF && r.Sym.Name != ".got" {
					any = true
					break
				}
			}
		}
		if any {
			if *flagLibGCC == "" {
				*flagLibGCC = ctxt.findLibPathCmd("--print-libgcc-file-name", "libgcc")
			}
			if runtime.GOOS == "openbsd" && *flagLibGCC == "libgcc.a" {
				// On OpenBSD `clang --print-libgcc-file-name` returns "libgcc.a".
				// In this case we fail to load libgcc.a and can encounter link
				// errors - see if we can find libcompiler_rt.a instead.
				*flagLibGCC = ctxt.findLibPathCmd("--print-file-name=libcompiler_rt.a", "libcompiler_rt")
			}
			if *flagLibGCC != "none" {
				hostArchive(ctxt, *flagLibGCC)
			}
			if ctxt.HeadType == objabi.Hwindows {
				if p := ctxt.findLibPath("libmingwex.a"); p != "none" {
					hostArchive(ctxt, p)
				}
				if p := ctxt.findLibPath("libmingw32.a"); p != "none" {
					hostArchive(ctxt, p)
				}
				// Link libmsvcrt.a to resolve '__acrt_iob_func' symbol
				// (see https://golang.org/issue/23649 for details).
				if p := ctxt.findLibPath("libmsvcrt.a"); p != "none" {
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
	}

	// We've loaded all the code now.
	ctxt.Loaded = true

	importcycles()

	if *flagNewobj {
		strictDupMsgCount = ctxt.loader.NStrictDupMsgs()
	}
}

// Set up dynexp list.
func setupdynexp(ctxt *Link) {
	dynexpMap := ctxt.cgo_export_dynamic
	if ctxt.LinkMode == LinkExternal {
		dynexpMap = ctxt.cgo_export_static
	}
	dynexp = make([]*sym.Symbol, 0, len(dynexpMap))
	for exp := range dynexpMap {
		s := ctxt.Syms.Lookup(exp, 0)
		dynexp = append(dynexp, s)
	}
	sort.Sort(byName(dynexp))

	// Resolve ABI aliases in the list of cgo-exported functions.
	// This is necessary because we load the ABI0 symbol for all
	// cgo exports.
	for i, s := range dynexp {
		if s.Type != sym.SABIALIAS {
			continue
		}
		t := resolveABIAlias(s)
		t.Attr |= s.Attr
		t.SetExtname(s.Extname())
		dynexp[i] = t
	}

	ctxt.cgo_export_static = nil
	ctxt.cgo_export_dynamic = nil
}

// Set up flags and special symbols depending on the platform build mode.
func (ctxt *Link) linksetup() {
	switch ctxt.BuildMode {
	case BuildModeCShared, BuildModePlugin:
		s := ctxt.Syms.Lookup("runtime.islibrary", 0)
		s.Type = sym.SNOPTRDATA
		s.Attr |= sym.AttrDuplicateOK
		s.AddUint8(1)
	case BuildModeCArchive:
		s := ctxt.Syms.Lookup("runtime.isarchive", 0)
		s.Type = sym.SNOPTRDATA
		s.Attr |= sym.AttrDuplicateOK
		s.AddUint8(1)
	}

	// Recalculate pe parameters now that we have ctxt.LinkMode set.
	if ctxt.HeadType == objabi.Hwindows {
		Peinit(ctxt)
	}

	if ctxt.HeadType == objabi.Hdarwin && ctxt.LinkMode == LinkExternal {
		*FlagTextAddr = 0
	}

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
	if ctxt.BuildMode == BuildModeExe {
		if havedynamic == 0 && ctxt.HeadType != objabi.Hdarwin && ctxt.HeadType != objabi.Hsolaris {
			*FlagD = true
		}
	}

	if ctxt.LinkMode == LinkExternal && ctxt.Arch.Family == sys.PPC64 && objabi.GOOS != "aix" {
		toc := ctxt.Syms.Lookup(".TOC.", 0)
		toc.Type = sym.SDYNIMPORT
	}

	// The Android Q linker started to complain about underalignment of the our TLS
	// section. We don't actually use the section on android, so dont't
	// generate it.
	if objabi.GOOS != "android" {
		tlsg := ctxt.Syms.Lookup("runtime.tlsg", 0)

		// runtime.tlsg is used for external linking on platforms that do not define
		// a variable to hold g in assembly (currently only intel).
		if tlsg.Type == 0 {
			tlsg.Type = sym.STLSBSS
			tlsg.Size = int64(ctxt.Arch.PtrSize)
		} else if tlsg.Type != sym.SDYNIMPORT {
			Errorf(nil, "runtime declared tlsg variable %v", tlsg.Type)
		}
		tlsg.Attr |= sym.AttrReachable
		ctxt.Tlsg = tlsg
	}

	var moduledata *sym.Symbol
	if ctxt.BuildMode == BuildModePlugin {
		moduledata = ctxt.Syms.Lookup("local.pluginmoduledata", 0)
		moduledata.Attr |= sym.AttrLocal
	} else {
		moduledata = ctxt.Syms.Lookup("runtime.firstmoduledata", 0)
	}
	if moduledata.Type != 0 && moduledata.Type != sym.SDYNIMPORT {
		// If the module (toolchain-speak for "executable or shared
		// library") we are linking contains the runtime package, it
		// will define the runtime.firstmoduledata symbol and we
		// truncate it back to 0 bytes so we can define its entire
		// contents in symtab.go:symtab().
		moduledata.Size = 0

		// In addition, on ARM, the runtime depends on the linker
		// recording the value of GOARM.
		if ctxt.Arch.Family == sys.ARM {
			s := ctxt.Syms.Lookup("runtime.goarm", 0)
			s.Type = sym.SDATA
			s.Size = 0
			s.AddUint8(uint8(objabi.GOARM))
		}

		if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
			s := ctxt.Syms.Lookup("runtime.framepointer_enabled", 0)
			s.Type = sym.SDATA
			s.Size = 0
			s.AddUint8(1)
		}
	} else {
		// If OTOH the module does not contain the runtime package,
		// create a local symbol for the moduledata.
		moduledata = ctxt.Syms.Lookup("local.moduledata", 0)
		moduledata.Attr |= sym.AttrLocal
	}
	// In all cases way we mark the moduledata as noptrdata to hide it from
	// the GC.
	moduledata.Type = sym.SNOPTRDATA
	moduledata.Attr |= sym.AttrReachable
	ctxt.Moduledata = moduledata

	// If package versioning is required, generate a hash of the
	// packages used in the link.
	if ctxt.BuildMode == BuildModeShared || ctxt.BuildMode == BuildModePlugin || ctxt.CanUsePlugins() {
		for _, lib := range ctxt.Library {
			if lib.Shlib == "" {
				genhash(ctxt, lib)
			}
		}
	}

	if ctxt.Arch == sys.Arch386 && ctxt.HeadType != objabi.Hwindows {
		if (ctxt.BuildMode == BuildModeCArchive && ctxt.IsELF) || ctxt.BuildMode == BuildModeCShared || ctxt.BuildMode == BuildModePIE || ctxt.DynlinkingGo() {
			got := ctxt.Syms.Lookup("_GLOBAL_OFFSET_TABLE_", 0)
			got.Type = sym.SDYNIMPORT
			got.Attr |= sym.AttrReachable
		}
	}
}

// mangleTypeSym shortens the names of symbols that represent Go types
// if they are visible in the symbol table.
//
// As the names of these symbols are derived from the string of
// the type, they can run to many kilobytes long. So we shorten
// them using a SHA-1 when the name appears in the final binary.
// This also removes characters that upset external linkers.
//
// These are the symbols that begin with the prefix 'type.' and
// contain run-time type information used by the runtime and reflect
// packages. All Go binaries contain these symbols, but only
// those programs loaded dynamically in multiple parts need these
// symbols to have entries in the symbol table.
func (ctxt *Link) mangleTypeSym() {
	if ctxt.BuildMode != BuildModeShared && !ctxt.linkShared && ctxt.BuildMode != BuildModePlugin && !ctxt.CanUsePlugins() {
		return
	}

	for _, s := range ctxt.Syms.Allsym {
		newName := typeSymbolMangle(s.Name)
		if newName != s.Name {
			ctxt.Syms.Rename(s.Name, newName, int(s.Version), ctxt.Reachparent)
		}
	}
}

// typeSymbolMangle mangles the given symbol name into something shorter.
//
// Keep the type.. prefix, which parts of the linker (like the
// DWARF generator) know means the symbol is not decodable.
// Leave type.runtime. symbols alone, because other parts of
// the linker manipulates them.
func typeSymbolMangle(name string) string {
	if !strings.HasPrefix(name, "type.") {
		return name
	}
	if strings.HasPrefix(name, "type.runtime.") {
		return name
	}
	if len(name) <= 14 && !strings.Contains(name, "@") { // Issue 19529
		return name
	}
	hash := sha1.Sum([]byte(name))
	prefix := "type."
	if name[5] == '.' {
		prefix = "type.."
	}
	return prefix + base64.StdEncoding.EncodeToString(hash[:6])
}

/*
 * look for the next file in an archive.
 * adapted from libmach.
 */
func nextar(bp *bio.Reader, off int64, a *ArHdr) int64 {
	if off&1 != 0 {
		off++
	}
	bp.MustSeek(off, 0)
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

func genhash(ctxt *Link, lib *sym.Library) {
	f, err := bio.Open(lib.File)
	if err != nil {
		Errorf(nil, "cannot open file %s for hash generation: %v", lib.File, err)
		return
	}
	defer f.Close()

	var magbuf [len(ARMAG)]byte
	if _, err := io.ReadFull(f, magbuf[:]); err != nil {
		Exitf("file %s too short", lib.File)
	}

	if string(magbuf[:]) != ARMAG {
		Exitf("%s is not an archive file", lib.File)
	}

	var arhdr ArHdr
	l := nextar(f, f.Offset(), &arhdr)
	if l <= 0 {
		Errorf(nil, "%s: short read on archive file symbol header", lib.File)
		return
	}
	if arhdr.name != pkgdef {
		Errorf(nil, "%s: missing package data entry", lib.File)
		return
	}

	h := sha1.New()

	// To compute the hash of a package, we hash the first line of
	// __.PKGDEF (which contains the toolchain version and any
	// GOEXPERIMENT flags) and the export data (which is between
	// the first two occurrences of "\n$$").

	pkgDefBytes := make([]byte, atolwhex(arhdr.size))
	_, err = io.ReadFull(f, pkgDefBytes)
	if err != nil {
		Errorf(nil, "%s: error reading package data: %v", lib.File, err)
		return
	}
	firstEOL := bytes.IndexByte(pkgDefBytes, '\n')
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
	lib.Hash = hex.EncodeToString(h.Sum(nil))
}

func loadobjfile(ctxt *Link, lib *sym.Library) {
	pkg := objabi.PathToPrefix(lib.Pkg)

	if ctxt.Debugvlog > 1 {
		ctxt.Logf("ldobj: %s (%s)\n", lib.File, pkg)
	}
	f, err := bio.Open(lib.File)
	if err != nil {
		Exitf("cannot open file %s: %v", lib.File, err)
	}
	defer f.Close()
	defer func() {
		if pkg == "main" && !lib.Main {
			Exitf("%s: not package main", lib.File)
		}

		// Ideally, we'd check that *all* object files within
		// the archive were marked safe, but here we settle
		// for *any*.
		//
		// Historically, cmd/link only checked the __.PKGDEF
		// file, which in turn came from the first object
		// file, typically produced by cmd/compile. The
		// remaining object files are normally produced by
		// cmd/asm, which doesn't support marking files as
		// safe anyway. So at least in practice, this matches
		// how safe mode has always worked.
		if *flagU && !lib.Safe {
			Exitf("%s: load of unsafe package %s", lib.File, pkg)
		}
	}()

	for i := 0; i < len(ARMAG); i++ {
		if c, err := f.ReadByte(); err == nil && c == ARMAG[i] {
			continue
		}

		/* load it as a regular file */
		l := f.MustSeek(0, 2)
		f.MustSeek(0, 0)
		ldobj(ctxt, f, lib, l, lib.File, lib.File)
		return
	}

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
	var arhdr ArHdr
	off := f.Offset()
	for {
		l := nextar(f, off, &arhdr)
		if l == 0 {
			break
		}
		if l < 0 {
			Exitf("%s: malformed archive", lib.File)
		}
		off += l

		// __.PKGDEF isn't a real Go object file, and it's
		// absent in -linkobj builds anyway. Skipping it
		// ensures consistency between -linkobj and normal
		// build modes.
		if arhdr.name == pkgdef {
			continue
		}

		// Skip other special (non-object-file) sections that
		// build tools may have added. Such sections must have
		// short names so that the suffix is not truncated.
		if len(arhdr.name) < 16 {
			if ext := filepath.Ext(arhdr.name); ext != ".o" && ext != ".syso" {
				continue
			}
		}

		pname := fmt.Sprintf("%s(%s)", lib.File, arhdr.name)
		l = atolwhex(arhdr.size)
		ldobj(ctxt, f, lib, l, pname, lib.File)
	}
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

func ldhostobj(ld func(*Link, *bio.Reader, string, int64, string), headType objabi.HeadType, f *bio.Reader, pkg string, length int64, pn string, file string) *Hostobj {
	isinternal := false
	for _, intpkg := range internalpkg {
		if pkg == intpkg {
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
	if headType == objabi.Hdragonfly {
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
	if ctxt.LinkMode != LinkInternal {
		return
	}
	var h *Hostobj

	for i := 0; i < len(hostobj); i++ {
		h = &hostobj[i]
		f, err := bio.Open(h.file)
		if err != nil {
			Exitf("cannot reopen %s: %v", h.pn, err)
		}

		f.MustSeek(h.off, 0)
		h.ld(ctxt, f, h.pkg, h.length, h.pn)
		f.Close()
	}
}

func hostlinksetup(ctxt *Link) {
	if ctxt.LinkMode != LinkExternal {
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
		ownTmpDir = true
		AtExit(func() {
			ctxt.Out.f.Close()
			os.RemoveAll(*flagTmpdir)
		})
	}

	// change our output to temporary object file
	ctxt.Out.f.Close()
	mayberemoveoutfile()

	p := filepath.Join(*flagTmpdir, "go.o")
	var err error
	f, err := os.OpenFile(p, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0775)
	if err != nil {
		Exitf("cannot create %s: %v", p, err)
	}

	ctxt.Out.w = bufio.NewWriter(f)
	ctxt.Out.f = f
	ctxt.Out.off = 0
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
			defer f.Close()
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
	if ctxt.BuildMode != BuildModeCArchive {
		return
	}

	exitIfErrors()

	if *flagExtar == "" {
		*flagExtar = "ar"
	}

	mayberemoveoutfile()

	// Force the buffer to flush here so that external
	// tools will see a complete file.
	ctxt.Out.Flush()
	if err := ctxt.Out.f.Close(); err != nil {
		Exitf("close: %v", err)
	}
	ctxt.Out.f = nil

	argv := []string{*flagExtar, "-q", "-c", "-s"}
	if ctxt.HeadType == objabi.Haix {
		argv = append(argv, "-X64")
	}
	argv = append(argv, *flagOutfile)
	argv = append(argv, filepath.Join(*flagTmpdir, "go.o"))
	argv = append(argv, hostobjCopy()...)

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("archive: %s\n", strings.Join(argv, " "))
	}

	// If supported, use syscall.Exec() to invoke the archive command,
	// which should be the final remaining step needed for the link.
	// This will reduce peak RSS for the link (and speed up linking of
	// large applications), since when the archive command runs we
	// won't be holding onto all of the linker's live memory.
	if syscallExecSupported && !ownTmpDir {
		runAtExitFuncs()
		ctxt.execArchive(argv)
		panic("should not get here")
	}

	// Otherwise invoke 'ar' in the usual way (fork + exec).
	if out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput(); err != nil {
		Exitf("running %s failed: %v\n%s", argv[0], err, out)
	}
}

func (ctxt *Link) hostlink() {
	if ctxt.LinkMode != LinkExternal || nerrors > 0 {
		return
	}
	if ctxt.BuildMode == BuildModeCArchive {
		return
	}

	var argv []string
	argv = append(argv, ctxt.extld())
	argv = append(argv, hostlinkArchArgs(ctxt.Arch)...)

	if *FlagS || debug_s {
		if ctxt.HeadType == objabi.Hdarwin {
			// Recent versions of macOS print
			//	ld: warning: option -s is obsolete and being ignored
			// so do not pass any arguments.
		} else {
			argv = append(argv, "-s")
		}
	}

	switch ctxt.HeadType {
	case objabi.Hdarwin:
		if machoPlatform == PLATFORM_MACOS {
			// -headerpad is incompatible with -fembed-bitcode.
			argv = append(argv, "-Wl,-headerpad,1144")
		}
		if ctxt.DynlinkingGo() && !ctxt.Arch.InFamily(sys.ARM, sys.ARM64) {
			argv = append(argv, "-Wl,-flat_namespace")
		}
	case objabi.Hopenbsd:
		argv = append(argv, "-Wl,-nopie")
	case objabi.Hwindows:
		if windowsgui {
			argv = append(argv, "-mwindows")
		} else {
			argv = append(argv, "-mconsole")
		}
		// Mark as having awareness of terminal services, to avoid
		// ancient compatibility hacks.
		argv = append(argv, "-Wl,--tsaware")

		// Enable DEP
		argv = append(argv, "-Wl,--nxcompat")

		argv = append(argv, fmt.Sprintf("-Wl,--major-os-version=%d", PeMinimumTargetMajorVersion))
		argv = append(argv, fmt.Sprintf("-Wl,--minor-os-version=%d", PeMinimumTargetMinorVersion))
		argv = append(argv, fmt.Sprintf("-Wl,--major-subsystem-version=%d", PeMinimumTargetMajorVersion))
		argv = append(argv, fmt.Sprintf("-Wl,--minor-subsystem-version=%d", PeMinimumTargetMinorVersion))
	case objabi.Haix:
		argv = append(argv, "-pthread")
		// prevent ld to reorder .text functions to keep the same
		// first/last functions for moduledata.
		argv = append(argv, "-Wl,-bnoobjreorder")
		// mcmodel=large is needed for every gcc generated files, but
		// ld still need -bbigtoc in order to allow larger TOC.
		argv = append(argv, "-mcmodel=large")
		argv = append(argv, "-Wl,-bbigtoc")
	}

	switch ctxt.BuildMode {
	case BuildModeExe:
		if ctxt.HeadType == objabi.Hdarwin {
			if machoPlatform == PLATFORM_MACOS {
				argv = append(argv, "-Wl,-no_pie")
				argv = append(argv, "-Wl,-pagezero_size,4000000")
			}
		}
	case BuildModePIE:
		switch ctxt.HeadType {
		case objabi.Hdarwin, objabi.Haix:
		case objabi.Hwindows:
			// Enable ASLR.
			argv = append(argv, "-Wl,--dynamicbase")
			// enable high-entropy ASLR on 64-bit.
			if ctxt.Arch.PtrSize >= 8 {
				argv = append(argv, "-Wl,--high-entropy-va")
			}
			// Work around binutils limitation that strips relocation table for dynamicbase.
			// See https://sourceware.org/bugzilla/show_bug.cgi?id=19011
			argv = append(argv, "-Wl,--export-all-symbols")
		default:
			// ELF.
			if ctxt.UseRelro() {
				argv = append(argv, "-Wl,-z,relro")
			}
			argv = append(argv, "-pie")
		}
	case BuildModeCShared:
		if ctxt.HeadType == objabi.Hdarwin {
			argv = append(argv, "-dynamiclib")
			if ctxt.Arch.Family != sys.AMD64 {
				argv = append(argv, "-Wl,-read_only_relocs,suppress")
			}
		} else {
			// ELF.
			argv = append(argv, "-Wl,-Bsymbolic")
			if ctxt.UseRelro() {
				argv = append(argv, "-Wl,-z,relro")
			}
			argv = append(argv, "-shared")
			if ctxt.HeadType != objabi.Hwindows {
				// Pass -z nodelete to mark the shared library as
				// non-closeable: a dlclose will do nothing.
				argv = append(argv, "-Wl,-z,nodelete")
			}
		}
	case BuildModeShared:
		if ctxt.UseRelro() {
			argv = append(argv, "-Wl,-z,relro")
		}
		argv = append(argv, "-shared")
	case BuildModePlugin:
		if ctxt.HeadType == objabi.Hdarwin {
			argv = append(argv, "-dynamiclib")
		} else {
			if ctxt.UseRelro() {
				argv = append(argv, "-Wl,-z,relro")
			}
			argv = append(argv, "-shared")
		}
	}

	if ctxt.IsELF && ctxt.DynlinkingGo() {
		// We force all symbol resolution to be done at program startup
		// because lazy PLT resolution can use large amounts of stack at
		// times we cannot allow it to do so.
		argv = append(argv, "-Wl,-znow")

		// Do not let the host linker generate COPY relocations. These
		// can move symbols out of sections that rely on stable offsets
		// from the beginning of the section (like sym.STYPE).
		argv = append(argv, "-Wl,-znocopyreloc")

		if ctxt.Arch.InFamily(sys.ARM, sys.ARM64) && objabi.GOOS == "linux" {
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

	if ctxt.Arch.Family == sys.ARM64 && objabi.GOOS == "freebsd" {
		// Switch to ld.bfd on freebsd/arm64.
		argv = append(argv, "-fuse-ld=bfd")

		// Provide a useful error if ld.bfd is missing.
		cmd := exec.Command(*flagExtld, "-fuse-ld=bfd", "-Wl,--version")
		if out, err := cmd.CombinedOutput(); err == nil {
			if !bytes.Contains(out, []byte("GNU ld")) {
				log.Fatalf("ARM64 external linker must be ld.bfd (issue #35197), please install devel/binutils")
			}
		}
	}

	if ctxt.IsELF && len(buildinfo) > 0 {
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
	if ctxt.IsELF {
		argv = append(argv, "-rdynamic")
	}
	if ctxt.HeadType == objabi.Haix {
		fileName := xcoffCreateExportFile(ctxt)
		argv = append(argv, "-Wl,-bE:"+fileName)
	}

	if strings.Contains(argv[0], "clang") {
		argv = append(argv, "-Qunused-arguments")
	}

	const compressDWARF = "-Wl,--compress-debug-sections=zlib-gnu"
	if ctxt.compressDWARF && linkerFlagSupported(argv[0], compressDWARF) {
		argv = append(argv, compressDWARF)
	}

	argv = append(argv, filepath.Join(*flagTmpdir, "go.o"))
	argv = append(argv, hostobjCopy()...)
	if ctxt.HeadType == objabi.Haix {
		// We want to have C files after Go files to remove
		// trampolines csects made by ld.
		argv = append(argv, "-nostartfiles")
		argv = append(argv, "/lib/crt0_64.o")

		extld := ctxt.extld()
		// Get starting files.
		getPathFile := func(file string) string {
			args := []string{"-maix64", "--print-file-name=" + file}
			out, err := exec.Command(extld, args...).CombinedOutput()
			if err != nil {
				log.Fatalf("running %s failed: %v\n%s", extld, err, out)
			}
			return strings.Trim(string(out), "\n")
		}
		argv = append(argv, getPathFile("crtcxa.o"))
		argv = append(argv, getPathFile("crtdbase.o"))
	}

	if ctxt.linkShared {
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
		for _, shlib := range ctxt.Shlibs {
			addshlib(shlib.Path)
			for _, dep := range shlib.Deps {
				if dep == "" {
					continue
				}
				libpath := findshlib(ctxt, dep)
				if libpath != "" {
					addshlib(libpath)
				}
			}
		}
	}

	// clang, unlike GCC, passes -rdynamic to the linker
	// even when linking with -static, causing a linker
	// error when using GNU ld. So take out -rdynamic if
	// we added it. We do it in this order, rather than
	// only adding -rdynamic later, so that -*extldflags
	// can override -rdynamic without using -static.
	checkStatic := func(arg string) {
		if ctxt.IsELF && arg == "-static" {
			for i := range argv {
				if argv[i] == "-rdynamic" {
					argv[i] = "-static"
				}
			}
		}
	}

	for _, p := range ldflag {
		argv = append(argv, p)
		checkStatic(p)
	}

	// When building a program with the default -buildmode=exe the
	// gc compiler generates code requires DT_TEXTREL in a
	// position independent executable (PIE). On systems where the
	// toolchain creates PIEs by default, and where DT_TEXTREL
	// does not work, the resulting programs will not run. See
	// issue #17847. To avoid this problem pass -no-pie to the
	// toolchain if it is supported.
	if ctxt.BuildMode == BuildModeExe && !ctxt.linkShared {
		// GCC uses -no-pie, clang uses -nopie.
		for _, nopie := range []string{"-no-pie", "-nopie"} {
			if linkerFlagSupported(argv[0], nopie) {
				argv = append(argv, nopie)
				break
			}
		}
	}

	for _, p := range strings.Fields(*flagExtldflags) {
		argv = append(argv, p)
		checkStatic(p)
	}
	if ctxt.HeadType == objabi.Hwindows {
		// use gcc linker script to work around gcc bug
		// (see https://golang.org/issue/20183 for details).
		p := writeGDBLinkerScript()
		argv = append(argv, "-Wl,-T,"+p)
		// libmingw32 and libmingwex have some inter-dependencies,
		// so must use linker groups.
		argv = append(argv, "-Wl,--start-group", "-lmingwex", "-lmingw32", "-Wl,--end-group")
		argv = append(argv, peimporteddlls()...)
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("host link:")
		for _, v := range argv {
			ctxt.Logf(" %q", v)
		}
		ctxt.Logf("\n")
	}

	out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput()
	if err != nil {
		Exitf("running %s failed: %v\n%s", argv[0], err, out)
	}

	// Filter out useless linker warnings caused by bugs outside Go.
	// See also cmd/go/internal/work/exec.go's gccld method.
	var save [][]byte
	var skipLines int
	for _, line := range bytes.SplitAfter(out, []byte("\n")) {
		// golang.org/issue/26073 - Apple Xcode bug
		if bytes.Contains(line, []byte("ld: warning: text-based stub file")) {
			continue
		}

		if skipLines > 0 {
			skipLines--
			continue
		}

		// Remove TOC overflow warning on AIX.
		if bytes.Contains(line, []byte("ld: 0711-783")) {
			skipLines = 2
			continue
		}

		save = append(save, line)
	}
	out = bytes.Join(save, nil)

	if len(out) > 0 {
		// always print external output even if the command is successful, so that we don't
		// swallow linker warnings (see https://golang.org/issue/17935).
		ctxt.Logf("%s", out)
	}

	if !*FlagS && !*FlagW && !debug_s && ctxt.HeadType == objabi.Hdarwin {
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
		exef, err := os.Open(*flagOutfile)
		if err != nil {
			Exitf("%s: combining dwarf failed: %v", os.Args[0], err)
		}
		defer exef.Close()
		exem, err := macho.NewFile(exef)
		if err != nil {
			Exitf("%s: parsing Mach-O header failed: %v", os.Args[0], err)
		}
		// Only macOS supports unmapped segments such as our __DWARF segment.
		if machoPlatform == PLATFORM_MACOS {
			if err := machoCombineDwarf(ctxt, exef, exem, dsym, combinedOutput); err != nil {
				Exitf("%s: combining dwarf failed: %v", os.Args[0], err)
			}
			os.Remove(*flagOutfile)
			if err := os.Rename(combinedOutput, *flagOutfile); err != nil {
				Exitf("%s: %v", os.Args[0], err)
			}
		}
	}
}

var createTrivialCOnce sync.Once

func linkerFlagSupported(linker, flag string) bool {
	createTrivialCOnce.Do(func() {
		src := filepath.Join(*flagTmpdir, "trivial.c")
		if err := ioutil.WriteFile(src, []byte("int main() { return 0; }"), 0666); err != nil {
			Errorf(nil, "WriteFile trivial.c failed: %v", err)
		}
	})

	flagsWithNextArgSkip := []string{
		"-F",
		"-l",
		"-L",
		"-framework",
		"-Wl,-framework",
		"-Wl,-rpath",
		"-Wl,-undefined",
	}
	flagsWithNextArgKeep := []string{
		"-arch",
		"-isysroot",
		"--sysroot",
		"-target",
	}
	prefixesToKeep := []string{
		"-f",
		"-m",
		"-p",
		"-Wl,",
		"-arch",
		"-isysroot",
		"--sysroot",
		"-target",
	}

	var flags []string
	keep := false
	skip := false
	extldflags := strings.Fields(*flagExtldflags)
	for _, f := range append(extldflags, ldflag...) {
		if keep {
			flags = append(flags, f)
			keep = false
		} else if skip {
			skip = false
		} else if f == "" || f[0] != '-' {
		} else if contains(flagsWithNextArgSkip, f) {
			skip = true
		} else if contains(flagsWithNextArgKeep, f) {
			flags = append(flags, f)
			keep = true
		} else {
			for _, p := range prefixesToKeep {
				if strings.HasPrefix(f, p) {
					flags = append(flags, f)
					break
				}
			}
		}
	}

	flags = append(flags, flag, "trivial.c")

	cmd := exec.Command(linker, flags...)
	cmd.Dir = *flagTmpdir
	cmd.Env = append([]string{"LC_ALL=C"}, os.Environ()...)
	out, err := cmd.CombinedOutput()
	// GCC says "unrecognized command line option ‘-no-pie’"
	// clang says "unknown argument: '-no-pie'"
	return err == nil && !bytes.Contains(out, []byte("unrecognized")) && !bytes.Contains(out, []byte("unknown"))
}

// hostlinkArchArgs returns arguments to pass to the external linker
// based on the architecture.
func hostlinkArchArgs(arch *sys.Arch) []string {
	switch arch.Family {
	case sys.I386:
		return []string{"-m32"}
	case sys.AMD64, sys.S390X:
		return []string{"-m64"}
	case sys.ARM:
		return []string{"-marm"}
	case sys.ARM64:
		// nothing needed
	case sys.MIPS64:
		return []string{"-mabi=64"}
	case sys.MIPS:
		return []string{"-mabi=32"}
	case sys.PPC64:
		if objabi.GOOS == "aix" {
			return []string{"-maix64"}
		} else {
			return []string{"-m64"}
		}

	}
	return nil
}

// ldobj loads an input object. If it is a host object (an object
// compiled by a non-Go compiler) it returns the Hostobj pointer. If
// it is a Go object, it returns nil.
func ldobj(ctxt *Link, f *bio.Reader, lib *sym.Library, length int64, pn string, file string) *Hostobj {
	pkg := objabi.PathToPrefix(lib.Pkg)

	eof := f.Offset() + length
	start := f.Offset()
	c1 := bgetc(f)
	c2 := bgetc(f)
	c3 := bgetc(f)
	c4 := bgetc(f)
	f.MustSeek(start, 0)

	unit := &sym.CompilationUnit{Lib: lib}
	lib.Units = append(lib.Units, unit)

	magic := uint32(c1)<<24 | uint32(c2)<<16 | uint32(c3)<<8 | uint32(c4)
	if magic == 0x7f454c46 { // \x7F E L F
		if *flagNewobj {
			ldelf := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, flags, err := loadelf.Load(ctxt.loader, ctxt.Arch, ctxt.Syms, f, pkg, length, pn, ehdr.flags)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				ehdr.flags = flags
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldelf, ctxt.HeadType, f, pkg, length, pn, file)
		} else {
			ldelf := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, flags, err := loadelf.LoadOld(ctxt.Arch, ctxt.Syms, f, pkg, length, pn, ehdr.flags)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				ehdr.flags = flags
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldelf, ctxt.HeadType, f, pkg, length, pn, file)
		}
	}

	if magic&^1 == 0xfeedface || magic&^0x01000000 == 0xcefaedfe {
		if *flagNewobj {
			ldmacho := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, err := loadmacho.Load(ctxt.loader, ctxt.Arch, ctxt.Syms, f, pkg, length, pn)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldmacho, ctxt.HeadType, f, pkg, length, pn, file)
		} else {
			ldmacho := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, err := loadmacho.LoadOld(ctxt.Arch, ctxt.Syms, f, pkg, length, pn)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldmacho, ctxt.HeadType, f, pkg, length, pn, file)
		}
	}

	if c1 == 0x4c && c2 == 0x01 || c1 == 0x64 && c2 == 0x86 {
		if *flagNewobj {
			ldpe := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, rsrc, err := loadpe.Load(ctxt.loader, ctxt.Arch, ctxt.Syms, f, pkg, length, pn)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				if rsrc != nil {
					setpersrc(ctxt, rsrc)
				}
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldpe, ctxt.HeadType, f, pkg, length, pn, file)
		} else {
			ldpe := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, rsrc, err := loadpe.LoadOld(ctxt.Arch, ctxt.Syms, f, pkg, length, pn)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				if rsrc != nil {
					setpersrc(ctxt, rsrc)
				}
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldpe, ctxt.HeadType, f, pkg, length, pn, file)
		}
	}

	if c1 == 0x01 && (c2 == 0xD7 || c2 == 0xF7) {
		if *flagNewobj {
			ldxcoff := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, err := loadxcoff.Load(ctxt.loader, ctxt.Arch, ctxt.Syms, f, pkg, length, pn)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldxcoff, ctxt.HeadType, f, pkg, length, pn, file)
		} else {
			ldxcoff := func(ctxt *Link, f *bio.Reader, pkg string, length int64, pn string) {
				textp, err := loadxcoff.LoadOld(ctxt.Arch, ctxt.Syms, f, pkg, length, pn)
				if err != nil {
					Errorf(nil, "%v", err)
					return
				}
				ctxt.Textp = append(ctxt.Textp, textp...)
			}
			return ldhostobj(ldxcoff, ctxt.HeadType, f, pkg, length, pn, file)
		}
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

		if line == ctxt.Arch.Name {
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

	// Skip over exports and other info -- ends with \n!\n.
	//
	// Note: It's possible for "\n!\n" to appear within the binary
	// package export data format. To avoid truncating the package
	// definition prematurely (issue 21703), we keep track of
	// how many "$$" delimiters we've seen.

	import0 := f.Offset()

	c1 = '\n' // the last line ended in \n
	c2 = bgetc(f)
	c3 = bgetc(f)
	markers := 0
	for {
		if c1 == '\n' {
			if markers%2 == 0 && c2 == '!' && c3 == '\n' {
				break
			}
			if c2 == '$' && c3 == '$' {
				markers++
			}
		}

		c1 = c2
		c2 = c3
		c3 = bgetc(f)
		if c3 == -1 {
			Errorf(nil, "truncated object file: %s", pn)
			return nil
		}
	}

	import1 := f.Offset()

	f.MustSeek(import0, 0)
	ldpkg(ctxt, f, lib, import1-import0-2, pn) // -2 for !\n
	f.MustSeek(import1, 0)

	flags := 0
	switch *FlagStrictDups {
	case 0:
		break
	case 1:
		flags = objfile.StrictDupsWarnFlag
	case 2:
		flags = objfile.StrictDupsErrFlag
	default:
		log.Fatalf("invalid -strictdups flag value %d", *FlagStrictDups)
	}
	var c int
	if *flagNewobj {
		ctxt.loader.Preload(ctxt.Arch, ctxt.Syms, f, lib, unit, eof-f.Offset(), pn, flags)
	} else {
		c = objfile.Load(ctxt.Arch, ctxt.Syms, f, lib, unit, eof-f.Offset(), pn, flags)
	}
	strictDupMsgCount += c
	addImports(ctxt, lib, pn)
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
	if filepath.IsAbs(shlib) {
		return shlib
	}
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
	var libpath string
	if filepath.IsAbs(shlib) {
		libpath = shlib
		shlib = filepath.Base(shlib)
	} else {
		libpath = findshlib(ctxt, shlib)
		if libpath == "" {
			return
		}
	}
	for _, processedlib := range ctxt.Shlibs {
		if processedlib.Path == libpath {
			return
		}
	}
	if ctxt.Debugvlog > 1 {
		ctxt.Logf("ldshlibsyms: found library with name %s at %s\n", shlib, libpath)
	}

	f, err := elf.Open(libpath)
	if err != nil {
		Errorf(nil, "cannot open shared library: %s", libpath)
		return
	}
	// Keep the file open as decodetypeGcprog needs to read from it.
	// TODO: fix. Maybe mmap the file.
	//defer f.Close()

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
	var deps []string
	for _, dep := range strings.Split(string(depsbytes), "\n") {
		if dep == "" {
			continue
		}
		if !filepath.IsAbs(dep) {
			// If the dep can be interpreted as a path relative to the shlib
			// in which it was found, do that. Otherwise, we will leave it
			// to be resolved by libdir lookup.
			abs := filepath.Join(filepath.Dir(libpath), dep)
			if _, err := os.Stat(abs); err == nil {
				dep = abs
			}
		}
		deps = append(deps, dep)
	}

	syms, err := f.DynamicSymbols()
	if err != nil {
		Errorf(nil, "cannot read symbols from shared library: %s", libpath)
		return
	}
	gcdataLocations := make(map[uint64]*sym.Symbol)
	for _, elfsym := range syms {
		if elf.ST_TYPE(elfsym.Info) == elf.STT_NOTYPE || elf.ST_TYPE(elfsym.Info) == elf.STT_SECTION {
			continue
		}

		// Symbols whose names start with "type." are compiler
		// generated, so make functions with that prefix internal.
		ver := 0
		if elf.ST_TYPE(elfsym.Info) == elf.STT_FUNC && strings.HasPrefix(elfsym.Name, "type.") {
			ver = sym.SymVerABIInternal
		}

		var lsym *sym.Symbol
		if *flagNewobj {
			i := ctxt.loader.AddExtSym(elfsym.Name, ver)
			if i == 0 {
				continue
			}
			lsym = ctxt.Syms.Newsym(elfsym.Name, ver)
			ctxt.loader.Syms[i] = lsym
		} else {
			lsym = ctxt.Syms.Lookup(elfsym.Name, ver)
		}
		// Because loadlib above loads all .a files before loading any shared
		// libraries, any non-dynimport symbols we find that duplicate symbols
		// already loaded should be ignored (the symbols from the .a files
		// "win").
		if lsym.Type != 0 && lsym.Type != sym.SDYNIMPORT {
			continue
		}
		lsym.Type = sym.SDYNIMPORT
		lsym.SetElfType(elf.ST_TYPE(elfsym.Info))
		lsym.Size = int64(elfsym.Size)
		if elfsym.Section != elf.SHN_UNDEF {
			// Set .File for the library that actually defines the symbol.
			lsym.File = libpath
			// The decodetype_* functions in decodetype.go need access to
			// the type data.
			if strings.HasPrefix(lsym.Name, "type.") && !strings.HasPrefix(lsym.Name, "type..") {
				lsym.P = readelfsymboldata(ctxt, f, &elfsym)
				gcdataLocations[elfsym.Value+2*uint64(ctxt.Arch.PtrSize)+8+1*uint64(ctxt.Arch.PtrSize)] = lsym
			}
		}
		// For function symbols, we don't know what ABI is
		// available, so alias it under both ABIs.
		//
		// TODO(austin): This is almost certainly wrong once
		// the ABIs are actually different. We might have to
		// mangle Go function names in the .so to include the
		// ABI.
		if elf.ST_TYPE(elfsym.Info) == elf.STT_FUNC && ver == 0 {
			var alias *sym.Symbol
			if *flagNewobj {
				i := ctxt.loader.AddExtSym(elfsym.Name, sym.SymVerABIInternal)
				if i == 0 {
					continue
				}
				alias = ctxt.Syms.Newsym(elfsym.Name, sym.SymVerABIInternal)
				ctxt.loader.Syms[i] = alias
			} else {
				alias = ctxt.Syms.Lookup(elfsym.Name, sym.SymVerABIInternal)
			}
			if alias.Type != 0 {
				continue
			}
			alias.Type = sym.SABIALIAS
			alias.R = []sym.Reloc{{Sym: lsym}}
		}
	}
	gcdataAddresses := make(map[*sym.Symbol]uint64)
	if ctxt.Arch.Family == sys.ARM64 {
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

func addsection(arch *sys.Arch, seg *sym.Segment, name string, rwx int) *sym.Section {
	sect := new(sym.Section)
	sect.Rwx = uint8(rwx)
	sect.Name = name
	sect.Seg = seg
	sect.Align = int32(arch.PtrSize) // everything is at least pointer-aligned
	seg.Sections = append(seg.Sections, sect)
	return sect
}

type chain struct {
	sym   *sym.Symbol
	up    *chain
	limit int // limit on entry to sym
}

var morestack *sym.Symbol

// TODO: Record enough information in new object files to
// allow stack checks here.

func haslinkregister(ctxt *Link) bool {
	return ctxt.FixedFrameSize() != 0
}

func callsize(ctxt *Link) int {
	if haslinkregister(ctxt) {
		return 0
	}
	return ctxt.Arch.RegSize
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
	if objabi.GOARCH == "arm64" {
		// need extra 8 bytes below SP to save FP
		ch.limit -= 8
	}

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
		s.Attr |= sym.AttrStackCheck
	}

	if depth > 500 {
		Errorf(s, "nosplit stack check too deep")
		stkbroke(ctxt, up, 0)
		return -1
	}

	if s.Attr.External() || s.FuncInfo == nil {
		// external function.
		// should never be called directly.
		// onlyctxt.Diagnose the direct caller.
		// TODO(mwhudson): actually think about this.
		// TODO(khr): disabled for now. Calls to external functions can only happen on the g0 stack.
		// See the trampolines in src/runtime/sys_darwin_$ARCH.go.
		if depth == 1 && s.Type != sym.SXREF && !ctxt.DynlinkingGo() &&
			ctxt.BuildMode != BuildModeCArchive && ctxt.BuildMode != BuildModePIE && ctxt.BuildMode != BuildModeCShared && ctxt.BuildMode != BuildModePlugin {
			//Errorf(s, "call to external function")
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
		limit = objabi.StackLimit + int(locals) + int(ctxt.FixedFrameSize())
	}

	// Walk through sp adjustments in function, consuming relocs.
	ri := 0

	endr := len(s.R)
	var ch1 chain
	pcsp := obj.NewPCIter(uint32(ctxt.Arch.MinLC))
	var r *sym.Reloc
	for pcsp.Init(s.FuncInfo.Pcsp.P); !pcsp.Done; pcsp.Next() {
		// pcsp.value is in effect for [pcsp.pc, pcsp.nextpc).

		// Check stack size in effect for this span.
		if int32(limit)-pcsp.Value < 0 {
			stkbroke(ctxt, up, int(int32(limit)-pcsp.Value))
			return -1
		}

		// Process calls in this span.
		for ; ri < endr && uint32(s.R[ri].Off) < pcsp.NextPC; ri++ {
			r = &s.R[ri]
			switch {
			case r.Type.IsDirectCall():
				ch.limit = int(int32(limit) - pcsp.Value - int32(callsize(ctxt)))
				ch.sym = r.Sym
				if stkcheck(ctxt, &ch, depth+1) < 0 {
					return -1
				}

			// Indirect call. Assume it is a call to a splitting function,
			// so we have to make sure it can call morestack.
			// Arrange the data structures to report both calls, so that
			// if there is an error, stkprint shows all the steps involved.
			case r.Type == objabi.R_CALLIND:
				ch.limit = int(int32(limit) - pcsp.Value - int32(callsize(ctxt)))
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

func usage() {
	fmt.Fprintf(os.Stderr, "usage: link [options] main.o\n")
	objabi.Flagprint(os.Stderr)
	Exit(2)
}

type SymbolType int8

const (
	// see also https://9p.io/magic/man2html/1/nm
	TextSym      SymbolType = 'T'
	DataSym      SymbolType = 'D'
	BSSSym       SymbolType = 'B'
	UndefinedSym SymbolType = 'U'
	TLSSym       SymbolType = 't'
	FrameSym     SymbolType = 'm'
	ParamSym     SymbolType = 'p'
	AutoSym      SymbolType = 'a'

	// Deleted auto (not a real sym, just placeholder for type)
	DeletedAutoSym = 'x'
)

func genasmsym(ctxt *Link, put func(*Link, *sym.Symbol, string, SymbolType, int64, *sym.Symbol)) {
	// These symbols won't show up in the first loop below because we
	// skip sym.STEXT symbols. Normal sym.STEXT symbols are emitted by walking textp.
	s := ctxt.Syms.Lookup("runtime.text", 0)
	if s.Type == sym.STEXT {
		// We've already included this symbol in ctxt.Textp
		// if ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin or
		// on AIX with external linker.
		// See data.go:/textaddress
		if !(ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin) && !(ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
			put(ctxt, s, s.Name, TextSym, s.Value, nil)
		}
	}

	n := 0

	// Generate base addresses for all text sections if there are multiple
	for _, sect := range Segtext.Sections {
		if n == 0 {
			n++
			continue
		}
		if sect.Name != ".text" || (ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
			// On AIX, runtime.text.X are symbols already in the symtab.
			break
		}
		s = ctxt.Syms.ROLookup(fmt.Sprintf("runtime.text.%d", n), 0)
		if s == nil {
			break
		}
		if s.Type == sym.STEXT {
			put(ctxt, s, s.Name, TextSym, s.Value, nil)
		}
		n++
	}

	s = ctxt.Syms.Lookup("runtime.etext", 0)
	if s.Type == sym.STEXT {
		// We've already included this symbol in ctxt.Textp
		// if ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin or
		// on AIX with external linker.
		// See data.go:/textaddress
		if !(ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin) && !(ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
			put(ctxt, s, s.Name, TextSym, s.Value, nil)
		}
	}

	shouldBeInSymbolTable := func(s *sym.Symbol) bool {
		if s.Attr.NotInSymbolTable() {
			return false
		}
		if ctxt.HeadType == objabi.Haix && s.Name == ".go.buildinfo" {
			// On AIX, .go.buildinfo must be in the symbol table as
			// it has relocations.
			return true
		}
		if (s.Name == "" || s.Name[0] == '.') && !s.IsFileLocal() && s.Name != ".rathole" && s.Name != ".TOC." {
			return false
		}
		return true
	}

	for _, s := range ctxt.Syms.Allsym {
		if !shouldBeInSymbolTable(s) {
			continue
		}
		switch s.Type {
		case sym.SCONST,
			sym.SRODATA,
			sym.SSYMTAB,
			sym.SPCLNTAB,
			sym.SINITARR,
			sym.SDATA,
			sym.SNOPTRDATA,
			sym.SELFROSECT,
			sym.SMACHOGOT,
			sym.STYPE,
			sym.SSTRING,
			sym.SGOSTRING,
			sym.SGOFUNC,
			sym.SGCBITS,
			sym.STYPERELRO,
			sym.SSTRINGRELRO,
			sym.SGOSTRINGRELRO,
			sym.SGOFUNCRELRO,
			sym.SGCBITSRELRO,
			sym.SRODATARELRO,
			sym.STYPELINK,
			sym.SITABLINK,
			sym.SWINDOWS:
			if !s.Attr.Reachable() {
				continue
			}
			put(ctxt, s, s.Name, DataSym, Symaddr(s), s.Gotype)

		case sym.SBSS, sym.SNOPTRBSS, sym.SLIBFUZZER_EXTRA_COUNTER:
			if !s.Attr.Reachable() {
				continue
			}
			if len(s.P) > 0 {
				Errorf(s, "should not be bss (size=%d type=%v special=%v)", len(s.P), s.Type, s.Attr.Special())
			}
			put(ctxt, s, s.Name, BSSSym, Symaddr(s), s.Gotype)

		case sym.SUNDEFEXT:
			if ctxt.HeadType == objabi.Hwindows || ctxt.HeadType == objabi.Haix || ctxt.IsELF {
				put(ctxt, s, s.Name, UndefinedSym, s.Value, nil)
			}

		case sym.SHOSTOBJ:
			if !s.Attr.Reachable() {
				continue
			}
			if ctxt.HeadType == objabi.Hwindows || ctxt.IsELF {
				put(ctxt, s, s.Name, UndefinedSym, s.Value, nil)
			}

		case sym.SDYNIMPORT:
			if !s.Attr.Reachable() {
				continue
			}
			put(ctxt, s, s.Extname(), UndefinedSym, 0, nil)

		case sym.STLSBSS:
			if ctxt.LinkMode == LinkExternal {
				put(ctxt, s, s.Name, TLSSym, Symaddr(s), s.Gotype)
			}
		}
	}

	for _, s := range ctxt.Textp {
		put(ctxt, s, s.Name, TextSym, s.Value, s.Gotype)

		locals := int32(0)
		if s.FuncInfo != nil {
			locals = s.FuncInfo.Locals
		}
		// NOTE(ality): acid can't produce a stack trace without .frame symbols
		put(ctxt, nil, ".frame", FrameSym, int64(locals)+int64(ctxt.Arch.PtrSize), nil)

		if s.FuncInfo == nil {
			continue
		}
	}

	if ctxt.Debugvlog != 0 || *flagN {
		ctxt.Logf("symsize = %d\n", uint32(Symsize))
	}
}

func Symaddr(s *sym.Symbol) int64 {
	if !s.Attr.Reachable() {
		Errorf(s, "unreachable symbol in symaddr")
	}
	return s.Value
}

func (ctxt *Link) xdefine(p string, t sym.SymKind, v int64) {
	s := ctxt.Syms.Lookup(p, 0)
	s.Type = t
	s.Value = v
	s.Attr |= sym.AttrReachable
	s.Attr |= sym.AttrSpecial
	s.Attr |= sym.AttrLocal
}

func datoff(s *sym.Symbol, addr int64) int64 {
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
	if ctxt.HeadType != objabi.Haix && s.Type != sym.STEXT {
		Errorf(s, "entry not text")
	}
	return s.Value
}

func undefsym(ctxt *Link, s *sym.Symbol) {
	var r *sym.Reloc

	for i := 0; i < len(s.R); i++ {
		r = &s.R[i]
		if r.Sym == nil { // happens for some external ARM relocs
			continue
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (r.Sym.Type == sym.Sxxx || r.Sym.Type == sym.SXREF) && !r.Sym.Attr.VisibilityHidden() {
			Errorf(s, "undefined: %q", r.Sym.Name)
		}
		if !r.Sym.Attr.Reachable() && r.Type != objabi.R_WEAKADDROFF {
			Errorf(s, "relocation target %q", r.Sym.Name)
		}
	}
}

func (ctxt *Link) undef() {
	// undefsym performs checks (almost) identical to checks
	// that report undefined relocations in relocsym.
	// Both undefsym and relocsym can report same symbol as undefined,
	// which results in error message duplication (see #10978).
	//
	// The undef is run after Arch.Asmb and could detect some
	// programming errors there, but if object being linked is already
	// failed with errors, it is better to avoid duplicated errors.
	if nerrors > 0 {
		return
	}

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
	var r *sym.Reloc
	for _, s := range ctxt.Textp {
		for i = 0; i < len(s.R); i++ {
			r = &s.R[i]
			if r.Sym == nil {
				continue
			}
			if r.Type.IsDirectCall() && r.Sym.Type == sym.STEXT {
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
	_ markKind = iota
	visiting
	visited
)

func postorder(libs []*sym.Library) []*sym.Library {
	order := make([]*sym.Library, 0, len(libs)) // hold the result
	mark := make(map[*sym.Library]markKind, len(libs))
	for _, lib := range libs {
		dfs(lib, mark, &order)
	}
	return order
}

func dfs(lib *sym.Library, mark map[*sym.Library]markKind, order *[]*sym.Library) {
	if mark[lib] == visited {
		return
	}
	if mark[lib] == visiting {
		panic("found import cycle while visiting " + lib.Pkg)
	}
	mark[lib] = visiting
	for _, i := range lib.Imports {
		dfs(i, mark, order)
	}
	mark[lib] = visited
	*order = append(*order, lib)
}

func (ctxt *Link) loadlibfull() {
	// Load full symbol contents, resolve indexed references.
	ctxt.loader.LoadFull(ctxt.Arch, ctxt.Syms)

	// Pull the symbols out.
	ctxt.loader.ExtractSymbols(ctxt.Syms)

	// Load cgo directives.
	for _, d := range ctxt.cgodata {
		setCgoAttr(ctxt, ctxt.Syms.Lookup, d.file, d.pkg, d.directives)
	}

	setupdynexp(ctxt)

	// Populate ctxt.Reachparent if appropriate.
	if ctxt.Reachparent != nil {
		for i := 0; i < len(ctxt.loader.Reachparent); i++ {
			p := ctxt.loader.Reachparent[i]
			if p == 0 {
				continue
			}
			if p == loader.Sym(i) {
				panic("self-cycle in reachparent")
			}
			sym := ctxt.loader.Syms[i]
			psym := ctxt.loader.Syms[p]
			ctxt.Reachparent[sym] = psym
		}
	}

	// Drop the reference.
	ctxt.loader = nil
	ctxt.cgodata = nil

	addToTextp(ctxt)
}

func (ctxt *Link) dumpsyms() {
	for _, s := range ctxt.Syms.Allsym {
		fmt.Printf("%s %s %p %v %v\n", s, s.Type, s, s.Attr.Reachable(), s.Attr.OnList())
		for i := range s.R {
			fmt.Println("\t", s.R[i].Type, s.R[i].Sym)
		}
	}
}
