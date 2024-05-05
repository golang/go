// Inferno utils/include/ar.h
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/include/ar.h
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
	"cmd/internal/bio"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"internal/binary"
	"internal/buildcfg"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	SARMAG  = 8
	SAR_HDR = 16 + 44
)

const (
	ARMAG = "!<arch>\n"
)

type ArHdr struct {
	name string
	date string
	uid  string
	gid  string
	mode string
	size string
	fmag string
}

// pruneUndefsForWindows trims the list "undefs" of currently
// outstanding unresolved symbols to remove references to DLL import
// symbols (e.g. "__imp_XXX"). In older versions of the linker, we
// would just immediately forward references from the import sym
// (__imp_XXX) to the DLL sym (XXX), but with newer compilers this
// strategy falls down in certain cases. We instead now do this
// forwarding later on as a post-processing step, and meaning that
// during the middle part of host object loading we can see a lot of
// unresolved (SXREF) import symbols. We do not, however, want to
// trigger the inclusion of an object from a host archive if the
// reference is going to be eventually forwarded to the corresponding
// SDYNIMPORT symbol, so here we strip out such refs from the undefs
// list.
func pruneUndefsForWindows(ldr *loader.Loader, undefs, froms []loader.Sym) ([]loader.Sym, []loader.Sym) {
	var newundefs []loader.Sym
	var newfroms []loader.Sym
	for _, s := range undefs {
		sname := ldr.SymName(s)
		if strings.HasPrefix(sname, "__imp_") {
			dname := sname[len("__imp_"):]
			ds := ldr.Lookup(dname, 0)
			if ds != 0 && ldr.SymType(ds) == sym.SDYNIMPORT {
				// Don't try to pull things out of a host archive to
				// satisfy this symbol.
				continue
			}
		}
		newundefs = append(newundefs, s)
		newfroms = append(newfroms, s)
	}
	return newundefs, newfroms
}

// hostArchive reads an archive file holding host objects and links in
// required objects. The general format is the same as a Go archive
// file, but it has an armap listing symbols and the objects that
// define them. This is used for the compiler support library
// libgcc.a.
func hostArchive(ctxt *Link, name string) {
	if ctxt.Debugvlog > 1 {
		ctxt.Logf("hostArchive(%s)\n", name)
	}
	f, err := bio.Open(name)
	if err != nil {
		if os.IsNotExist(err) {
			// It's OK if we don't have a libgcc file at all.
			if ctxt.Debugvlog != 0 {
				ctxt.Logf("skipping libgcc file: %v\n", err)
			}
			return
		}
		Exitf("cannot open file %s: %v", name, err)
	}
	defer f.Close()

	var magbuf [len(ARMAG)]byte
	if _, err := io.ReadFull(f, magbuf[:]); err != nil {
		Exitf("file %s too short", name)
	}

	if string(magbuf[:]) != ARMAG {
		Exitf("%s is not an archive file", name)
	}

	var arhdr ArHdr
	l := nextar(f, f.Offset(), &arhdr)
	if l <= 0 {
		Exitf("%s missing armap", name)
	}

	var armap archiveMap
	if arhdr.name == "/" || arhdr.name == "/SYM64/" {
		armap = readArmap(name, f, arhdr)
	} else {
		Exitf("%s missing armap", name)
	}

	loaded := make(map[uint64]bool)
	any := true
	for any {
		var load []uint64
		returnAllUndefs := -1
		undefs, froms := ctxt.loader.UndefinedRelocTargets(returnAllUndefs)
		if buildcfg.GOOS == "windows" {
			undefs, froms = pruneUndefsForWindows(ctxt.loader, undefs, froms)
		}
		for k, symIdx := range undefs {
			sname := ctxt.loader.SymName(symIdx)
			if off := armap[sname]; off != 0 && !loaded[off] {
				load = append(load, off)
				loaded[off] = true
				if ctxt.Debugvlog > 1 {
					ctxt.Logf("hostArchive(%s): selecting object at offset %x to resolve %s [%d] reference from %s [%d]\n", name, off, sname, symIdx, ctxt.loader.SymName(froms[k]), froms[k])
				}
			}
		}

		for _, off := range load {
			l := nextar(f, int64(off), &arhdr)
			if l <= 0 {
				Exitf("%s missing archive entry at offset %d", name, off)
			}
			pname := fmt.Sprintf("%s(%s)", name, arhdr.name)
			l = atolwhex(arhdr.size)

			pkname := filepath.Base(name)
			if i := strings.LastIndex(pkname, ".a"); i >= 0 {
				pkname = pkname[:i]
			}
			libar := sym.Library{Pkg: pkname}
			h := ldobj(ctxt, f, &libar, l, pname, name)
			if h.ld == nil {
				Errorf(nil, "%s unrecognized object file at offset %d", name, off)
				continue
			}
			f.MustSeek(h.off, 0)
			h.ld(ctxt, f, h.pkg, h.length, h.pn)
			if *flagCaptureHostObjs != "" {
				captureHostObj(h)
			}
		}

		any = len(load) > 0
	}
}

// archiveMap is an archive symbol map: a mapping from symbol name to
// offset within the archive file.
type archiveMap map[string]uint64

// readArmap reads the archive symbol map.
func readArmap(filename string, f *bio.Reader, arhdr ArHdr) archiveMap {
	is64 := arhdr.name == "/SYM64/"
	wordSize := 4
	if is64 {
		wordSize = 8
	}

	contents := make([]byte, atolwhex(arhdr.size))
	if _, err := io.ReadFull(f, contents); err != nil {
		Exitf("short read from %s", filename)
	}

	var c uint64
	if is64 {
		c = binary.BigEndian.Uint64(contents)
	} else {
		c = uint64(binary.BigEndian.Uint32(contents))
	}
	contents = contents[wordSize:]

	ret := make(archiveMap)

	names := contents[c*uint64(wordSize):]
	for i := uint64(0); i < c; i++ {
		n := 0
		for names[n] != 0 {
			n++
		}
		name := string(names[:n])
		names = names[n+1:]

		// For Mach-O and PE/386 files we strip a leading
		// underscore from the symbol name.
		if buildcfg.GOOS == "darwin" || buildcfg.GOOS == "ios" || (buildcfg.GOOS == "windows" && buildcfg.GOARCH == "386") {
			if name[0] == '_' && len(name) > 1 {
				name = name[1:]
			}
		}

		var off uint64
		if is64 {
			off = binary.BigEndian.Uint64(contents)
		} else {
			off = uint64(binary.BigEndian.Uint32(contents))
		}
		contents = contents[wordSize:]

		ret[name] = off
	}

	return ret
}
