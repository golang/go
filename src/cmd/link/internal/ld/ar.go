// Inferno utils/include/ar.h
// http://code.google.com/p/inferno-os/source/browse/utils/include/ar.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"os"
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

// hostArchive reads an archive file holding host objects and links in
// required objects.  The general format is the same as a Go archive
// file, but it has an armap listing symbols and the objects that
// define them.  This is used for the compiler support library
// libgcc.a.
func hostArchive(name string) {
	f, err := obj.Bopenr(name)
	if err != nil {
		if os.IsNotExist(err) {
			// It's OK if we don't have a libgcc file at all.
			if Debug['v'] != 0 {
				fmt.Fprintf(&Bso, "skipping libgcc file: %v\n", err)
			}
			return
		}
		Exitf("cannot open file %s: %v", name, err)
	}
	defer obj.Bterm(f)

	magbuf := make([]byte, len(ARMAG))
	if obj.Bread(f, magbuf) != len(magbuf) {
		Exitf("file %s too short", name)
	}

	var arhdr ArHdr
	l := nextar(f, obj.Boffset(f), &arhdr)
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
		for s := Ctxt.Allsym; s != nil; s = s.Allsym {
			for _, r := range s.R {
				if r.Sym != nil && r.Sym.Type&obj.SMASK == obj.SXREF {
					if off := armap[r.Sym.Name]; off != 0 && !loaded[off] {
						load = append(load, off)
						loaded[off] = true
					}
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

			h := ldobj(f, "libgcc", l, pname, name, ArchiveObj)
			obj.Bseek(f, h.off, 0)
			h.ld(f, h.pkg, h.length, h.pn)
		}

		any = len(load) > 0
	}
}

// archiveMap is an archive symbol map: a mapping from symbol name to
// offset within the archive file.
type archiveMap map[string]uint64

// readArmap reads the archive symbol map.
func readArmap(filename string, f *obj.Biobuf, arhdr ArHdr) archiveMap {
	is64 := arhdr.name == "/SYM64/"
	wordSize := 4
	if is64 {
		wordSize = 8
	}

	l := atolwhex(arhdr.size)
	contents := make([]byte, l)
	if obj.Bread(f, contents) != int(l) {
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
		if goos == "darwin" || (goos == "windows" && goarch == "386") {
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
