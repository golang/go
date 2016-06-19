// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
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
	"cmd/internal/obj"
	"cmd/internal/sys"
	"log"
	"strconv"
)

var headers = []struct {
	name string
	val  int
}{
	{"darwin", obj.Hdarwin},
	{"dragonfly", obj.Hdragonfly},
	{"freebsd", obj.Hfreebsd},
	{"linux", obj.Hlinux},
	{"android", obj.Hlinux}, // must be after "linux" entry or else headstr(Hlinux) == "android"
	{"nacl", obj.Hnacl},
	{"netbsd", obj.Hnetbsd},
	{"openbsd", obj.Hopenbsd},
	{"plan9", obj.Hplan9},
	{"solaris", obj.Hsolaris},
	{"windows", obj.Hwindows},
	{"windowsgui", obj.Hwindows},
}

func linknew(arch *sys.Arch) *Link {
	ctxt := &Link{
		Hash: []map[string]*LSym{
			// preallocate about 2mb for hash of
			// non static symbols
			make(map[string]*LSym, 100000),
		},
		Allsym: make([]*LSym, 0, 100000),
		Arch:   arch,
		Goroot: obj.Getgoroot(),
	}

	p := obj.Getgoarch()
	if p != arch.Name {
		log.Fatalf("invalid goarch %s (want %s)", p, arch.Name)
	}

	ctxt.Headtype = headtype(obj.Getgoos())
	if ctxt.Headtype < 0 {
		log.Fatalf("unknown goos %s", obj.Getgoos())
	}

	// Record thread-local storage offset.
	// TODO(rsc): Move tlsoffset back into the linker.
	switch ctxt.Headtype {
	default:
		log.Fatalf("unknown thread-local storage offset for %s", Headstr(ctxt.Headtype))

	case obj.Hplan9, obj.Hwindows:
		break

		/*
		 * ELF uses TLS offset negative from FS.
		 * Translate 0(FS) and 8(FS) into -16(FS) and -8(FS).
		 * Known to low-level assembly in package runtime and runtime/cgo.
		 */
	case obj.Hlinux,
		obj.Hfreebsd,
		obj.Hnetbsd,
		obj.Hopenbsd,
		obj.Hdragonfly,
		obj.Hsolaris:
		if obj.Getgoos() == "android" {
			switch ctxt.Arch.Family {
			case sys.AMD64:
				// Android/amd64 constant - offset from 0(FS) to our TLS slot.
				// Explained in src/runtime/cgo/gcc_android_*.c
				ctxt.Tlsoffset = 0x1d0
			case sys.I386:
				// Android/386 constant - offset from 0(GS) to our TLS slot.
				ctxt.Tlsoffset = 0xf8
			default:
				ctxt.Tlsoffset = -1 * ctxt.Arch.PtrSize
			}
		} else {
			ctxt.Tlsoffset = -1 * ctxt.Arch.PtrSize
		}

	case obj.Hnacl:
		switch ctxt.Arch.Family {
		default:
			log.Fatalf("unknown thread-local storage offset for nacl/%s", ctxt.Arch.Name)

		case sys.ARM:
			ctxt.Tlsoffset = 0

		case sys.AMD64:
			ctxt.Tlsoffset = 0

		case sys.I386:
			ctxt.Tlsoffset = -8
		}

		/*
		 * OS X system constants - offset from 0(GS) to our TLS.
		 * Explained in src/runtime/cgo/gcc_darwin_*.c.
		 */
	case obj.Hdarwin:
		switch ctxt.Arch.Family {
		default:
			log.Fatalf("unknown thread-local storage offset for darwin/%s", ctxt.Arch.Name)

		case sys.ARM:
			ctxt.Tlsoffset = 0 // dummy value, not needed

		case sys.AMD64:
			ctxt.Tlsoffset = 0x8a0

		case sys.ARM64:
			ctxt.Tlsoffset = 0 // dummy value, not needed

		case sys.I386:
			ctxt.Tlsoffset = 0x468
		}
	}

	// On arm, record goarm.
	if ctxt.Arch.Family == sys.ARM {
		ctxt.Goarm = obj.Getgoarm()
	}

	return ctxt
}

func linknewsym(ctxt *Link, name string, v int) *LSym {
	batch := ctxt.LSymBatch
	if len(batch) == 0 {
		batch = make([]LSym, 1000)
	}
	s := &batch[0]
	ctxt.LSymBatch = batch[1:]

	s.Dynid = -1
	s.Plt = -1
	s.Got = -1
	s.Name = name
	s.Version = int16(v)
	ctxt.Allsym = append(ctxt.Allsym, s)

	return s
}

func Linklookup(ctxt *Link, name string, v int) *LSym {
	m := ctxt.Hash[v]
	s := m[name]
	if s != nil {
		return s
	}
	s = linknewsym(ctxt, name, v)
	s.Extname = s.Name
	m[name] = s
	return s
}

// read-only lookup
func Linkrlookup(ctxt *Link, name string, v int) *LSym {
	return ctxt.Hash[v][name]
}

func Headstr(v int) string {
	for i := 0; i < len(headers); i++ {
		if v == headers[i].val {
			return headers[i].name
		}
	}
	return strconv.Itoa(v)
}

func headtype(name string) int {
	for i := 0; i < len(headers); i++ {
		if name == headers[i].name {
			return headers[i].val
		}
	}
	return -1
}
