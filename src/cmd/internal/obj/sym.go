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

package obj

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
)

func yy_isalpha(c int) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

var headers = []struct {
	name string
	val  int
}{
	struct {
		name string
		val  int
	}{"darwin", Hdarwin},
	struct {
		name string
		val  int
	}{"dragonfly", Hdragonfly},
	struct {
		name string
		val  int
	}{"elf", Helf},
	struct {
		name string
		val  int
	}{"freebsd", Hfreebsd},
	struct {
		name string
		val  int
	}{"linux", Hlinux},
	struct {
		name string
		val  int
	}{"android", Hlinux}, // must be after "linux" entry or else headstr(Hlinux) == "android"
	struct {
		name string
		val  int
	}{"nacl", Hnacl},
	struct {
		name string
		val  int
	}{"netbsd", Hnetbsd},
	struct {
		name string
		val  int
	}{"openbsd", Hopenbsd},
	struct {
		name string
		val  int
	}{"plan9", Hplan9},
	struct {
		name string
		val  int
	}{"solaris", Hsolaris},
	struct {
		name string
		val  int
	}{"windows", Hwindows},
	struct {
		name string
		val  int
	}{"windowsgui", Hwindows},
}

func headtype(name string) int {
	for i := 0; i < len(headers); i++ {
		if name == headers[i].name {
			return headers[i].val
		}
	}
	return -1
}

var headstr_buf string

func Headstr(v int) string {
	for i := 0; i < len(headers); i++ {
		if v == headers[i].val {
			return headers[i].name
		}
	}
	headstr_buf = fmt.Sprintf("%d", v)
	return headstr_buf
}

func Linknew(arch *LinkArch) *Link {
	var buf string

	linksetexp()

	ctxt := new(Link)
	ctxt.Arch = arch
	ctxt.Version = HistVersion
	ctxt.Goroot = Getgoroot()
	ctxt.Goroot_final = os.Getenv("GOROOT_FINAL")
	if runtime.GOOS == "windows" {
		// TODO(rsc): Remove ctxt.Windows and let callers use runtime.GOOS.
		ctxt.Windows = 1
	}

	buf, _ = os.Getwd()
	if buf == "" {
		buf = "/???"
	}
	buf = filepath.ToSlash(buf)

	ctxt.Pathname = buf

	ctxt.Headtype = headtype(Getgoos())
	if ctxt.Headtype < 0 {
		log.Fatalf("unknown goos %s", Getgoos())
	}

	// Record thread-local storage offset.
	// TODO(rsc): Move tlsoffset back into the linker.
	switch ctxt.Headtype {
	default:
		log.Fatalf("unknown thread-local storage offset for %s", Headstr(ctxt.Headtype))

	case Hplan9,
		Hwindows:
		break

		/*
		 * ELF uses TLS offset negative from FS.
		 * Translate 0(FS) and 8(FS) into -16(FS) and -8(FS).
		 * Known to low-level assembly in package runtime and runtime/cgo.
		 */
	case Hlinux,
		Hfreebsd,
		Hnetbsd,
		Hopenbsd,
		Hdragonfly,
		Hsolaris:
		ctxt.Tlsoffset = -2 * ctxt.Arch.Ptrsize

	case Hnacl:
		switch ctxt.Arch.Thechar {
		default:
			log.Fatalf("unknown thread-local storage offset for nacl/%s", ctxt.Arch.Name)

		case '5':
			ctxt.Tlsoffset = 0

		case '6':
			ctxt.Tlsoffset = 0

		case '8':
			ctxt.Tlsoffset = -8
		}

		/*
		 * OS X system constants - offset from 0(GS) to our TLS.
		 * Explained in ../../runtime/cgo/gcc_darwin_*.c.
		 */
	case Hdarwin:
		switch ctxt.Arch.Thechar {
		default:
			log.Fatalf("unknown thread-local storage offset for darwin/%s", ctxt.Arch.Name)

		case '6':
			ctxt.Tlsoffset = 0x8a0

		case '8':
			ctxt.Tlsoffset = 0x468

		case '5':
			ctxt.Tlsoffset = 0 // dummy value, not needed
		}
	}

	// On arm, record goarm.
	if ctxt.Arch.Thechar == '5' {
		p := Getgoarm()
		if p != "" {
			ctxt.Goarm = int32(Atoi(p))
		} else {
			ctxt.Goarm = 6
		}
	}

	return ctxt
}

func linknewsym(ctxt *Link, symb string, v int) *LSym {
	s := new(LSym)
	*s = LSym{}

	s.Dynid = -1
	s.Plt = -1
	s.Got = -1
	s.Name = symb
	s.Type = 0
	s.Version = int16(v)
	s.Value = 0
	s.Sig = 0
	s.Size = 0
	ctxt.Nsymbol++

	s.Allsym = ctxt.Allsym
	ctxt.Allsym = s

	return s
}

func _lookup(ctxt *Link, symb string, v int, creat int) *LSym {
	h := uint32(v)
	for i := 0; i < len(symb); i++ {
		c := int(symb[i])
		h = h + h + h + uint32(c)
	}
	h &= 0xffffff
	h %= LINKHASH
	for s := ctxt.Hash[h]; s != nil; s = s.Hash {
		if int(s.Version) == v && s.Name == symb {
			return s
		}
	}
	if creat == 0 {
		return nil
	}

	s := linknewsym(ctxt, symb, v)
	s.Extname = s.Name
	s.Hash = ctxt.Hash[h]
	ctxt.Hash[h] = s

	return s
}

func Linklookup(ctxt *Link, name string, v int) *LSym {
	return _lookup(ctxt, name, v, 1)
}

// read-only lookup
func linkrlookup(ctxt *Link, name string, v int) *LSym {
	return _lookup(ctxt, name, v, 0)
}

func Linksymfmt(s *LSym) string {
	if s == nil {
		return "<nil>"
	}
	return s.Name
}
