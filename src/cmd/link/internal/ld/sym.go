// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/span.c
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
)

func linknew(arch *sys.Arch) *Link {
	ctxt := &Link{
		Syms: &Symbols{
			hash: []map[string]*Symbol{
				// preallocate about 2mb for hash of
				// non static symbols
				make(map[string]*Symbol, 100000),
			},
			Allsym: make([]*Symbol, 0, 100000),
		},
		Arch: arch,
	}

	if obj.GOARCH != arch.Name {
		log.Fatalf("invalid obj.GOARCH %s (want %s)", obj.GOARCH, arch.Name)
	}

	return ctxt
}

// computeTLSOffset records the thread-local storage offset.
func (ctxt *Link) computeTLSOffset() {
	switch Headtype {
	default:
		log.Fatalf("unknown thread-local storage offset for %v", Headtype)

	case obj.Hplan9, obj.Hwindows, obj.Hwindowsgui:
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
		if obj.GOOS == "android" {
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

}
