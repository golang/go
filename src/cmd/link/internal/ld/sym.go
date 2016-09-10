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
	"fmt"
	"log"
)

func linknew(arch *sys.Arch) *Link {
	ctxt := &Link{
		Hash: []map[string]*Symbol{
			// preallocate about 2mb for hash of
			// non static symbols
			make(map[string]*Symbol, 100000),
		},
		Allsym: make([]*Symbol, 0, 100000),
		Arch:   arch,
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

func linknewsym(ctxt *Link, name string, v int) *Symbol {
	batch := ctxt.SymbolBatch
	if len(batch) == 0 {
		batch = make([]Symbol, 1000)
	}
	s := &batch[0]
	ctxt.SymbolBatch = batch[1:]

	s.Dynid = -1
	s.Plt = -1
	s.Got = -1
	s.Name = name
	s.Version = int16(v)
	ctxt.Allsym = append(ctxt.Allsym, s)

	return s
}

func Linklookup(ctxt *Link, name string, v int) *Symbol {
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
func Linkrlookup(ctxt *Link, name string, v int) *Symbol {
	return ctxt.Hash[v][name]
}

// A BuildMode indicates the sort of object we are building:
//   "exe": build a main package and everything it imports into an executable.
//   "c-shared": build a main package, plus all packages that it imports, into a
//     single C shared library. The only callable symbols will be those functions
//     marked as exported.
//   "shared": combine all packages passed on the command line, and their
//     dependencies, into a single shared library that will be used when
//     building with the -linkshared option.
type BuildMode uint8

const (
	BuildmodeUnset BuildMode = iota
	BuildmodeExe
	BuildmodePIE
	BuildmodeCArchive
	BuildmodeCShared
	BuildmodeShared
)

func (mode *BuildMode) Set(s string) error {
	badmode := func() error {
		return fmt.Errorf("buildmode %s not supported on %s/%s", s, obj.GOOS, obj.GOARCH)
	}
	switch s {
	default:
		return fmt.Errorf("invalid buildmode: %q", s)
	case "exe":
		*mode = BuildmodeExe
	case "pie":
		switch obj.GOOS {
		case "android", "linux":
		default:
			return badmode()
		}
		*mode = BuildmodePIE
	case "c-archive":
		switch obj.GOOS {
		case "darwin", "linux":
		case "windows":
			switch obj.GOARCH {
			case "amd64", "386":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildmodeCArchive
	case "c-shared":
		switch obj.GOARCH {
		case "386", "amd64", "arm", "arm64":
		default:
			return badmode()
		}
		*mode = BuildmodeCShared
	case "shared":
		switch obj.GOOS {
		case "linux":
			switch obj.GOARCH {
			case "386", "amd64", "arm", "arm64", "ppc64le", "s390x":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildmodeShared
	}
	return nil
}

func (mode *BuildMode) String() string {
	switch *mode {
	case BuildmodeUnset:
		return "" // avoid showing a default in usage message
	case BuildmodeExe:
		return "exe"
	case BuildmodePIE:
		return "pie"
	case BuildmodeCArchive:
		return "c-archive"
	case BuildmodeCShared:
		return "c-shared"
	case BuildmodeShared:
		return "shared"
	}
	return fmt.Sprintf("BuildMode(%d)", uint8(*mode))
}
