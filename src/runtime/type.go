// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime type representation.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// tflag is documented in ../reflect/type.go.
type tflag uint8

const tflagUncommon tflag = 1

// Needs to be in sync with ../cmd/compile/internal/ld/decodesym.go:/^func.commonsize,
// ../cmd/compile/internal/gc/reflect.go:/^func.dcommontype and
// ../reflect/type.go:/^type.rtype.
type _type struct {
	size       uintptr
	ptrdata    uintptr // size of memory prefix holding all pointers
	hash       uint32
	tflag      tflag
	align      uint8
	fieldalign uint8
	kind       uint8
	alg        *typeAlg
	// gcdata stores the GC type data for the garbage collector.
	// If the KindGCProg bit is set in kind, gcdata is a GC program.
	// Otherwise it is a ptrmask bitmap. See mbitmap.go for details.
	gcdata  *byte
	_string string
}

func (t *_type) uncommon() *uncommontype {
	if t.tflag&tflagUncommon == 0 {
		return nil
	}
	switch t.kind & kindMask {
	case kindStruct:
		type u struct {
			structtype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindPtr:
		type u struct {
			ptrtype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindFunc:
		type u struct {
			functype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindSlice:
		type u struct {
			slicetype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindArray:
		type u struct {
			arraytype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindChan:
		type u struct {
			chantype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindMap:
		type u struct {
			maptype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	case kindInterface:
		type u struct {
			interfacetype
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	default:
		type u struct {
			_type
			u uncommontype
		}
		return &(*u)(unsafe.Pointer(t)).u
	}
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func (t *_type) name() string {
	if hasPrefix(t._string, "map[") {
		return ""
	}
	if hasPrefix(t._string, "struct {") {
		return ""
	}
	if hasPrefix(t._string, "chan ") {
		return ""
	}
	if hasPrefix(t._string, "chan<-") {
		return ""
	}
	if hasPrefix(t._string, "func(") {
		return ""
	}
	switch t._string[0] {
	case '[', '*', '<':
		return ""
	}
	i := len(t._string) - 1
	for i >= 0 {
		if t._string[i] == '.' {
			break
		}
		i--
	}
	return t._string[i+1:]
}

func (t *functype) in() []*_type {
	// See funcType in reflect/type.go for details on data layout.
	uadd := uintptr(unsafe.Sizeof(functype{}))
	if t.typ.tflag&tflagUncommon != 0 {
		uadd += unsafe.Sizeof(uncommontype{})
	}
	return (*[1 << 20]*_type)(add(unsafe.Pointer(t), uadd))[:t.inCount]
}

func (t *functype) out() []*_type {
	// See funcType in reflect/type.go for details on data layout.
	uadd := uintptr(unsafe.Sizeof(functype{}))
	if t.typ.tflag&tflagUncommon != 0 {
		uadd += unsafe.Sizeof(uncommontype{})
	}
	outCount := t.outCount & (1<<15 - 1)
	return (*[1 << 20]*_type)(add(unsafe.Pointer(t), uadd))[t.inCount : t.inCount+outCount]
}

func (t *functype) dotdotdot() bool {
	return t.outCount&(1<<15) != 0
}

type method struct {
	name name
	mtyp *_type
	ifn  unsafe.Pointer
	tfn  unsafe.Pointer
}

type uncommontype struct {
	pkgpath *string
	mhdr    []method
}

type imethod struct {
	name  name
	_type *_type
}

type interfacetype struct {
	typ     _type
	pkgpath *string
	mhdr    []imethod
}

type maptype struct {
	typ           _type
	key           *_type
	elem          *_type
	bucket        *_type // internal type representing a hash bucket
	hmap          *_type // internal type representing a hmap
	keysize       uint8  // size of key slot
	indirectkey   bool   // store ptr to key instead of key itself
	valuesize     uint8  // size of value slot
	indirectvalue bool   // store ptr to value instead of value itself
	bucketsize    uint16 // size of bucket
	reflexivekey  bool   // true if k==k for all keys
	needkeyupdate bool   // true if we need to update key on an overwrite
}

type arraytype struct {
	typ   _type
	elem  *_type
	slice *_type
	len   uintptr
}

type chantype struct {
	typ  _type
	elem *_type
	dir  uintptr
}

type slicetype struct {
	typ  _type
	elem *_type
}

type functype struct {
	typ      _type
	inCount  uint16
	outCount uint16
}

type ptrtype struct {
	typ  _type
	elem *_type
}

type structfield struct {
	name   name
	typ    *_type
	offset uintptr
}

type structtype struct {
	typ     _type
	pkgPath *string
	fields  []structfield
}

// name is an encoded type name with optional extra data.
// See reflect/type.go for details.
type name struct {
	bytes *byte
}

func (n *name) data(off int) *byte {
	return (*byte)(add(unsafe.Pointer(n.bytes), uintptr(off)))
}

func (n *name) isExported() bool {
	return (*n.bytes)&(1<<0) != 0
}

func (n *name) nameLen() int {
	return int(uint16(*n.data(1))<<8 | uint16(*n.data(2)))
}

func (n *name) tagLen() int {
	if *n.data(0)&(1<<1) == 0 {
		return 0
	}
	off := 3 + n.nameLen()
	return int(uint16(*n.data(off))<<8 | uint16(*n.data(off + 1)))
}

func (n *name) name() (s string) {
	nl := n.nameLen()
	if nl == 0 {
		return ""
	}
	hdr := (*stringStruct)(unsafe.Pointer(&s))
	hdr.str = unsafe.Pointer(n.data(3))
	hdr.len = nl
	return s
}

func (n *name) pkgPath() *string {
	if *n.data(0)&(1<<2) == 0 {
		return nil
	}
	off := 3 + n.nameLen()
	if tl := n.tagLen(); tl > 0 {
		off += 2 + tl
	}
	off = int(round(uintptr(off), sys.PtrSize))
	return *(**string)(unsafe.Pointer(n.data(off)))
}
