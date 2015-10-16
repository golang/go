// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime type representation.

package runtime

import "unsafe"

// Needs to be in sync with ../cmd/compile/internal/ld/decodesym.go:/^func.commonsize,
// ../cmd/compile/internal/gc/reflect.go:/^func.dcommontype and
// ../reflect/type.go:/^type.rtype.
type _type struct {
	size       uintptr
	ptrdata    uintptr // size of memory prefix holding all pointers
	hash       uint32
	_unused    uint8
	align      uint8
	fieldalign uint8
	kind       uint8
	alg        *typeAlg
	// gcdata stores the GC type data for the garbage collector.
	// If the KindGCProg bit is set in kind, gcdata is a GC program.
	// Otherwise it is a ptrmask bitmap. See mbitmap.go for details.
	gcdata  *byte
	_string *string
	x       *uncommontype
	ptrto   *_type
}

type method struct {
	name    *string
	pkgpath *string
	mtyp    *_type
	typ     *_type
	ifn     unsafe.Pointer
	tfn     unsafe.Pointer
}

type uncommontype struct {
	name    *string
	pkgpath *string
	mhdr    []method
}

type imethod struct {
	name    *string
	pkgpath *string
	_type   *_type
}

type interfacetype struct {
	typ  _type
	mhdr []imethod
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
	typ       _type
	dotdotdot bool
	in        []*_type
	out       []*_type
}

type ptrtype struct {
	typ  _type
	elem *_type
}

type structfield struct {
	name    *string
	pkgpath *string
	typ     *_type
	tag     *string
	offset  uintptr
}

type structtype struct {
	typ    _type
	fields []structfield
}
