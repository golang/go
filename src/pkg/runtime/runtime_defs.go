// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go definitions of internal structures. Master is runtime.h

// TODO(lvd): automate conversion to all the _defs.go files

package runtime

import "unsafe"

const (
	gidle = iota
	grunnable
	grunning
	gsyscall
	gwaiting
	gmoribund
	gdead
	grecovery
)

// const ( Structrnd = sizeof(uintptr) )

type string_ struct {
	str *byte
	len int32
}

type iface struct {
	tab  *itab
	data unsafe.Pointer
}

type eface struct {
	type_ *Type
	data  unsafe.Pointer
}

type complex64 struct {
	real float32
	imag float32
}

type complex128 struct {
	real float64
	imag float64
}

type slice struct {
	array *byte
	len   uint32
	cap   uint32
}

type gobuf struct {
	sp unsafe.Pointer
	pc unsafe.Pointer
	g  *g_
}

type g_ struct {
	stackguard  unsafe.Pointer
	stackbase   unsafe.Pointer
	defer_      *defer_
	panic_      *panic_
	sched       gobuf
	stack0      unsafe.Pointer
	entry       unsafe.Pointer
	alllink     *g_
	param       unsafe.Pointer
	status      int16
	goid        int32
	selgen      uint32
	schedlink   *g_
	readyonstop bool
	ispanic     bool
	m           *m_
	lockedm     *m_
	sig         int32
	sigcode0    uintptr
	sigcode1    uintptr
}

type m_ struct {
	g0        *g_
	morepc    unsafe.Pointer
	morefp    unsafe.Pointer
	morebuf   gobuf
	moreframe uint32
	moreargs  uint32
	cret      uintptr
	procid    uint64
	gsignal   *g_
	tls       [8]uint32
	sched     gobuf
	curg      *g_
	id        int32
	mallocing int32
	gcing     int32
	locks     int32
	nomemprof int32
	waitnextg int32
	havenextg note
	nextg     *g_
	alllink   *m_
	schedlink *m_
	machport  uint32
	mcache    *mCache
	lockedg   *g_
	freg      [8]uint64
	// gostack	unsafe.Pointer  // __WINDOWS__
}

type stktop struct {
	stackguard *uint8
	stackbase  *uint8
	gobuf      gobuf
	args       uint32
	fp         *uint8
	free       bool
	panic_     bool
}

type alg struct {
	hash  func(uint32, unsafe.Pointer) uintptr
	equal func(uint32, unsafe.Pointer, unsafe.Pointer) uint32
	print func(uint32, unsafe.Pointer)
	copy  func(uint32, unsafe.Pointer, unsafe.Pointer)
}

type sigtab struct {
	flags int32
	name  *int8
}

const (
	sigCatch = (1 << iota)
	sigIgnore
	sigRestart
	sigQueue
	sigPanic
)

type Func struct {
	name   string
	typ    string
	src    string
	pcln   []byte
	entry  uintptr
	pc0    uintptr
	ln0    int32
	frame  int32
	args   int32
	locals int32
}

const (
	aMEM = iota
	aNOEQ
	aSTRING
	aINTER
	aNILINTER
	aMEMWORD
	amax
)

type defer_ struct {
	siz  int32
	sp   unsafe.Pointer
	pc   unsafe.Pointer
	fn   unsafe.Pointer
	link *defer_
	args [8]byte // padded to actual size
}

type panic_ struct {
	arg       eface
	stackbase unsafe.Pointer
	link      *panic_
	recovered bool
}

/*
 * External data.
 */

var (
	algarray    [amax]alg
	emptystring string
	allg        *g_
	allm        *m_
	goidgen     int32
	gomaxprocs  int32
	panicking   int32
	fd          int32
	gcwaiting   int32
	goos        *int8
)
