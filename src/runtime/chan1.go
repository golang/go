// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//#define	MAXALIGN	8

type waitq struct {
	first *sudog
	last  *sudog
}

type hchan struct {
	qcount   uint // total data in the q
	dataqsiz uint // size of the circular q
	buf      *byte
	elemsize uint16
	closed   uint32
	elemtype *_type // element type
	sendx    uint   // send index
	recvx    uint   // receive index
	recvq    waitq  // list of recv waiters
	sendq    waitq  // list of send waiters
	lock     mutex
}

// Buffer follows Hchan immediately in memory.
// chanbuf(c, i) is pointer to the i'th slot in the buffer.
// #define chanbuf(c, i) ((byte*)((c)->buf)+(uintptr)(c)->elemsize*(i))

const (
	// scase.kind
	_CaseRecv = iota
	_CaseSend
	_CaseDefault
)

// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
type scase struct {
	elem        unsafe.Pointer // data element
	_chan       *hchan         // chan
	pc          uintptr        // return pc
	kind        uint16
	so          uint16 // vararg of selected bool
	receivedp   *bool  // pointer to received bool (recv2)
	releasetime int64
}

// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
type _select struct {
	tcase     uint16   // total count of scase[]
	ncase     uint16   // currently filled scase[]
	pollorder *uint16  // case poll order
	lockorder **hchan  // channel lock order
	scase     [1]scase // one per case (in order of appearance)
}
