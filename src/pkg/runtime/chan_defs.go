// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go definitions of internal structures. Master is chan.c

package runtime

type sudoG struct {
	g      *g_
	selgen uint32
	offset int16
	isfree int8
	link   *sudoG
	elem   [8]byte
}

type waitQ struct {
	first *sudoG
	last  *sudoG
}

type hChan struct {
	qcount    uint32
	dataqsiz  uint32
	elemsize  uint16
	closed    uint16
	elemalign uint8
	elemalg   *alg
	senddataq *link
	recvdataq *link
	recvq     waitQ
	sendq     waitQ
	free      sudoG
	lock
}

type link struct {
	link *link
	elem [8]byte
}

type scase struct {
	chan_ *hChan
	pc    *byte
	send  uint16
	so    uint16
	elemp *byte // union elem [8]byte
}

type select_ struct {
	tcase uint16
	ncase uint16
	link  *select_
	scase [1]*scase
}
