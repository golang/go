// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket control messages

package syscall

import (
	"unsafe"
)

// Round the length of a raw sockaddr up to align it propery.
func cmsgAlignOf(salen int) int {
	salign := sizeofPtr
	// NOTE: It seems like 64-bit Darwin kernel still requires 32-bit
	// aligned access to BSD subsystem.
	if darwinAMD64 {
		salign = 4
	}
	if salen == 0 {
		return salign
	}
	return (salen + salign - 1) & ^(salign - 1)
}

func cmsgLen(datalen int) int {
	return cmsgAlignOf(SizeofCmsghdr) + datalen
}

type SocketControlMessage struct {
	Header Cmsghdr
	Data   []byte
}

func ParseSocketControlMessage(buf []byte) ([]SocketControlMessage, int) {
	var (
		h     *Cmsghdr
		dbuf  []byte
		e     int
		cmsgs []SocketControlMessage
	)

	for len(buf) >= cmsgLen(0) {
		h, dbuf, e = socketControlMessageHeaderAndData(buf)
		if e != 0 {
			break
		}
		m := SocketControlMessage{}
		m.Header = *h
		m.Data = dbuf[:int(h.Len)-cmsgAlignOf(SizeofCmsghdr)]
		cmsgs = append(cmsgs, m)
		buf = buf[cmsgAlignOf(int(h.Len)):]
	}

	return cmsgs, e
}

func socketControlMessageHeaderAndData(buf []byte) (*Cmsghdr, []byte, int) {
	h := (*Cmsghdr)(unsafe.Pointer(&buf[0]))
	if h.Len < SizeofCmsghdr || int(h.Len) > len(buf) {
		return nil, nil, EINVAL
	}
	return h, buf[cmsgAlignOf(SizeofCmsghdr):], 0
}
