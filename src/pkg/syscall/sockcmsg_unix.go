// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

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

// CmsgLen returns the value to store in the Len field of the Cmsghdr
// structure, taking into account any necessary alignment.
func CmsgLen(datalen int) int {
	return cmsgAlignOf(SizeofCmsghdr) + datalen
}

// CmsgSpace returns the number of bytes an ancillary element with
// payload of the passed data length occupies.
func CmsgSpace(datalen int) int {
	return cmsgAlignOf(SizeofCmsghdr) + cmsgAlignOf(datalen)
}

func cmsgData(cmsg *Cmsghdr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(unsafe.Pointer(cmsg)) + SizeofCmsghdr)
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

	for len(buf) >= CmsgLen(0) {
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

// UnixRights encodes a set of open file descriptors into a socket
// control message for sending to another process.
func UnixRights(fds ...int) []byte {
	datalen := len(fds) * 4
	buf := make([]byte, CmsgSpace(datalen))
	cmsg := (*Cmsghdr)(unsafe.Pointer(&buf[0]))
	cmsg.Level = SOL_SOCKET
	cmsg.Type = SCM_RIGHTS
	cmsg.SetLen(CmsgLen(datalen))

	data := uintptr(cmsgData(cmsg))
	for _, fd := range fds {
		*(*int32)(unsafe.Pointer(data)) = int32(fd)
		data += 4
	}

	return buf
}

// ParseUnixRights decodes a socket control message that contains an
// integer array of open file descriptors from another process.
func ParseUnixRights(msg *SocketControlMessage) ([]int, int) {
	if msg.Header.Level != SOL_SOCKET {
		return nil, EINVAL
	}
	if msg.Header.Type != SCM_RIGHTS {
		return nil, EINVAL
	}
	fds := make([]int, len(msg.Data)>>2)
	for i, j := 0, 0; i < len(msg.Data); i += 4 {
		fds[j] = int(*(*int32)(unsafe.Pointer(&msg.Data[i])))
		j++
	}
	return fds, 0
}
