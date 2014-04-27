// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

// Socket control messages

package syscall

import "unsafe"

// Round the length of a raw sockaddr up to align it properly.
func cmsgAlignOf(salen int) int {
	salign := sizeofPtr
	// NOTE: It seems like 64-bit Darwin and DragonFly BSD kernels
	// still require 32-bit aligned access to network subsystem.
	if darwin64Bit || dragonfly64Bit {
		salign = 4
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

func cmsgData(h *Cmsghdr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(unsafe.Pointer(h)) + uintptr(cmsgAlignOf(SizeofCmsghdr)))
}

// SocketControlMessage represents a socket control message.
type SocketControlMessage struct {
	Header Cmsghdr
	Data   []byte
}

// ParseSocketControlMessage parses b as an array of socket control
// messages.
func ParseSocketControlMessage(b []byte) ([]SocketControlMessage, error) {
	var msgs []SocketControlMessage
	i := 0
	for i+CmsgLen(0) <= len(b) {
		h, dbuf, err := socketControlMessageHeaderAndData(b[i:])
		if err != nil {
			return nil, err
		}
		m := SocketControlMessage{Header: *h, Data: dbuf}
		msgs = append(msgs, m)
		i += cmsgAlignOf(int(h.Len))
	}
	return msgs, nil
}

func socketControlMessageHeaderAndData(b []byte) (*Cmsghdr, []byte, error) {
	h := (*Cmsghdr)(unsafe.Pointer(&b[0]))
	if h.Len < SizeofCmsghdr || int(h.Len) > len(b) {
		return nil, nil, EINVAL
	}
	return h, b[cmsgAlignOf(SizeofCmsghdr):h.Len], nil
}

// UnixRights encodes a set of open file descriptors into a socket
// control message for sending to another process.
func UnixRights(fds ...int) []byte {
	datalen := len(fds) * 4
	b := make([]byte, CmsgSpace(datalen))
	h := (*Cmsghdr)(unsafe.Pointer(&b[0]))
	h.Level = SOL_SOCKET
	h.Type = SCM_RIGHTS
	h.SetLen(CmsgLen(datalen))
	data := uintptr(cmsgData(h))
	for _, fd := range fds {
		*(*int32)(unsafe.Pointer(data)) = int32(fd)
		data += 4
	}
	return b
}

// ParseUnixRights decodes a socket control message that contains an
// integer array of open file descriptors from another process.
func ParseUnixRights(m *SocketControlMessage) ([]int, error) {
	if m.Header.Level != SOL_SOCKET {
		return nil, EINVAL
	}
	if m.Header.Type != SCM_RIGHTS {
		return nil, EINVAL
	}
	fds := make([]int, len(m.Data)>>2)
	for i, j := 0, 0; i < len(m.Data); i += 4 {
		fds[j] = int(*(*int32)(unsafe.Pointer(&m.Data[i])))
		j++
	}
	return fds, nil
}
