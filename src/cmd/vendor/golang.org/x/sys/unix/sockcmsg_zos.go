// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket control messages

package unix

import "unsafe"

// UnixCredentials encodes credentials into a socket control message
// for sending to another process. This can be used for
// authentication.
func UnixCredentials(ucred *Ucred) []byte {
	b := make([]byte, CmsgSpace(SizeofUcred))
	h := (*Cmsghdr)(unsafe.Pointer(&b[0]))
	h.Level = SOL_SOCKET
	h.Type = SCM_CREDENTIALS
	h.SetLen(CmsgLen(SizeofUcred))
	*(*Ucred)(h.data(0)) = *ucred
	return b
}

// ParseUnixCredentials decodes a socket control message that contains
// credentials in a Ucred structure. To receive such a message, the
// SO_PASSCRED option must be enabled on the socket.
func ParseUnixCredentials(m *SocketControlMessage) (*Ucred, error) {
	if m.Header.Level != SOL_SOCKET {
		return nil, EINVAL
	}
	if m.Header.Type != SCM_CREDENTIALS {
		return nil, EINVAL
	}
	ucred := *(*Ucred)(unsafe.Pointer(&m.Data[0]))
	return &ucred, nil
}

// PktInfo4 encodes Inet4Pktinfo into a socket control message of type IP_PKTINFO.
func PktInfo4(info *Inet4Pktinfo) []byte {
	b := make([]byte, CmsgSpace(SizeofInet4Pktinfo))
	h := (*Cmsghdr)(unsafe.Pointer(&b[0]))
	h.Level = SOL_IP
	h.Type = IP_PKTINFO
	h.SetLen(CmsgLen(SizeofInet4Pktinfo))
	*(*Inet4Pktinfo)(h.data(0)) = *info
	return b
}

// PktInfo6 encodes Inet6Pktinfo into a socket control message of type IPV6_PKTINFO.
func PktInfo6(info *Inet6Pktinfo) []byte {
	b := make([]byte, CmsgSpace(SizeofInet6Pktinfo))
	h := (*Cmsghdr)(unsafe.Pointer(&b[0]))
	h.Level = SOL_IPV6
	h.Type = IPV6_PKTINFO
	h.SetLen(CmsgLen(SizeofInet6Pktinfo))
	*(*Inet6Pktinfo)(h.data(0)) = *info
	return b
}
