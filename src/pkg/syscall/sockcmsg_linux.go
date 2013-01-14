// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket control messages

package syscall

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
	*((*Ucred)(cmsgData(h))) = *ucred
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
