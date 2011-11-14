// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket control messages

package syscall

import (
	"unsafe"
)

// UnixCredentials encodes credentials into a socket control message
// for sending to another process. This can be used for
// authentication.
func UnixCredentials(ucred *Ucred) []byte {
	buf := make([]byte, CmsgSpace(SizeofUcred))
	cmsg := (*Cmsghdr)(unsafe.Pointer(&buf[0]))
	cmsg.Level = SOL_SOCKET
	cmsg.Type = SCM_CREDENTIALS
	cmsg.SetLen(CmsgLen(SizeofUcred))
	*((*Ucred)(cmsgData(cmsg))) = *ucred
	return buf
}

// ParseUnixCredentials decodes a socket control message that contains
// credentials in a Ucred structure. To receive such a message, the
// SO_PASSCRED option must be enabled on the socket.
func ParseUnixCredentials(msg *SocketControlMessage) (*Ucred, error) {
	if msg.Header.Level != SOL_SOCKET {
		return nil, EINVAL
	}
	if msg.Header.Type != SCM_CREDENTIALS {
		return nil, EINVAL
	}
	ucred := *(*Ucred)(unsafe.Pointer(&msg.Data[0]))
	return &ucred, nil
}
