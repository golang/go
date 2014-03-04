// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd,amd64

package syscall

import "unsafe"

func (any *anyMessage) parseInterfaceMessage(b []byte) *InterfaceMessage {
	p := (*InterfaceMessage)(unsafe.Pointer(any))
	return &InterfaceMessage{Header: p.Header, Data: b[SizeofIfMsghdr:any.Msglen]}
}
