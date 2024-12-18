// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd

package routebsd

import (
	"runtime"
	"syscall"
)

func (m *RouteMessage) marshal() ([]byte, error) {
	w, ok := wireFormats[m.Type]
	if !ok {
		return nil, errUnsupportedMessage
	}
	l := w.bodyOff + addrsSpace(m.Addrs)
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		// Fix stray pointer writes on macOS.
		// See golang.org/issue/22456.
		l += 1024
	}
	b := make([]byte, l)
	nativeEndian.PutUint16(b[:2], uint16(l))
	if m.Version == 0 {
		b[2] = rtmVersion
	} else {
		b[2] = byte(m.Version)
	}
	b[3] = byte(m.Type)
	nativeEndian.PutUint32(b[8:12], uint32(m.Flags))
	nativeEndian.PutUint16(b[4:6], uint16(m.Index))
	nativeEndian.PutUint32(b[16:20], uint32(m.ID))
	nativeEndian.PutUint32(b[20:24], uint32(m.Seq))
	attrs, err := marshalAddrs(b[w.bodyOff:], m.Addrs)
	if err != nil {
		return nil, err
	}
	if attrs > 0 {
		nativeEndian.PutUint32(b[12:16], uint32(attrs))
	}
	return b, nil
}

func (w *wireFormat) parseRouteMessage(typ RIBType, b []byte) (Message, error) {
	if len(b) < w.bodyOff {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &RouteMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[8:12])),
		Index:   int(nativeEndian.Uint16(b[4:6])),
		ID:      uintptr(nativeEndian.Uint32(b[16:20])),
		Seq:     int(nativeEndian.Uint32(b[20:24])),
		extOff:  w.extOff,
		raw:     b[:l],
	}
	errno := syscall.Errno(nativeEndian.Uint32(b[28:32]))
	if errno != 0 {
		m.Err = errno
	}
	var err error
	m.Addrs, err = parseAddrs(uint(nativeEndian.Uint32(b[12:16])), parseKernelInetAddr, b[w.bodyOff:])
	if err != nil {
		return nil, err
	}
	return m, nil
}
