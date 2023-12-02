// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd

package route

func (w *wireFormat) parseInterfaceMulticastAddrMessage(_ RIBType, b []byte) (Message, error) {
	if len(b) < w.bodyOff {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &InterfaceMulticastAddrMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[8:12])),
		Index:   int(nativeEndian.Uint16(b[12:14])),
		raw:     b[:l],
	}
	var err error
	m.Addrs, err = parseAddrs(uint(nativeEndian.Uint32(b[4:8])), parseKernelInetAddr, b[w.bodyOff:])
	if err != nil {
		return nil, err
	}
	return m, nil
}
