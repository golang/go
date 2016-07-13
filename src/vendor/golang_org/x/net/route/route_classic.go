// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd

package route

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
		extOff:  w.extOff,
		raw:     b[:l],
	}
	var err error
	m.Addrs, err = parseAddrs(uint(nativeEndian.Uint32(b[12:16])), parseKernelInetAddr, b[w.bodyOff:])
	if err != nil {
		return nil, err
	}
	return m, nil
}
