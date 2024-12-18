// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import "syscall"

func (w *wireFormat) parseInterfaceMessage(b []byte) (Message, error) {
	extOff := w.extOff
	bodyOff := w.bodyOff
	if len(b) < extOff || len(b) < bodyOff {
		return nil, errInvalidMessage
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	attrs := uint(nativeEndian.Uint32(b[4:8]))
	if attrs&syscall.RTA_IFP == 0 {
		return nil, nil
	}
	m := &InterfaceMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[8:12])),
		Index:   int(nativeEndian.Uint16(b[12:14])),
		Addrs:   make([]Addr, syscall.RTAX_MAX),
		extOff:  extOff,
		raw:     b[:l],
	}
	a, err := parseLinkAddr(b[bodyOff:])
	if err != nil {
		return nil, err
	}
	m.Addrs[syscall.RTAX_IFP] = a
	m.Name = a.(*LinkAddr).Name
	return m, nil
}

func (w *wireFormat) parseInterfaceAddrMessage(b []byte) (Message, error) {
	bodyOff := w.bodyOff
	if len(b) < bodyOff {
		return nil, errInvalidMessage
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &InterfaceAddrMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[8:12])),
		Index:   int(nativeEndian.Uint16(b[12:14])),
		raw:     b[:l],
	}
	var err error
	m.Addrs, err = parseAddrs(uint(nativeEndian.Uint32(b[4:8])), b[bodyOff:])
	if err != nil {
		return nil, err
	}
	return m, nil
}
