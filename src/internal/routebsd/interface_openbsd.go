// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import "syscall"

func (*wireFormat) parseInterfaceMessage(_ RIBType, b []byte) (Message, error) {
	if len(b) < 32 {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	attrs := uint(nativeEndian.Uint32(b[12:16]))
	if attrs&syscall.RTA_IFP == 0 {
		return nil, nil
	}
	m := &InterfaceMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[16:20])),
		Index:   int(nativeEndian.Uint16(b[6:8])),
		Addrs:   make([]Addr, syscall.RTAX_MAX),
		raw:     b[:l],
	}
	ll := int(nativeEndian.Uint16(b[4:6]))
	if len(b) < ll {
		return nil, errInvalidMessage
	}
	a, err := parseLinkAddr(b[ll:])
	if err != nil {
		return nil, err
	}
	m.Addrs[syscall.RTAX_IFP] = a
	m.Name = a.(*LinkAddr).Name
	return m, nil
}

func (*wireFormat) parseInterfaceAddrMessage(_ RIBType, b []byte) (Message, error) {
	if len(b) < 24 {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	bodyOff := int(nativeEndian.Uint16(b[4:6]))
	if len(b) < bodyOff {
		return nil, errInvalidMessage
	}
	m := &InterfaceAddrMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[12:16])),
		Index:   int(nativeEndian.Uint16(b[6:8])),
		raw:     b[:l],
	}
	var err error
	m.Addrs, err = parseAddrs(uint(nativeEndian.Uint32(b[12:16])), parseKernelInetAddr, b[bodyOff:])
	if err != nil {
		return nil, err
	}
	return m, nil
}

func (*wireFormat) parseInterfaceAnnounceMessage(_ RIBType, b []byte) (Message, error) {
	if len(b) < 26 {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &InterfaceAnnounceMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Index:   int(nativeEndian.Uint16(b[6:8])),
		What:    int(nativeEndian.Uint16(b[8:10])),
		raw:     b[:l],
	}
	for i := 0; i < 16; i++ {
		if b[10+i] != 0 {
			continue
		}
		m.Name = string(b[10 : 10+i])
		break
	}
	return m, nil
}
