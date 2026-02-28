// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || netbsd

package routebsd

import (
	"runtime"
	"syscall"
)

func (w *wireFormat) parseInterfaceMessage(b []byte) (Message, error) {
	if len(b) < w.bodyOff {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	attrs := uint(nativeEndian.Uint32(b[4:8]))

	m := &InterfaceMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Addrs:   make([]Addr, syscall.RTAX_MAX),
		Flags:   int(nativeEndian.Uint32(b[8:12])),
		Index:   int(nativeEndian.Uint16(b[12:14])),
		extOff:  w.extOff,
		raw:     b[:l],
	}

	// We used to require that RTA_IFP always be set.
	// It turns out that on darwin messages about the
	// utun interface may not include a name. Issue #71064.
	if attrs&syscall.RTA_IFP != 0 {
		a, err := parseLinkAddr(b[w.bodyOff:])
		if err != nil {
			return nil, err
		}
		m.Addrs[syscall.RTAX_IFP] = a
		m.Name = a.(*LinkAddr).Name
	} else {
		// DragonFly seems to have unnamed interfaces
		// that we can't look up again. Just skip them.
		if runtime.GOOS == "dragonfly" {
			return nil, nil
		}
	}

	return m, nil
}

func (w *wireFormat) parseInterfaceAddrMessage(b []byte) (Message, error) {
	if len(b) < w.bodyOff {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &InterfaceAddrMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[8:12])),
		raw:     b[:l],
	}
	if runtime.GOOS == "netbsd" {
		m.Index = int(nativeEndian.Uint16(b[16:18]))
	} else {
		m.Index = int(nativeEndian.Uint16(b[12:14]))
	}
	var err error
	m.Addrs, err = parseAddrs(uint(nativeEndian.Uint32(b[4:8])), b[w.bodyOff:])
	if err != nil {
		return nil, err
	}
	return m, nil
}
