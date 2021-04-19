// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

func (*wireFormat) parseRouteMessage(_ RIBType, b []byte) (Message, error) {
	if len(b) < 40 {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &RouteMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Flags:   int(nativeEndian.Uint32(b[16:20])),
		Index:   int(nativeEndian.Uint16(b[6:8])),
		raw:     b[:l],
	}
	ll := int(nativeEndian.Uint16(b[4:6]))
	if len(b) < ll {
		return nil, errInvalidMessage
	}
	as, err := parseAddrs(uint(nativeEndian.Uint32(b[12:16])), parseKernelInetAddr, b[ll:])
	if err != nil {
		return nil, err
	}
	m.Addrs = as
	return m, nil
}
