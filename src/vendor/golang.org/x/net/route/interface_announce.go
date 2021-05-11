// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd
// +build dragonfly freebsd netbsd

package route

func (w *wireFormat) parseInterfaceAnnounceMessage(_ RIBType, b []byte) (Message, error) {
	if len(b) < w.bodyOff {
		return nil, errMessageTooShort
	}
	l := int(nativeEndian.Uint16(b[:2]))
	if len(b) < l {
		return nil, errInvalidMessage
	}
	m := &InterfaceAnnounceMessage{
		Version: int(b[2]),
		Type:    int(b[3]),
		Index:   int(nativeEndian.Uint16(b[4:6])),
		What:    int(nativeEndian.Uint16(b[22:24])),
		raw:     b[:l],
	}
	for i := 0; i < 16; i++ {
		if b[6+i] != 0 {
			continue
		}
		m.Name = string(b[6 : 6+i])
		break
	}
	return m, nil
}
