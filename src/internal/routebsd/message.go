// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package routebsd

// A Message represents a routing message.
type Message interface {
	message()
}

// parseRIB parses b as a routing information base and returns a list
// of routing messages.
func parseRIB(b []byte) ([]Message, error) {
	var msgs []Message
	nmsgs, nskips := 0, 0
	for len(b) > 4 {
		nmsgs++
		l := int(nativeEndian.Uint16(b[:2]))
		if l == 0 {
			return nil, errInvalidMessage
		}
		if len(b) < l {
			return nil, errMessageTooShort
		}
		if b[2] != rtmVersion {
			b = b[l:]
			continue
		}
		if w, ok := wireFormats[int(b[3])]; !ok {
			nskips++
		} else {
			m, err := w.parse(b[:l])
			if err != nil {
				return nil, err
			}
			if m == nil {
				nskips++
			} else {
				msgs = append(msgs, m)
			}
		}
		b = b[l:]
	}

	// We failed to parse any of the messages - version mismatch?
	if nmsgs != len(msgs)+nskips {
		return nil, errMessageMismatch
	}
	return msgs, nil
}
