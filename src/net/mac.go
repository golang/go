// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

const hexDigit = "0123456789abcdef"

// A HardwareAddr represents a physical hardware address.
type HardwareAddr []byte

func (a HardwareAddr) String() string {
	if len(a) == 0 {
		return ""
	}
	buf := make([]byte, 0, len(a)*3-1)
	for i, b := range a {
		if i > 0 {
			buf = append(buf, ':')
		}
		buf = append(buf, hexDigit[b>>4])
		buf = append(buf, hexDigit[b&0xF])
	}
	return string(buf)
}

// ParseMAC parses s as an IEEE 802 MAC-48, EUI-48, EUI-64, or a 20-octet
// IP over InfiniBand link-layer address using one of the following formats:
//
//	00:00:5e:00:53:01
//	02:00:5e:10:00:00:00:01
//	00:00:00:00:fe:80:00:00:00:00:00:00:02:00:5e:10:00:00:00:01
//	00-00-5e-00-53-01
//	02-00-5e-10-00-00-00-01
//	00-00-00-00-fe-80-00-00-00-00-00-00-02-00-5e-10-00-00-00-01
//	0000.5e00.5301
//	0200.5e10.0000.0001
//	0000.0000.fe80.0000.0000.0000.0200.5e10.0000.0001
//	00005e005301
func ParseMAC(s string) (hw HardwareAddr, err error) {
	if len(s) < 12 {
		goto error
	}

	if s[2] == ':' || s[2] == '-' {
		if (len(s)+1)%3 != 0 {
			goto error
		}
		n := (len(s) + 1) / 3
		if n != 6 && n != 8 && n != 20 {
			goto error
		}
		hw = make(HardwareAddr, n)
		for x, i := 0, 0; i < n; i++ {
			var ok bool
			if hw[i], ok = xtoi2(s[x:], s[2]); !ok {
				goto error
			}
			x += 3
		}
	} else if s[4] == '.' {
		if (len(s)+1)%5 != 0 {
			goto error
		}
		n := 2 * (len(s) + 1) / 5
		if n != 6 && n != 8 && n != 20 {
			goto error
		}
		hw = make(HardwareAddr, n)
		for x, i := 0, 0; i < n; i += 2 {
			var ok bool
			if hw[i], ok = xtoi2(s[x:x+2], 0); !ok {
				goto error
			}
			if hw[i+1], ok = xtoi2(s[x+2:], s[4]); !ok {
				goto error
			}
			x += 5
		}
	} else {
		if len(s)%2 != 0 {
			goto error
		}

		n := len(s) / 2
		if n != 6 && n != 8 && n != 20 {
			goto error
		}

		hw = make(HardwareAddr, len(s)/2)
		for x, i := 0, 0; i < n; i++ {
			var ok bool
			if hw[i], ok = xtoi2(s[x:x+2], 0); !ok {
				goto error
			}
			x += 2
		}
	}
	return hw, nil

error:
	return nil, &AddrError{Err: "invalid MAC address", Addr: s}
}

// MarshalText implements the [encoding.TextMarshaler] interface.
// The encoding is the same as the one returned by [HardwareAddr.String].
// This will be enabled in a future Go release; see issue #29678.
func (a HardwareAddr) MarshalText() ([]byte, error) {
	// For backward compatibility, marshal as plain []byte.
	if netmarshalOld() {
		return base64Encode(a), nil
	}

	return []byte(a.String()), nil
}

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
// In older Go versions the JSON encoding of HardwareAddr was
// that of a []byte. In order to support new Go programs reading JSON
// encodings produced by old Go programs, we support the []byte encoding.
func (a *HardwareAddr) UnmarshalText(text []byte) error {
	hw, err := ParseMAC(string(text))
	if err == nil {
		*a = hw
		return nil
	}

	// ParseMAC failed: try base64.

	dst := make([]byte, len(text)/4*3)
	n, ok := base64Decode(dst, text)
	if ok {
		*a = HardwareAddr(dst[:n])
		return nil
	}

	// The base64 decode failed: return the ParseMAC error.
	return err
}
