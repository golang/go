// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x11

import (
	"bufio"
	"io"
	"os"
)

// Reads the DISPLAY environment variable, and returns the "12" in ":12.0".
func getDisplay() string {
	d := os.Getenv("DISPLAY")
	if len(d) < 1 || d[0] != ':' {
		return ""
	}
	i := 1
	for ; i < len(d); i++ {
		if d[i] < '0' || d[i] > '9' {
			break
		}
	}
	return d[1:i]
}

// Reads a big-endian uint16 from r, using b as a scratch buffer.
func readU16BE(r io.Reader, b []byte) (uint16, os.Error) {
	_, err := io.ReadFull(r, b[0:2])
	if err != nil {
		return 0, err
	}
	return uint16(b[0])<<8 + uint16(b[1]), nil
}

// Reads a length-prefixed string from r, using b as a scratch buffer.
func readStr(r io.Reader, b []byte) (s string, err os.Error) {
	n, err := readU16BE(r, b)
	if err != nil {
		return
	}
	if int(n) > len(b) {
		return s, os.NewError("Xauthority entry too long for buffer")
	}
	_, err = io.ReadFull(r, b[0:n])
	if err != nil {
		return
	}
	return string(b[0:n]), nil
}

// Reads the ~/.Xauthority file and returns the name/data pair for the DISPLAY.
// b is a scratch buffer to use, and should be at least 256 bytes long (i.e. it should be able to hold a hostname).
func readAuth(b []byte) (name, data string, err os.Error) {
	// As per /usr/include/X11/Xauth.h.
	const familyLocal = 256

	home := os.Getenv("HOME")
	if len(home) == 0 {
		err = os.NewError("unknown HOME")
		return
	}
	r, err := os.Open(home+"/.Xauthority", os.O_RDONLY, 0444)
	if err != nil {
		return
	}
	defer r.Close()
	br := bufio.NewReader(r)

	hostname, err := os.Hostname()
	if err != nil {
		return
	}
	display := getDisplay()
	for {
		family, err := readU16BE(br, b[0:2])
		if err != nil {
			return
		}
		addr, err := readStr(br, b[0:])
		if err != nil {
			return
		}
		disp, err := readStr(br, b[0:])
		if err != nil {
			return
		}
		name0, err := readStr(br, b[0:])
		if err != nil {
			return
		}
		data0, err := readStr(br, b[0:])
		if err != nil {
			return
		}
		if family == familyLocal && addr == hostname && disp == display {
			return name0, data0, nil
		}
	}
	panic("unreachable")
}
