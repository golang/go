// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xgb

import (
	"bufio"
	"io"
	"os"
)

func getU16BE(r io.Reader, b []byte) (uint16, os.Error) {
	_, err := io.ReadFull(r, b[0:2])
	if err != nil {
		return 0, err
	}
	return uint16(b[0])<<8 + uint16(b[1]), nil
}

func getBytes(r io.Reader, b []byte) ([]byte, os.Error) {
	n, err := getU16BE(r, b)
	if err != nil {
		return nil, err
	}
	if int(n) > len(b) {
		return nil, os.NewError("bytes too long for buffer")
	}
	_, err = io.ReadFull(r, b[0:n])
	if err != nil {
		return nil, err
	}
	return b[0:n], nil
}

func getString(r io.Reader, b []byte) (string, os.Error) {
	b, err := getBytes(r, b)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// readAuthority reads the X authority file for the DISPLAY.
// If hostname == "" or hostname == "localhost",
// readAuthority uses the system's hostname (as returned by os.Hostname) instead.
func readAuthority(hostname, display string) (name string, data []byte, err os.Error) {
	// b is a scratch buffer to use and should be at least 256 bytes long
	// (i.e. it should be able to hold a hostname).
	var b [256]byte

	// As per /usr/include/X11/Xauth.h.
	const familyLocal = 256

	if len(hostname) == 0 || hostname == "localhost" {
		hostname, err = os.Hostname()
		if err != nil {
			return "", nil, err
		}
	}

	fname := os.Getenv("XAUTHORITY")
	if len(fname) == 0 {
		home := os.Getenv("HOME")
		if len(home) == 0 {
			err = os.NewError("Xauthority not found: $XAUTHORITY, $HOME not set")
			return "", nil, err
		}
		fname = home + "/.Xauthority"
	}

	r, err := os.Open(fname, os.O_RDONLY, 0444)
	if err != nil {
		return "", nil, err
	}
	defer r.Close()

	br := bufio.NewReader(r)
	for {
		family, err := getU16BE(br, b[0:2])
		if err != nil {
			return "", nil, err
		}

		addr, err := getString(br, b[0:])
		if err != nil {
			return "", nil, err
		}

		disp, err := getString(br, b[0:])
		if err != nil {
			return "", nil, err
		}

		name0, err := getString(br, b[0:])
		if err != nil {
			return "", nil, err
		}

		data0, err := getBytes(br, b[0:])
		if err != nil {
			return "", nil, err
		}

		if family == familyLocal && addr == hostname && disp == display {
			return name0, data0, nil
		}
	}
	panic("unreachable")
}
