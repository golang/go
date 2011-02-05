// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"compress/flate"
	"compress/zlib"
	"crypto/openpgp/error"
	"io"
	"os"
	"strconv"
)

// Compressed represents a compressed OpenPGP packet. The decompressed contents
// will contain more OpenPGP packets. See RFC 4880, section 5.6.
type Compressed struct {
	Body io.Reader
}

func (c *Compressed) parse(r io.Reader) os.Error {
	var buf [1]byte
	_, err := readFull(r, buf[:])
	if err != nil {
		return err
	}

	switch buf[0] {
	case 1:
		c.Body = flate.NewReader(r)
	case 2:
		c.Body, err = zlib.NewReader(r)
	default:
		err = error.UnsupportedError("unknown compression algorithm: " + strconv.Itoa(int(buf[0])))
	}

	return err
}
