// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bufio"
	"io"
)

type byteReader interface {
	io.Reader
	io.ByteReader
}

// unpackBits decodes the PackBits-compressed data in src and returns the
// uncompressed data.
//
// The PackBits compression format is described in section 9 (p. 42)
// of the TIFF spec.
func unpackBits(r io.Reader) ([]byte, error) {
	buf := make([]byte, 128)
	dst := make([]byte, 0, 1024)
	br, ok := r.(byteReader)
	if !ok {
		br = bufio.NewReader(r)
	}

	for {
		b, err := br.ReadByte()
		if err != nil {
			if err == io.EOF {
				return dst, nil
			}
			return nil, err
		}
		code := int(int8(b))
		switch {
		case code >= 0:
			n, err := io.ReadFull(br, buf[:code+1])
			if err != nil {
				return nil, err
			}
			dst = append(dst, buf[:n]...)
		case code == -128:
			// No-op.
		default:
			if b, err = br.ReadByte(); err != nil {
				return nil, err
			}
			for j := 0; j < 1-code; j++ {
				buf[j] = b
			}
			dst = append(dst, buf[:1-code]...)
		}
	}
	panic("unreachable")
}
