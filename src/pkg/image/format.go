// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"bufio"
	"io"
	"os"
)

// An UnknownFormatErr indicates that decoding encountered an unknown format.
var UnknownFormatErr = os.NewError("image: unknown format")

// A format holds an image format's name, magic header and how to decode it.
type format struct {
	name, magic string
	decode      func(io.Reader) (Image, os.Error)
}

// Formats is the list of registered formats.
var formats []format

// RegisterFormat registers an image format for use by Decode.
// Name is the name of the format, like "jpeg" or "png".
// Magic is the magic prefix that identifies the format's encoding.
// Decode is the function that decodes the encoded image.
func RegisterFormat(name, magic string, decode func(io.Reader) (Image, os.Error)) {
	n := len(formats)
	if n == cap(formats) {
		x := make([]format, n+1, 2*n+4)
		copy(x, formats)
		formats = x
	} else {
		formats = formats[0 : n+1]
	}
	formats[n] = format{name, magic, decode}
}

// A reader is an io.Reader that can also peek ahead.
type reader interface {
	io.Reader
	Peek(int) ([]byte, os.Error)
}

// AsReader converts an io.Reader to a reader.
func asReader(r io.Reader) reader {
	if rr, ok := r.(reader); ok {
		return rr
	}
	return bufio.NewReader(r)
}

// Decode decodes an image that has been encoded in a registered format.
// Format registration is typically done by the init method of the codec-
// specific package.
func Decode(r io.Reader) (m Image, formatName string, err os.Error) {
	var f format
	rr := asReader(r)
	for _, g := range formats {
		s, err := rr.Peek(len(g.magic))
		if err == nil && string(s) == g.magic {
			f = g
			break
		}
	}
	if f.decode == nil {
		return nil, "", UnknownFormatErr
	}
	m, err = f.decode(rr)
	return m, f.name, err
}
