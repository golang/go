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
	name, magic  string
	decode       func(io.Reader) (Image, os.Error)
	decodeConfig func(io.Reader) (Config, os.Error)
}

// Formats is the list of registered formats.
var formats []format

// RegisterFormat registers an image format for use by Decode.
// Name is the name of the format, like "jpeg" or "png".
// Magic is the magic prefix that identifies the format's encoding.
// Decode is the function that decodes the encoded image.
// DecodeConfig is the function that decodes just its configuration.
func RegisterFormat(name, magic string, decode func(io.Reader) (Image, os.Error), decodeConfig func(io.Reader) (Config, os.Error)) {
	formats = append(formats, format{name, magic, decode, decodeConfig})
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

// sniff determines the format of r's data.
func sniff(r reader) format {
	for _, f := range formats {
		s, err := r.Peek(len(f.magic))
		if err == nil && string(s) == f.magic {
			return f
		}
	}
	return format{}
}

// Decode decodes an image that has been encoded in a registered format.
// The string returned is the format name used during format registration.
// Format registration is typically done by the init method of the codec-
// specific package.
func Decode(r io.Reader) (Image, string, os.Error) {
	rr := asReader(r)
	f := sniff(rr)
	if f.decode == nil {
		return nil, "", UnknownFormatErr
	}
	m, err := f.decode(rr)
	return m, f.name, err
}

// DecodeConfig decodes the color model and dimensions of an image that has
// been encoded in a registered format. The string returned is the format name
// used during format registration. Format registration is typically done by
// the init method of the codec-specific package.
func DecodeConfig(r io.Reader) (Config, string, os.Error) {
	rr := asReader(r)
	f := sniff(rr)
	if f.decodeConfig == nil {
		return Config{}, "", UnknownFormatErr
	}
	c, err := f.decodeConfig(rr)
	return c, f.name, err
}
