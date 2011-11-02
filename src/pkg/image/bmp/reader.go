// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bmp implements a BMP image decoder.
//
// The BMP specification is at http://www.digicamsoft.com/bmp/bmp.html.
package bmp

import (
	"errors"
	"image"
	"image/color"
	"io"
)

// ErrUnsupported means that the input BMP image uses a valid but unsupported
// feature.
var ErrUnsupported = errors.New("bmp: unsupported BMP image")

func readUint16(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1])<<8
}

func readUint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// decodePaletted reads an 8 bit-per-pixel BMP image from r.
func decodePaletted(r io.Reader, c image.Config) (image.Image, error) {
	var tmp [4]byte
	paletted := image.NewPaletted(image.Rect(0, 0, c.Width, c.Height), c.ColorModel.(color.Palette))
	// BMP images are stored bottom-up rather than top-down.
	for y := c.Height - 1; y >= 0; y-- {
		p := paletted.Pix[y*paletted.Stride : y*paletted.Stride+c.Width]
		_, err := io.ReadFull(r, p)
		if err != nil {
			return nil, err
		}
		// Each row is 4-byte aligned.
		if c.Width%4 != 0 {
			_, err := io.ReadFull(r, tmp[:4-c.Width%4])
			if err != nil {
				return nil, err
			}
		}
	}
	return paletted, nil
}

// decodeRGBA reads a 24 bit-per-pixel BMP image from r.
func decodeRGBA(r io.Reader, c image.Config) (image.Image, error) {
	rgba := image.NewRGBA(image.Rect(0, 0, c.Width, c.Height))
	// There are 3 bytes per pixel, and each row is 4-byte aligned.
	b := make([]byte, (3*c.Width+3)&^3)
	// BMP images are stored bottom-up rather than top-down.
	for y := c.Height - 1; y >= 0; y-- {
		_, err := io.ReadFull(r, b)
		if err != nil {
			return nil, err
		}
		p := rgba.Pix[y*rgba.Stride : y*rgba.Stride+c.Width*4]
		for i, j := 0, 0; i < len(p); i, j = i+4, j+3 {
			// BMP images are stored in BGR order rather than RGB order.
			p[i+0] = b[j+2]
			p[i+1] = b[j+1]
			p[i+2] = b[j+0]
			p[i+3] = 0xFF
		}
	}
	return rgba, nil
}

// Decode reads a BMP image from r and returns it as an image.Image.
// Limitation: The file must be 8 or 24 bits per pixel.
func Decode(r io.Reader) (image.Image, error) {
	c, err := DecodeConfig(r)
	if err != nil {
		return nil, err
	}
	if c.ColorModel == color.RGBAModel {
		return decodeRGBA(r, c)
	}
	return decodePaletted(r, c)
}

// DecodeConfig returns the color model and dimensions of a BMP image without
// decoding the entire image.
// Limitation: The file must be 8 or 24 bits per pixel.
func DecodeConfig(r io.Reader) (config image.Config, err error) {
	// We only support those BMP images that are a BITMAPFILEHEADER
	// immediately followed by a BITMAPINFOHEADER.
	const (
		fileHeaderLen = 14
		infoHeaderLen = 40
	)
	var b [1024]byte
	if _, err = io.ReadFull(r, b[:fileHeaderLen+infoHeaderLen]); err != nil {
		return
	}
	if string(b[:2]) != "BM" {
		err = errors.New("bmp: invalid format")
		return
	}
	offset := readUint32(b[10:14])
	if readUint32(b[14:18]) != infoHeaderLen {
		err = ErrUnsupported
		return
	}
	width := int(readUint32(b[18:22]))
	height := int(readUint32(b[22:26]))
	if width < 0 || height < 0 {
		err = ErrUnsupported
		return
	}
	// We only support 1 plane, 8 or 24 bits per pixel and no compression.
	planes, bpp, compression := readUint16(b[26:28]), readUint16(b[28:30]), readUint32(b[30:34])
	if planes != 1 || compression != 0 {
		err = ErrUnsupported
		return
	}
	switch bpp {
	case 8:
		if offset != fileHeaderLen+infoHeaderLen+256*4 {
			err = ErrUnsupported
			return
		}
		_, err = io.ReadFull(r, b[:256*4])
		if err != nil {
			return
		}
		pcm := make(color.Palette, 256)
		for i := range pcm {
			// BMP images are stored in BGR order rather than RGB order.
			// Every 4th byte is padding.
			pcm[i] = color.RGBA{b[4*i+2], b[4*i+1], b[4*i+0], 0xFF}
		}
		return image.Config{pcm, width, height}, nil
	case 24:
		if offset != fileHeaderLen+infoHeaderLen {
			err = ErrUnsupported
			return
		}
		return image.Config{color.RGBAModel, width, height}, nil
	}
	err = ErrUnsupported
	return
}

func init() {
	image.RegisterFormat("bmp", "BM????\x00\x00\x00\x00", Decode, DecodeConfig)
}
