// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gif implements a GIF image decoder.
//
// The GIF specification is at http://www.w3.org/Graphics/GIF/spec-gif89a.txt.
package gif

import (
	"bufio"
	"compress/lzw"
	"fmt"
	"image"
	"io"
	"os"
)

// If the io.Reader does not also have ReadByte, then decode will introduce its own buffering.
type reader interface {
	io.Reader
	io.ByteReader
}

// Masks etc.
const (
	// Fields.
	fColorMapFollows = 1 << 7

	// Image fields.
	ifInterlace = 1 << 6

	// Graphic control flags.
	gcTransparentColorSet = 1 << 0
)

// Section indicators.
const (
	sExtension       = 0x21
	sImageDescriptor = 0x2C
	sTrailer         = 0x3B
)

// Extensions.
const (
	eText           = 0x01 // Plain Text
	eGraphicControl = 0xF9 // Graphic Control
	eComment        = 0xFE // Comment
	eApplication    = 0xFF // Application
)

// decoder is the type used to decode a GIF file.
type decoder struct {
	r reader

	// From header.
	vers            string
	width           int
	height          int
	flags           byte
	headerFields    byte
	backgroundIndex byte
	loopCount       int
	delayTime       int

	// Unused from header.
	aspect byte

	// From image descriptor.
	imageFields byte

	// From graphics control.
	transparentIndex byte

	// Computed.
	pixelSize      uint
	globalColorMap image.PalettedColorModel

	// Used when decoding.
	delay []int
	image []*image.Paletted
	tmp   [1024]byte // must be at least 768 so we can read color map
}

// blockReader parses the block structure of GIF image data, which
// comprises (n, (n bytes)) blocks, with 1 <= n <= 255.  It is the
// reader given to the LZW decoder, which is thus immune to the
// blocking.  After the LZW decoder completes, there will be a 0-byte
// block remaining (0, ()), but under normal execution blockReader
// doesn't consume it, so it is handled in decode.
type blockReader struct {
	r     reader
	slice []byte
	tmp   [256]byte
}

func (b *blockReader) Read(p []byte) (n int, err os.Error) {
	if len(p) == 0 {
		return
	}
	if len(b.slice) > 0 {
		n = copy(p, b.slice)
		b.slice = b.slice[n:]
		return
	}
	var blockLen uint8
	blockLen, err = b.r.ReadByte()
	if err != nil {
		return
	}
	if blockLen == 0 {
		return 0, os.EOF
	}
	b.slice = b.tmp[0:blockLen]
	if _, err = io.ReadFull(b.r, b.slice); err != nil {
		return
	}
	return b.Read(p)
}

// decode reads a GIF image from r and stores the result in d.
func (d *decoder) decode(r io.Reader, configOnly bool) os.Error {
	// Add buffering if r does not provide ReadByte.
	if rr, ok := r.(reader); ok {
		d.r = rr
	} else {
		d.r = bufio.NewReader(r)
	}

	err := d.readHeaderAndScreenDescriptor()
	if err != nil {
		return err
	}
	if configOnly {
		return nil
	}

	if d.headerFields&fColorMapFollows != 0 {
		if d.globalColorMap, err = d.readColorMap(); err != nil {
			return err
		}
	}

	d.image = nil

Loop:
	for err == nil {
		var c byte
		c, err = d.r.ReadByte()
		if err == os.EOF {
			break
		}
		switch c {
		case sExtension:
			err = d.readExtension()

		case sImageDescriptor:
			var m *image.Paletted
			m, err = d.newImageFromDescriptor()
			if err != nil {
				break
			}
			if d.imageFields&fColorMapFollows != 0 {
				m.Palette, err = d.readColorMap()
				if err != nil {
					break
				}
				// TODO: do we set transparency in this map too? That would be
				// d.setTransparency(m.Palette)
			} else {
				m.Palette = d.globalColorMap
			}
			var litWidth uint8
			litWidth, err = d.r.ReadByte()
			if err != nil {
				return err
			}
			if litWidth > 8 {
				return fmt.Errorf("gif: pixel size in decode out of range: %d", litWidth)
			}
			// A wonderfully Go-like piece of magic. Unfortunately it's only at its
			// best for 8-bit pixels.
			lzwr := lzw.NewReader(&blockReader{r: d.r}, lzw.LSB, int(litWidth))
			if _, err = io.ReadFull(lzwr, m.Pix); err != nil {
				break
			}

			// There should be a "0" block remaining; drain that.
			c, err = d.r.ReadByte()
			if err != nil {
				return err
			}
			if c != 0 {
				return os.ErrorString("gif: extra data after image")
			}
			d.image = append(d.image, m)
			d.delay = append(d.delay, d.delayTime)
			d.delayTime = 0 // TODO: is this correct, or should we hold on to the value?

		case sTrailer:
			break Loop

		default:
			err = fmt.Errorf("gif: unknown block type: 0x%.2x", c)
		}
	}
	if err != nil {
		return err
	}
	if len(d.image) == 0 {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (d *decoder) readHeaderAndScreenDescriptor() os.Error {
	_, err := io.ReadFull(d.r, d.tmp[0:13])
	if err != nil {
		return err
	}
	d.vers = string(d.tmp[0:6])
	if d.vers != "GIF87a" && d.vers != "GIF89a" {
		return fmt.Errorf("gif: can't recognize format %s", d.vers)
	}
	d.width = int(d.tmp[6]) + int(d.tmp[7])<<8
	d.height = int(d.tmp[8]) + int(d.tmp[9])<<8
	d.headerFields = d.tmp[10]
	d.backgroundIndex = d.tmp[11]
	d.aspect = d.tmp[12]
	d.loopCount = -1
	d.pixelSize = uint(d.headerFields&7) + 1
	return nil
}

func (d *decoder) readColorMap() (image.PalettedColorModel, os.Error) {
	if d.pixelSize > 8 {
		return nil, fmt.Errorf("gif: can't handle %d bits per pixel", d.pixelSize)
	}
	numColors := 1 << d.pixelSize
	numValues := 3 * numColors
	_, err := io.ReadFull(d.r, d.tmp[0:numValues])
	if err != nil {
		return nil, fmt.Errorf("gif: short read on color map: %s", err)
	}
	colorMap := make(image.PalettedColorModel, numColors)
	j := 0
	for i := range colorMap {
		colorMap[i] = image.RGBAColor{d.tmp[j+0], d.tmp[j+1], d.tmp[j+2], 0xFF}
		j += 3
	}
	return colorMap, nil
}

func (d *decoder) readExtension() os.Error {
	extension, err := d.r.ReadByte()
	if err != nil {
		return err
	}
	size := 0
	switch extension {
	case eText:
		size = 13
	case eGraphicControl:
		return d.readGraphicControl()
	case eComment:
		// nothing to do but read the data.
	case eApplication:
		b, err := d.r.ReadByte()
		if err != nil {
			return err
		}
		// The spec requires size be 11, but Adobe sometimes uses 10.
		size = int(b)
	default:
		return fmt.Errorf("gif: unknown extension 0x%.2x", extension)
	}
	if size > 0 {
		if _, err := d.r.Read(d.tmp[0:size]); err != nil {
			return err
		}
	}

	// Application Extension with "NETSCAPE2.0" as string and 1 in data means
	// this extension defines a loop count.
	if extension == eApplication && string(d.tmp[:size]) == "NETSCAPE2.0" {
		n, err := d.readBlock()
		if n == 0 || err != nil {
			return err
		}
		if n == 3 && d.tmp[0] == 1 {
			d.loopCount = int(d.tmp[1]) | int(d.tmp[2])<<8
		}
	}
	for {
		n, err := d.readBlock()
		if n == 0 || err != nil {
			return err
		}
	}
	panic("unreachable")
}

func (d *decoder) readGraphicControl() os.Error {
	if _, err := io.ReadFull(d.r, d.tmp[0:6]); err != nil {
		return fmt.Errorf("gif: can't read graphic control: %s", err)
	}
	d.flags = d.tmp[1]
	d.delayTime = int(d.tmp[2]) | int(d.tmp[3])<<8
	if d.flags&gcTransparentColorSet != 0 {
		d.transparentIndex = d.tmp[4]
		d.setTransparency(d.globalColorMap)
	}
	return nil
}

func (d *decoder) setTransparency(colorMap image.PalettedColorModel) {
	if int(d.transparentIndex) < len(colorMap) {
		colorMap[d.transparentIndex] = image.RGBAColor{}
	}
}

func (d *decoder) newImageFromDescriptor() (*image.Paletted, os.Error) {
	if _, err := io.ReadFull(d.r, d.tmp[0:9]); err != nil {
		return nil, fmt.Errorf("gif: can't read image descriptor: %s", err)
	}
	_ = int(d.tmp[0]) + int(d.tmp[1])<<8 // TODO: honor left value
	_ = int(d.tmp[2]) + int(d.tmp[3])<<8 // TODO: honor top value
	width := int(d.tmp[4]) + int(d.tmp[5])<<8
	height := int(d.tmp[6]) + int(d.tmp[7])<<8
	d.imageFields = d.tmp[8]
	if d.imageFields&ifInterlace != 0 {
		return nil, os.ErrorString("gif: can't handle interlaced images")
	}
	return image.NewPaletted(width, height, nil), nil
}

func (d *decoder) readBlock() (int, os.Error) {
	n, err := d.r.ReadByte()
	if n == 0 || err != nil {
		return 0, err
	}
	return io.ReadFull(d.r, d.tmp[0:n])
}

// Decode reads a GIF image from r and returns the first embedded
// image as an image.Image.
// Limitation: The file must be 8 bits per pixel and have no interlacing.
func Decode(r io.Reader) (image.Image, os.Error) {
	var d decoder
	if err := d.decode(r, false); err != nil {
		return nil, err
	}
	return d.image[0], nil
}

// GIF represents the possibly multiple images stored in a GIF file.
type GIF struct {
	Image     []*image.Paletted // The successive images.
	Delay     []int             // The successive delay times, one per frame, in 100ths of a second.
	LoopCount int               // The loop count.
}

// DecodeAll reads a GIF image from r and returns the sequential frames
// and timing information.
// Limitation: The file must be 8 bits per pixel and have no interlacing.
func DecodeAll(r io.Reader) (*GIF, os.Error) {
	var d decoder
	if err := d.decode(r, false); err != nil {
		return nil, err
	}
	gif := &GIF{
		Image:     d.image,
		LoopCount: d.loopCount,
		Delay:     d.delay,
	}
	return gif, nil
}

// DecodeConfig returns the color model and dimensions of a GIF image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, os.Error) {
	var d decoder
	if err := d.decode(r, true); err != nil {
		return image.Config{}, err
	}
	colorMap := d.globalColorMap
	return image.Config{colorMap, d.width, d.height}, nil
}

func init() {
	image.RegisterFormat("gif", "GIF8?a", Decode, DecodeConfig)
}
