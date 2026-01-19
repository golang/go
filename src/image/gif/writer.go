// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gif

import (
	"bufio"
	"bytes"
	"compress/lzw"
	"errors"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"internal/byteorder"
	"io"
	"math/bits"
)

// Graphic control extension fields.
const (
	gcLabel     = 0xF9
	gcBlockSize = 0x04
)

func log2(x int) int {
	if x < 2 {
		return 0
	}
	return bits.Len(uint(x-1)) - 1
}

// writer is a buffered writer.
type writer interface {
	Flush() error
	io.Writer
	io.ByteWriter
}

// encoder encodes an image to the GIF format.
type encoder struct {
	// w is the writer to write to. err is the first error encountered during
	// writing. All attempted writes after the first error become no-ops.
	w   writer
	err error
	// g is a reference to the data that is being encoded.
	g GIF
	// globalCT is the size in bytes of the global color table.
	globalCT int
	// buf is a scratch buffer. It must be at least 256 for the blockWriter.
	buf              [256]byte
	globalColorTable [3 * 256]byte
	localColorTable  [3 * 256]byte
}

// blockWriter writes the block structure of GIF image data, which
// comprises (n, (n bytes)) blocks, with 1 <= n <= 255. It is the
// writer given to the LZW encoder, which is thus immune to the
// blocking.
type blockWriter struct {
	e *encoder
}

func (b blockWriter) setup() {
	b.e.buf[0] = 0
}

func (b blockWriter) Flush() error {
	return b.e.err
}

func (b blockWriter) WriteByte(c byte) error {
	if b.e.err != nil {
		return b.e.err
	}

	// Append c to buffered sub-block.
	b.e.buf[0]++
	b.e.buf[b.e.buf[0]] = c
	if b.e.buf[0] < 255 {
		return nil
	}

	// Flush block
	b.e.write(b.e.buf[:256])
	b.e.buf[0] = 0
	return b.e.err
}

// blockWriter must be an io.Writer for lzw.NewWriter, but this is never
// actually called.
func (b blockWriter) Write(data []byte) (int, error) {
	for i, c := range data {
		if err := b.WriteByte(c); err != nil {
			return i, err
		}
	}
	return len(data), nil
}

func (b blockWriter) close() {
	// Write the block terminator (0x00), either by itself, or along with a
	// pending sub-block.
	if b.e.buf[0] == 0 {
		b.e.writeByte(0)
	} else {
		n := uint(b.e.buf[0])
		b.e.buf[n+1] = 0
		b.e.write(b.e.buf[:n+2])
	}
	b.e.flush()
}

func (e *encoder) flush() {
	if e.err != nil {
		return
	}
	e.err = e.w.Flush()
}

func (e *encoder) write(p []byte) {
	if e.err != nil {
		return
	}
	_, e.err = e.w.Write(p)
}

func (e *encoder) writeByte(b byte) {
	if e.err != nil {
		return
	}
	e.err = e.w.WriteByte(b)
}

func (e *encoder) writeHeader() {
	if e.err != nil {
		return
	}
	_, e.err = io.WriteString(e.w, "GIF89a")
	if e.err != nil {
		return
	}

	// Logical screen width and height.
	byteorder.LEPutUint16(e.buf[0:2], uint16(e.g.Config.Width))
	byteorder.LEPutUint16(e.buf[2:4], uint16(e.g.Config.Height))
	e.write(e.buf[:4])

	if p, ok := e.g.Config.ColorModel.(color.Palette); ok && len(p) > 0 {
		paddedSize := log2(len(p)) // Size of Global Color Table: 2^(1+n).
		e.buf[0] = fColorTable | uint8(paddedSize)
		e.buf[1] = e.g.BackgroundIndex
		e.buf[2] = 0x00 // Pixel Aspect Ratio.
		e.write(e.buf[:3])
		var err error
		e.globalCT, err = encodeColorTable(e.globalColorTable[:], p, paddedSize)
		if err != nil && e.err == nil {
			e.err = err
			return
		}
		e.write(e.globalColorTable[:e.globalCT])
	} else {
		// All frames have a local color table, so a global color table
		// is not needed.
		e.buf[0] = 0x00
		e.buf[1] = 0x00 // Background Color Index.
		e.buf[2] = 0x00 // Pixel Aspect Ratio.
		e.write(e.buf[:3])
	}

	// Add animation info if necessary.
	if len(e.g.Image) > 1 && e.g.LoopCount >= 0 {
		e.buf[0] = 0x21 // Extension Introducer.
		e.buf[1] = 0xff // Application Label.
		e.buf[2] = 0x0b // Block Size.
		e.write(e.buf[:3])
		_, err := io.WriteString(e.w, "NETSCAPE2.0") // Application Identifier.
		if err != nil && e.err == nil {
			e.err = err
			return
		}
		e.buf[0] = 0x03 // Block Size.
		e.buf[1] = 0x01 // Sub-block Index.
		byteorder.LEPutUint16(e.buf[2:4], uint16(e.g.LoopCount))
		e.buf[4] = 0x00 // Block Terminator.
		e.write(e.buf[:5])
	}
}

func encodeColorTable(dst []byte, p color.Palette, size int) (int, error) {
	if uint(size) >= 8 {
		return 0, errors.New("gif: cannot encode color table with more than 256 entries")
	}
	for i, c := range p {
		if c == nil {
			return 0, errors.New("gif: cannot encode color table with nil entries")
		}
		var r, g, b uint8
		// It is most likely that the palette is full of color.RGBAs, so they
		// get a fast path.
		if rgba, ok := c.(color.RGBA); ok {
			r, g, b = rgba.R, rgba.G, rgba.B
		} else {
			rr, gg, bb, _ := c.RGBA()
			r, g, b = uint8(rr>>8), uint8(gg>>8), uint8(bb>>8)
		}
		dst[3*i+0] = r
		dst[3*i+1] = g
		dst[3*i+2] = b
	}
	n := 1 << (size + 1)
	if n > len(p) {
		// Pad with black.
		clear(dst[3*len(p) : 3*n])
	}
	return 3 * n, nil
}

func (e *encoder) colorTablesMatch(localLen, transparentIndex int) bool {
	localSize := 3 * localLen
	if transparentIndex >= 0 {
		trOff := 3 * transparentIndex
		return bytes.Equal(e.globalColorTable[:trOff], e.localColorTable[:trOff]) &&
			bytes.Equal(e.globalColorTable[trOff+3:localSize], e.localColorTable[trOff+3:localSize])
	}
	return bytes.Equal(e.globalColorTable[:localSize], e.localColorTable[:localSize])
}

func (e *encoder) writeImageBlock(pm *image.Paletted, delay int, disposal byte) {
	if e.err != nil {
		return
	}

	if len(pm.Palette) == 0 {
		e.err = errors.New("gif: cannot encode image block with empty palette")
		return
	}

	b := pm.Bounds()
	if b.Min.X < 0 || b.Max.X >= 1<<16 || b.Min.Y < 0 || b.Max.Y >= 1<<16 {
		e.err = errors.New("gif: image block is too large to encode")
		return
	}
	if !b.In(image.Rectangle{Max: image.Point{e.g.Config.Width, e.g.Config.Height}}) {
		e.err = errors.New("gif: image block is out of bounds")
		return
	}

	transparentIndex := -1
	for i, c := range pm.Palette {
		if c == nil {
			e.err = errors.New("gif: cannot encode color table with nil entries")
			return
		}
		if _, _, _, a := c.RGBA(); a == 0 {
			transparentIndex = i
			break
		}
	}

	if delay > 0 || disposal != 0 || transparentIndex != -1 {
		e.buf[0] = sExtension  // Extension Introducer.
		e.buf[1] = gcLabel     // Graphic Control Label.
		e.buf[2] = gcBlockSize // Block Size.
		if transparentIndex != -1 {
			e.buf[3] = 0x01 | disposal<<2
		} else {
			e.buf[3] = 0x00 | disposal<<2
		}
		byteorder.LEPutUint16(e.buf[4:6], uint16(delay)) // Delay Time (1/100ths of a second)

		// Transparent color index.
		if transparentIndex != -1 {
			e.buf[6] = uint8(transparentIndex)
		} else {
			e.buf[6] = 0x00
		}
		e.buf[7] = 0x00 // Block Terminator.
		e.write(e.buf[:8])
	}
	e.buf[0] = sImageDescriptor
	byteorder.LEPutUint16(e.buf[1:3], uint16(b.Min.X))
	byteorder.LEPutUint16(e.buf[3:5], uint16(b.Min.Y))
	byteorder.LEPutUint16(e.buf[5:7], uint16(b.Dx()))
	byteorder.LEPutUint16(e.buf[7:9], uint16(b.Dy()))
	e.write(e.buf[:9])

	// To determine whether or not this frame's palette is the same as the
	// global palette, we can check a couple things. First, do they actually
	// point to the same []color.Color? If so, they are equal so long as the
	// frame's palette is not longer than the global palette...
	paddedSize := log2(len(pm.Palette)) // Size of Local Color Table: 2^(1+n).
	if gp, ok := e.g.Config.ColorModel.(color.Palette); ok && len(pm.Palette) <= len(gp) && &gp[0] == &pm.Palette[0] {
		e.writeByte(0) // Use the global color table.
	} else {
		ct, err := encodeColorTable(e.localColorTable[:], pm.Palette, paddedSize)
		if err != nil {
			if e.err == nil {
				e.err = err
			}
			return
		}
		// This frame's palette is not the very same slice as the global
		// palette, but it might be a copy, possibly with one value turned into
		// transparency by DecodeAll.
		if ct <= e.globalCT && e.colorTablesMatch(len(pm.Palette), transparentIndex) {
			e.writeByte(0) // Use the global color table.
		} else {
			// Use a local color table.
			e.writeByte(fColorTable | uint8(paddedSize))
			e.write(e.localColorTable[:ct])
		}
	}

	litWidth := paddedSize + 1
	if litWidth < 2 {
		litWidth = 2
	}
	e.writeByte(uint8(litWidth)) // LZW Minimum Code Size.

	bw := blockWriter{e: e}
	bw.setup()
	lzww := lzw.NewWriter(bw, lzw.LSB, litWidth)
	if dx := b.Dx(); dx == pm.Stride {
		_, e.err = lzww.Write(pm.Pix[:dx*b.Dy()])
		if e.err != nil {
			lzww.Close()
			return
		}
	} else {
		for i, y := 0, b.Min.Y; y < b.Max.Y; i, y = i+pm.Stride, y+1 {
			_, e.err = lzww.Write(pm.Pix[i : i+dx])
			if e.err != nil {
				lzww.Close()
				return
			}
		}
	}
	lzww.Close() // flush to bw
	bw.close()   // flush to e.w
}

// Options are the encoding parameters.
type Options struct {
	// NumColors is the maximum number of colors used in the image.
	// It ranges from 1 to 256.
	NumColors int

	// Quantizer is used to produce a palette with size NumColors.
	// palette.Plan9 is used in place of a nil Quantizer.
	Quantizer draw.Quantizer

	// Drawer is used to convert the source image to the desired palette.
	// draw.FloydSteinberg is used in place of a nil Drawer.
	Drawer draw.Drawer
}

// EncodeAll writes the images in g to w in GIF format with the
// given loop count and delay between frames.
func EncodeAll(w io.Writer, g *GIF) error {
	if len(g.Image) == 0 {
		return errors.New("gif: must provide at least one image")
	}

	if len(g.Image) != len(g.Delay) {
		return errors.New("gif: mismatched image and delay lengths")
	}

	e := encoder{g: *g}
	// The GIF.Disposal, GIF.Config and GIF.BackgroundIndex fields were added
	// in Go 1.5. Valid Go 1.4 code, such as when the Disposal field is omitted
	// in a GIF struct literal, should still produce valid GIFs.
	if e.g.Disposal != nil && len(e.g.Image) != len(e.g.Disposal) {
		return errors.New("gif: mismatched image and disposal lengths")
	}
	if e.g.Config == (image.Config{}) {
		p := g.Image[0].Bounds().Max
		e.g.Config.Width = p.X
		e.g.Config.Height = p.Y
	} else if e.g.Config.ColorModel != nil {
		if _, ok := e.g.Config.ColorModel.(color.Palette); !ok {
			return errors.New("gif: GIF color model must be a color.Palette")
		}
	}

	if ww, ok := w.(writer); ok {
		e.w = ww
	} else {
		e.w = bufio.NewWriter(w)
	}

	e.writeHeader()
	for i, pm := range g.Image {
		disposal := uint8(0)
		if g.Disposal != nil {
			disposal = g.Disposal[i]
		}
		e.writeImageBlock(pm, g.Delay[i], disposal)
	}
	e.writeByte(sTrailer)
	e.flush()
	return e.err
}

// Encode writes the Image m to w in GIF format.
func Encode(w io.Writer, m image.Image, o *Options) error {
	// Check for bounds and size restrictions.
	b := m.Bounds()
	if b.Dx() >= 1<<16 || b.Dy() >= 1<<16 {
		return errors.New("gif: image is too large to encode")
	}

	opts := Options{}
	if o != nil {
		opts = *o
	}
	if opts.NumColors < 1 || 256 < opts.NumColors {
		opts.NumColors = 256
	}
	if opts.Drawer == nil {
		opts.Drawer = draw.FloydSteinberg
	}

	pm, _ := m.(*image.Paletted)
	if pm == nil {
		if cp, ok := m.ColorModel().(color.Palette); ok {
			pm = image.NewPaletted(b, cp)
			for y := b.Min.Y; y < b.Max.Y; y++ {
				for x := b.Min.X; x < b.Max.X; x++ {
					pm.Set(x, y, cp.Convert(m.At(x, y)))
				}
			}
		}
	}
	if pm == nil || len(pm.Palette) > opts.NumColors {
		// Set pm to be a palettedized copy of m, including its bounds, which
		// might not start at (0, 0).
		//
		// TODO: Pick a better sub-sample of the Plan 9 palette.
		pm = image.NewPaletted(b, palette.Plan9[:opts.NumColors])
		if opts.Quantizer != nil {
			pm.Palette = opts.Quantizer.Quantize(make(color.Palette, 0, opts.NumColors), m)
		}
		opts.Drawer.Draw(pm, b, m, b.Min)
	}

	// When calling Encode instead of EncodeAll, the single-frame image is
	// translated such that its top-left corner is (0, 0), so that the single
	// frame completely fills the overall GIF's bounds.
	if pm.Rect.Min != (image.Point{}) {
		dup := *pm
		dup.Rect = dup.Rect.Sub(dup.Rect.Min)
		pm = &dup
	}

	return EncodeAll(w, &GIF{
		Image: []*image.Paletted{pm},
		Delay: []int{0},
		Config: image.Config{
			ColorModel: pm.Palette,
			Width:      b.Dx(),
			Height:     b.Dy(),
		},
	})
}
