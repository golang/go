// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bufio";
	"compress/zlib";
	"hash/crc32";
	"image";
	"io";
	"os";
	"strconv";
)

type encoder struct {
	w		io.Writer;
	m		image.Image;
	colorType	uint8;
	err		os.Error;
	header		[8]byte;
	footer		[4]byte;
	tmp		[3*256]byte;
}

// Big-endian.
func writeUint32(b []uint8, u uint32) {
	b[0] = uint8(u>>24);
	b[1] = uint8(u>>16);
	b[2] = uint8(u>>8);
	b[3] = uint8(u>>0);
}

// Returns whether or not the image is fully opaque.
func opaque(m image.Image) bool {
	for y := 0; y < m.Height(); y++ {
		for x := 0; x < m.Width(); x++ {
			_, _, _, a := m.At(x, y).RGBA();
			if a != 0xffffffff {
				return false;
			}
		}
	}
	return true;
}

func (e *encoder) writeChunk(b []byte, name string) {
	if e.err != nil {
		return;
	}
	n := uint32(len(b));
	if int(n) != len(b) {
		e.err = UnsupportedError(name + " chunk is too large: " + strconv.Itoa(len(b)));
		return;
	}
	writeUint32(e.header[0:4], n);
	e.header[4] = name[0];
	e.header[5] = name[1];
	e.header[6] = name[2];
	e.header[7] = name[3];
	crc := crc32.NewIEEE();
	crc.Write(e.header[4:8]);
	crc.Write(b);
	writeUint32(e.footer[0:4], crc.Sum32());

	_, e.err = e.w.Write(e.header[0:8]);
	if e.err != nil {
		return;
	}
	_, e.err = e.w.Write(b);
	if e.err != nil {
		return;
	}
	_, e.err = e.w.Write(e.footer[0:4]);
}

func (e *encoder) writeIHDR() {
	writeUint32(e.tmp[0:4], uint32(e.m.Width()));
	writeUint32(e.tmp[4:8], uint32(e.m.Height()));
	e.tmp[8] = 8;	// bit depth
	e.tmp[9] = e.colorType;
	e.tmp[10] = 0;	// default compression method
	e.tmp[11] = 0;	// default filter method
	e.tmp[12] = 0;	// non-interlaced
	e.writeChunk(e.tmp[0:13], "IHDR");
}

func (e *encoder) writePLTE(p image.PalettedColorModel) {
	if len(p) < 1 || len(p) > 256 {
		e.err = FormatError("bad palette length: " + strconv.Itoa(len(p)));
		return;
	}
	for i := 0; i < len(p); i++ {
		r, g, b, a := p[i].RGBA();
		if a != 0xffffffff {
			e.err = UnsupportedError("non-opaque palette color");
			return;
		}
		e.tmp[3*i + 0] = uint8(r>>24);
		e.tmp[3*i + 1] = uint8(g>>24);
		e.tmp[3*i + 2] = uint8(b>>24);
	}
	e.writeChunk(e.tmp[0 : 3*len(p)], "PLTE");
}

// An encoder is an io.Writer that satisfies writes by writing PNG IDAT chunks,
// including an 8-byte header and 4-byte CRC checksum per Write call. Such calls
// should be relatively infrequent, since writeIDATs uses a bufio.Writer.
//
// This method should only be called from writeIDATs (via writeImage).
// No other code should treat an encoder as an io.Writer.
//
// Note that, because the zlib deflater may involve an io.Pipe, e.Write calls may
// occur on a separate go-routine than the e.writeIDATs call, and care should be
// taken that e's state (such as its tmp buffer) is not modified concurrently.
func (e *encoder) Write(b []byte) (int, os.Error) {
	e.writeChunk(b, "IDAT");
	if e.err != nil {
		return 0, e.err;
	}
	return len(b), nil;
}

// Chooses the filter to use for encoding the current row, and applies it.
func filter(cr, pr []byte) {
	// TODO(nigeltao): For simplicity of implementation, this always picks the no-op filter.
	// To do this properly, we should use the same "minimize sum of absolute differences"
	// filter-choosing heuristic that libpng does.
	cr[0] = ftNone;
}

func writeImage(w io.Writer, m image.Image, ct uint8) os.Error {
	zw, err := zlib.NewDeflater(w);
	if err != nil {
		return err;
	}
	defer zw.Close();

	bpp := 0;	// Bytes per pixel.
	var paletted *image.Paletted;
	switch ct {
	case ctTrueColor:
		bpp = 3;
	case ctPaletted:
		bpp = 1;
		paletted = m.(*image.Paletted);
	case ctTrueColorAlpha:
		bpp = 4;
	}
	// cr and pr are the bytes for the current and previous row.
	// The +1 is for the per-row filter type, which is at cr[0].
	cr := make([]uint8, 1 + bpp * m.Width());
	pr := make([]uint8, 1 + bpp * m.Width());

	for y := 0; y < m.Height(); y++ {
		// Convert from colors to bytes.
		switch ct {
		case ctTrueColor:
			for x := 0; x < m.Width(); x++ {
				// We have previously verified that the alpha value is fully opaque.
				r, g, b, _ := m.At(x, y).RGBA();
				cr[3*x + 1] = uint8(r>>24);
				cr[3*x + 2] = uint8(g>>24);
				cr[3*x + 3] = uint8(b>>24);
			}
		case ctPaletted:
			for x := 0; x < m.Width(); x++ {
				cr[x+1] = paletted.ColorIndexAt(x, y);
			}
		case ctTrueColorAlpha:
			// Convert from image.Image (which is alpha-premultiplied) to PNG's non-alpha-premultiplied.
			for x := 0; x < m.Width(); x++ {
				c := image.NRGBAColorModel.Convert(m.At(x, y)).(image.NRGBAColor);
				cr[4*x + 1] = c.R;
				cr[4*x + 2] = c.G;
				cr[4*x + 3] = c.B;
				cr[4*x + 4] = c.A;
			}
		}

		// Apply the filter.
		filter(cr, pr);

		// Write the compressed bytes.
		_, err = zw.Write(cr);
		if err != nil {
			return err;
		}

		// The current row for y is the previous row for y+1.
		pr, cr = cr, pr;
	}
	return nil;
}

// Write the actual image data to one or more IDAT chunks.
func (e *encoder) writeIDATs() {
	if e.err != nil {
		return;
	}
	var bw *bufio.Writer;
	bw, e.err = bufio.NewWriterSize(e, 1<<15);
	if e.err != nil {
		return;
	}
	e.err = writeImage(bw, e.m, e.colorType);
	if e.err != nil {
		return;
	}
	e.err = bw.Flush();
}

func (e *encoder) writeIEND() {
	e.writeChunk(e.tmp[0:0], "IEND");
}

func Encode(w io.Writer, m image.Image) os.Error {
	// Obviously, negative widths and heights are invalid. Furthermore,
	// the PNG spec section 11.2.2 says that zero is an invalid dimension.
	mw, mh := int64(m.Width()), int64(m.Height());
	if mw <= 0 || mh <= 0 || mw >= 1<<32 || mh >= 1<<32 {
		return FormatError("invalid image size: " + strconv.Itoa64(mw) + "x" + strconv.Itoa64(mw));
	}

	var e encoder;
	e.w = w;
	e.m = m;
	e.colorType = uint8(ctTrueColorAlpha);
	pal, _ := m.(*image.Paletted);
	if pal != nil {
		e.colorType = ctPaletted;
	} else if opaque(m) {
		e.colorType = ctTrueColor;
	}

	_, e.err = io.WriteString(w, pngHeader);
	e.writeIHDR();
	if pal != nil {
		e.writePLTE(pal.Palette);
	}
	e.writeIDATs();
	e.writeIEND();
	return e.err;
}
