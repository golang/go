// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The png package implements a PNG image decoder and encoder.
//
// The PNG specification is at http://www.libpng.org/pub/png/spec/1.2/PNG-Contents.html
package png

import (
	"compress/zlib";
	"hash";
	"hash/crc32";
	"image";
	"io";
	"os";
)

// Color type, as per the PNG spec.
const (
	ctGrayscale		= 0;
	ctTrueColor		= 2;
	ctPaletted		= 3;
	ctGrayscaleAlpha	= 4;
	ctTrueColorAlpha	= 6;
)

// Filter type, as per the PNG spec.
const (
	ftNone		= 0;
	ftSub		= 1;
	ftUp		= 2;
	ftAverage	= 3;
	ftPaeth		= 4;
	nFilter		= 5;
)

// Decoding stage.
// The PNG specification says that the IHDR, PLTE (if present), IDAT and IEND
// chunks must appear in that order. There may be multiple IDAT chunks, and
// IDAT chunks must be sequential (i.e. they may not have any other chunks
// between them).
const (
	dsStart	= iota;
	dsSeenIHDR;
	dsSeenPLTE;
	dsSeenIDAT;
	dsSeenIEND;
)

const pngHeader = "\x89PNG\r\n\x1a\n"

type decoder struct {
	width, height	int;
	image		image.Image;
	colorType	uint8;
	stage		int;
	idatWriter	io.WriteCloser;
	idatDone	chan os.Error;
	tmp		[3 * 256]byte;
}

// A FormatError reports that the input is not a valid PNG.
type FormatError string

func (e FormatError) String() string	{ return "invalid PNG format: " + string(e) }

var chunkOrderError = FormatError("chunk out of order")

// An IDATDecodingError wraps an inner error (such as a ZLIB decoding error) encountered while processing an IDAT chunk.
type IDATDecodingError struct {
	Err os.Error;
}

func (e IDATDecodingError) String() string	{ return "IDAT decoding error: " + e.Err.String() }

// An UnsupportedError reports that the input uses a valid but unimplemented PNG feature.
type UnsupportedError string

func (e UnsupportedError) String() string	{ return "unsupported PNG feature: " + string(e) }

// Big-endian.
func parseUint32(b []uint8) uint32 {
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x;
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b;
}

func (d *decoder) parseIHDR(r io.Reader, crc hash.Hash32, length uint32) os.Error {
	if length != 13 {
		return FormatError("bad IHDR length")
	}
	_, err := io.ReadFull(r, d.tmp[0:13]);
	if err != nil {
		return err
	}
	crc.Write(d.tmp[0:13]);
	if d.tmp[8] != 8 {
		return UnsupportedError("bit depth")
	}
	if d.tmp[10] != 0 || d.tmp[11] != 0 || d.tmp[12] != 0 {
		return UnsupportedError("compression, filter or interlace method")
	}
	w := int32(parseUint32(d.tmp[0:4]));
	h := int32(parseUint32(d.tmp[4:8]));
	if w < 0 || h < 0 {
		return FormatError("negative dimension")
	}
	nPixels := int64(w) * int64(h);
	if nPixels != int64(int(nPixels)) {
		return UnsupportedError("dimension overflow")
	}
	d.colorType = d.tmp[9];
	switch d.colorType {
	case ctTrueColor:
		d.image = image.NewRGBA(int(w), int(h))
	case ctPaletted:
		d.image = image.NewPaletted(int(w), int(h), nil)
	case ctTrueColorAlpha:
		d.image = image.NewNRGBA(int(w), int(h))
	default:
		return UnsupportedError("color type")
	}
	d.width, d.height = int(w), int(h);
	return nil;
}

func (d *decoder) parsePLTE(r io.Reader, crc hash.Hash32, length uint32) os.Error {
	np := int(length / 3);	// The number of palette entries.
	if length%3 != 0 || np <= 0 || np > 256 {
		return FormatError("bad PLTE length")
	}
	n, err := io.ReadFull(r, d.tmp[0:3*np]);
	if err != nil {
		return err
	}
	crc.Write(d.tmp[0:n]);
	switch d.colorType {
	case ctPaletted:
		palette := make([]image.Color, np);
		for i := 0; i < np; i++ {
			palette[i] = image.RGBAColor{d.tmp[3*i+0], d.tmp[3*i+1], d.tmp[3*i+2], 0xff}
		}
		d.image.(*image.Paletted).Palette = image.PalettedColorModel(palette);
	case ctTrueColor, ctTrueColorAlpha:
		// As per the PNG spec, a PLTE chunk is optional (and for practical purposes,
		// ignorable) for the ctTrueColor and ctTrueColorAlpha color types (section 4.1.2).
		return nil
	default:
		return FormatError("PLTE, color type mismatch")
	}
	return nil;
}

// The Paeth filter function, as per the PNG specification.
func paeth(a, b, c uint8) uint8 {
	p := int(a) + int(b) - int(c);
	pa := abs(p - int(a));
	pb := abs(p - int(b));
	pc := abs(p - int(c));
	if pa <= pb && pa <= pc {
		return a
	} else if pb <= pc {
		return b
	}
	return c;
}

func (d *decoder) idatReader(idat io.Reader) os.Error {
	r, err := zlib.NewInflater(idat);
	if err != nil {
		return err
	}
	defer r.Close();
	bpp := 0;	// Bytes per pixel.
	maxPalette := uint8(0);
	var (
		rgba		*image.RGBA;
		nrgba		*image.NRGBA;
		paletted	*image.Paletted;
	)
	switch d.colorType {
	case ctTrueColor:
		bpp = 3;
		rgba = d.image.(*image.RGBA);
	case ctPaletted:
		bpp = 1;
		paletted = d.image.(*image.Paletted);
		maxPalette = uint8(len(paletted.Palette) - 1);
	case ctTrueColorAlpha:
		bpp = 4;
		nrgba = d.image.(*image.NRGBA);
	}
	// cr and pr are the bytes for the current and previous row.
	// The +1 is for the per-row filter type, which is at cr[0].
	cr := make([]uint8, 1+bpp*d.width);
	pr := make([]uint8, 1+bpp*d.width);

	for y := 0; y < d.height; y++ {
		// Read the decompressed bytes.
		_, err := io.ReadFull(r, cr);
		if err != nil {
			return err
		}

		// Apply the filter.
		cdat := cr[1:];
		pdat := pr[1:];
		switch cr[0] {
		case ftNone:
			// No-op.
		case ftSub:
			for i := bpp; i < len(cdat); i++ {
				cdat[i] += cdat[i-bpp]
			}
		case ftUp:
			for i := 0; i < len(cdat); i++ {
				cdat[i] += pdat[i]
			}
		case ftAverage:
			for i := 0; i < bpp; i++ {
				cdat[i] += pdat[i] / 2
			}
			for i := bpp; i < len(cdat); i++ {
				cdat[i] += uint8((int(cdat[i-bpp]) + int(pdat[i])) / 2)
			}
		case ftPaeth:
			for i := 0; i < bpp; i++ {
				cdat[i] += paeth(0, pdat[i], 0)
			}
			for i := bpp; i < len(cdat); i++ {
				cdat[i] += paeth(cdat[i-bpp], pdat[i], pdat[i-bpp])
			}
		default:
			return FormatError("bad filter type")
		}

		// Convert from bytes to colors.
		switch d.colorType {
		case ctTrueColor:
			for x := 0; x < d.width; x++ {
				rgba.Set(x, y, image.RGBAColor{cdat[3*x+0], cdat[3*x+1], cdat[3*x+2], 0xff})
			}
		case ctPaletted:
			for x := 0; x < d.width; x++ {
				if cdat[x] > maxPalette {
					return FormatError("palette index out of range")
				}
				paletted.SetColorIndex(x, y, cdat[x]);
			}
		case ctTrueColorAlpha:
			for x := 0; x < d.width; x++ {
				nrgba.Set(x, y, image.NRGBAColor{cdat[4*x+0], cdat[4*x+1], cdat[4*x+2], cdat[4*x+3]})
			}
		}

		// The current row for y is the previous row for y+1.
		pr, cr = cr, pr;
	}
	return nil;
}

func (d *decoder) parseIDAT(r io.Reader, crc hash.Hash32, length uint32) os.Error {
	// There may be more than one IDAT chunk, but their contents must be
	// treated as if it was one continuous stream (to the zlib decoder).
	// We bring up an io.Pipe and write the IDAT chunks into the pipe as
	// we see them, and decode the stream in a separate go-routine, which
	// signals its completion (successful or not) via a channel.
	if d.idatWriter == nil {
		pr, pw := io.Pipe();
		d.idatWriter = pw;
		d.idatDone = make(chan os.Error);
		go func() {
			err := d.idatReader(pr);
			if err == os.EOF {
				err = FormatError("too little IDAT")
			}
			pr.CloseWithError(FormatError("too much IDAT"));
			d.idatDone <- err;
		}();
	}
	var buf [4096]byte;
	for length > 0 {
		n, err1 := r.Read(buf[0:min(len(buf), int(length))]);
		// We delay checking err1. It is possible to get n bytes and an error,
		// but if the n bytes themselves contain a FormatError, for example, we
		// want to report that error, and not the one that made the Read stop.
		n, err2 := d.idatWriter.Write(buf[0:n]);
		if err2 != nil {
			return err2
		}
		if err1 != nil {
			return err1
		}
		crc.Write(buf[0:n]);
		length -= uint32(n);
	}
	return nil;
}

func (d *decoder) parseIEND(r io.Reader, crc hash.Hash32, length uint32) os.Error {
	if length != 0 {
		return FormatError("bad IEND length")
	}
	return nil;
}

func (d *decoder) parseChunk(r io.Reader) os.Error {
	// Read the length.
	n, err := io.ReadFull(r, d.tmp[0:4]);
	if err == os.EOF {
		return io.ErrUnexpectedEOF
	}
	if err != nil {
		return err
	}
	length := parseUint32(d.tmp[0:4]);

	// Read the chunk type.
	n, err = io.ReadFull(r, d.tmp[0:4]);
	if err == os.EOF {
		return io.ErrUnexpectedEOF
	}
	if err != nil {
		return err
	}
	crc := crc32.NewIEEE();
	crc.Write(d.tmp[0:4]);

	// Read the chunk data.
	switch string(d.tmp[0:4]) {
	case "IHDR":
		if d.stage != dsStart {
			return chunkOrderError
		}
		d.stage = dsSeenIHDR;
		err = d.parseIHDR(r, crc, length);
	case "PLTE":
		if d.stage != dsSeenIHDR {
			return chunkOrderError
		}
		d.stage = dsSeenPLTE;
		err = d.parsePLTE(r, crc, length);
	case "IDAT":
		if d.stage < dsSeenIHDR || d.stage > dsSeenIDAT || (d.colorType == ctPaletted && d.stage == dsSeenIHDR) {
			return chunkOrderError
		}
		d.stage = dsSeenIDAT;
		err = d.parseIDAT(r, crc, length);
	case "IEND":
		if d.stage != dsSeenIDAT {
			return chunkOrderError
		}
		d.stage = dsSeenIEND;
		err = d.parseIEND(r, crc, length);
	default:
		// Ignore this chunk (of a known length).
		var ignored [4096]byte;
		for length > 0 {
			n, err = io.ReadFull(r, ignored[0:min(len(ignored), int(length))]);
			if err != nil {
				return err
			}
			crc.Write(ignored[0:n]);
			length -= uint32(n);
		}
	}
	if err != nil {
		return err
	}

	// Read the checksum.
	n, err = io.ReadFull(r, d.tmp[0:4]);
	if err == os.EOF {
		return io.ErrUnexpectedEOF
	}
	if err != nil {
		return err
	}
	if parseUint32(d.tmp[0:4]) != crc.Sum32() {
		return FormatError("invalid checksum")
	}
	return nil;
}

func (d *decoder) checkHeader(r io.Reader) os.Error {
	_, err := io.ReadFull(r, d.tmp[0:8]);
	if err != nil {
		return err
	}
	if string(d.tmp[0:8]) != pngHeader {
		return FormatError("not a PNG file")
	}
	return nil;
}

// Decode reads a PNG formatted image from r and returns it as an image.Image.
// The type of Image returned depends on the PNG contents.
func Decode(r io.Reader) (image.Image, os.Error) {
	var d decoder;
	err := d.checkHeader(r);
	if err != nil {
		return nil, err
	}
	for d.stage = dsStart; d.stage != dsSeenIEND; {
		err = d.parseChunk(r);
		if err != nil {
			break
		}
	}
	if d.idatWriter != nil {
		d.idatWriter.Close();
		err1 := <-d.idatDone;
		if err == nil {
			err = err1
		}
	}
	if err != nil {
		return nil, err
	}
	return d.image, nil;
}
