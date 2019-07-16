// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Small in-memory unzip implementation.
// A simplified copy of the pre-Go 1 compress/flate/inflate.go
// and a modified copy of the zip reader in package time.
// (The one in package time does not support decompression; this one does.)

package syscall

const (
	maxCodeLen = 16    // max length of Huffman code
	maxHist    = 32768 // max history required
	maxLit     = 286
	maxDist    = 32
	numCodes   = 19 // number of codes in Huffman meta-code
)

type decompressor struct {
	in  string // compressed input
	out []byte // uncompressed output
	b   uint32 // input bits, at top of b
	nb  uint
	err bool // invalid input
	eof bool // reached EOF

	h1, h2   huffmanDecoder        // decoders for literal/length, distance
	bits     [maxLit + maxDist]int // lengths defining Huffman codes
	codebits [numCodes]int
}

func (f *decompressor) nextBlock() {
	for f.nb < 1+2 {
		if f.moreBits(); f.err {
			return
		}
	}
	f.eof = f.b&1 == 1
	f.b >>= 1
	typ := f.b & 3
	f.b >>= 2
	f.nb -= 1 + 2
	switch typ {
	case 0:
		f.dataBlock()
	case 1:
		// compressed, fixed Huffman tables
		f.huffmanBlock(&fixedHuffmanDecoder, nil)
	case 2:
		// compressed, dynamic Huffman tables
		if f.readHuffman(); f.err {
			break
		}
		f.huffmanBlock(&f.h1, &f.h2)
	default:
		// 3 is reserved.
		f.err = true
	}
}

// RFC 1951 section 3.2.7.
// Compression with dynamic Huffman codes

var codeOrder = [...]int{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}

func (f *decompressor) readHuffman() {
	// HLIT[5], HDIST[5], HCLEN[4].
	for f.nb < 5+5+4 {
		if f.moreBits(); f.err {
			return
		}
	}
	nlit := int(f.b&0x1F) + 257
	f.b >>= 5
	ndist := int(f.b&0x1F) + 1
	f.b >>= 5
	nclen := int(f.b&0xF) + 4
	f.b >>= 4
	f.nb -= 5 + 5 + 4

	// (HCLEN+4)*3 bits: code lengths in the magic codeOrder order.
	for i := 0; i < nclen; i++ {
		for f.nb < 3 {
			if f.moreBits(); f.err {
				return
			}
		}
		f.codebits[codeOrder[i]] = int(f.b & 0x7)
		f.b >>= 3
		f.nb -= 3
	}
	for i := nclen; i < len(codeOrder); i++ {
		f.codebits[codeOrder[i]] = 0
	}
	if !f.h1.init(f.codebits[0:]) {
		f.err = true
		return
	}

	// HLIT + 257 code lengths, HDIST + 1 code lengths,
	// using the code length Huffman code.
	for i, n := 0, nlit+ndist; i < n; {
		x := f.huffSym(&f.h1)
		if f.err {
			return
		}
		if x < 16 {
			// Actual length.
			f.bits[i] = x
			i++
			continue
		}
		// Repeat previous length or zero.
		var rep int
		var nb uint
		var b int
		switch x {
		default:
			f.err = true
			return
		case 16:
			rep = 3
			nb = 2
			if i == 0 {
				f.err = true
				return
			}
			b = f.bits[i-1]
		case 17:
			rep = 3
			nb = 3
			b = 0
		case 18:
			rep = 11
			nb = 7
			b = 0
		}
		for f.nb < nb {
			if f.moreBits(); f.err {
				return
			}
		}
		rep += int(f.b & uint32(1<<nb-1))
		f.b >>= nb
		f.nb -= nb
		if i+rep > n {
			f.err = true
			return
		}
		for j := 0; j < rep; j++ {
			f.bits[i] = b
			i++
		}
	}

	if !f.h1.init(f.bits[0:nlit]) || !f.h2.init(f.bits[nlit:nlit+ndist]) {
		f.err = true
		return
	}
}

// Decode a single Huffman block from f.
// hl and hd are the Huffman states for the lit/length values
// and the distance values, respectively. If hd == nil, using the
// fixed distance encoding associated with fixed Huffman blocks.
func (f *decompressor) huffmanBlock(hl, hd *huffmanDecoder) {
	for {
		v := f.huffSym(hl)
		if f.err {
			return
		}
		var n uint // number of bits extra
		var length int
		switch {
		case v < 256:
			f.out = append(f.out, byte(v))
			continue
		case v == 256:
			// Done with huffman block; read next block.
			return
		// otherwise, reference to older data
		case v < 265:
			length = v - (257 - 3)
			n = 0
		case v < 269:
			length = v*2 - (265*2 - 11)
			n = 1
		case v < 273:
			length = v*4 - (269*4 - 19)
			n = 2
		case v < 277:
			length = v*8 - (273*8 - 35)
			n = 3
		case v < 281:
			length = v*16 - (277*16 - 67)
			n = 4
		case v < 285:
			length = v*32 - (281*32 - 131)
			n = 5
		default:
			length = 258
			n = 0
		}
		if n > 0 {
			for f.nb < n {
				if f.moreBits(); f.err {
					return
				}
			}
			length += int(f.b & uint32(1<<n-1))
			f.b >>= n
			f.nb -= n
		}

		var dist int
		if hd == nil {
			for f.nb < 5 {
				if f.moreBits(); f.err {
					return
				}
			}
			dist = int(reverseByte[(f.b&0x1F)<<3])
			f.b >>= 5
			f.nb -= 5
		} else {
			if dist = f.huffSym(hd); f.err {
				return
			}
		}

		switch {
		case dist < 4:
			dist++
		case dist >= 30:
			f.err = true
			return
		default:
			nb := uint(dist-2) >> 1
			// have 1 bit in bottom of dist, need nb more.
			extra := (dist & 1) << nb
			for f.nb < nb {
				if f.moreBits(); f.err {
					return
				}
			}
			extra |= int(f.b & uint32(1<<nb-1))
			f.b >>= nb
			f.nb -= nb
			dist = 1<<(nb+1) + 1 + extra
		}

		// Copy [-dist:-dist+length] into output.
		// Encoding can be prescient, so no check on length.
		if dist > len(f.out) {
			f.err = true
			return
		}

		p := len(f.out) - dist
		for i := 0; i < length; i++ {
			f.out = append(f.out, f.out[p])
			p++
		}
	}
}

// Copy a single uncompressed data block from input to output.
func (f *decompressor) dataBlock() {
	// Uncompressed.
	// Discard current half-byte.
	f.nb = 0
	f.b = 0

	if len(f.in) < 4 {
		f.err = true
		return
	}

	buf := f.in[:4]
	f.in = f.in[4:]
	n := int(buf[0]) | int(buf[1])<<8
	nn := int(buf[2]) | int(buf[3])<<8
	if uint16(nn) != uint16(^n) {
		f.err = true
		return
	}

	if len(f.in) < n {
		f.err = true
		return
	}
	f.out = append(f.out, f.in[:n]...)
	f.in = f.in[n:]
}

func (f *decompressor) moreBits() {
	if len(f.in) == 0 {
		f.err = true
		return
	}
	c := f.in[0]
	f.in = f.in[1:]
	f.b |= uint32(c) << f.nb
	f.nb += 8
}

// Read the next Huffman-encoded symbol from f according to h.
func (f *decompressor) huffSym(h *huffmanDecoder) int {
	for n := uint(h.min); n <= uint(h.max); n++ {
		lim := h.limit[n]
		if lim == -1 {
			continue
		}
		for f.nb < n {
			if f.moreBits(); f.err {
				return 0
			}
		}
		v := int(f.b & uint32(1<<n-1))
		v <<= 16 - n
		v = int(reverseByte[v>>8]) | int(reverseByte[v&0xFF])<<8 // reverse bits
		if v <= lim {
			f.b >>= n
			f.nb -= n
			return h.codes[v-h.base[n]]
		}
	}
	f.err = true
	return 0
}

var reverseByte = [256]byte{
	0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
	0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
	0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
	0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
	0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
	0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
	0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
	0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
	0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
	0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
	0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
	0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
	0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
	0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
	0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
	0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
	0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
	0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
	0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
	0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
	0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
	0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
	0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
	0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
	0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
	0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
	0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
	0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
	0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
	0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
	0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
	0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
}

// Hard-coded Huffman tables for DEFLATE algorithm.
// See RFC 1951, section 3.2.6.
var fixedHuffmanDecoder = huffmanDecoder{
	7, 9,
	[maxCodeLen + 1]int{7: 23, 199, 511},
	[maxCodeLen + 1]int{7: 0, 24, 224},
	[]int{
		// length 7: 256-279
		256, 257, 258, 259, 260, 261, 262,
		263, 264, 265, 266, 267, 268, 269,
		270, 271, 272, 273, 274, 275, 276,
		277, 278, 279,

		// length 8: 0-143
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
		22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
		32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
		42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
		52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
		62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
		72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
		82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
		92, 93, 94, 95, 96, 97, 98, 99, 100,
		101, 102, 103, 104, 105, 106, 107, 108,
		109, 110, 111, 112, 113, 114, 115, 116,
		117, 118, 119, 120, 121, 122, 123, 124,
		125, 126, 127, 128, 129, 130, 131, 132,
		133, 134, 135, 136, 137, 138, 139, 140,
		141, 142, 143,

		// length 8: 280-287
		280, 281, 282, 283, 284, 285, 286, 287,

		// length 9: 144-255
		144, 145, 146, 147, 148, 149, 150, 151,
		152, 153, 154, 155, 156, 157, 158, 159,
		160, 161, 162, 163, 164, 165, 166, 167,
		168, 169, 170, 171, 172, 173, 174, 175,
		176, 177, 178, 179, 180, 181, 182, 183,
		184, 185, 186, 187, 188, 189, 190, 191,
		192, 193, 194, 195, 196, 197, 198, 199,
		200, 201, 202, 203, 204, 205, 206, 207,
		208, 209, 210, 211, 212, 213, 214, 215,
		216, 217, 218, 219, 220, 221, 222, 223,
		224, 225, 226, 227, 228, 229, 230, 231,
		232, 233, 234, 235, 236, 237, 238, 239,
		240, 241, 242, 243, 244, 245, 246, 247,
		248, 249, 250, 251, 252, 253, 254, 255,
	},
}

// Huffman decoder is based on
// J. Brian Connell, ``A Huffman-Shannon-Fano Code,''
// Proceedings of the IEEE, 61(7) (July 1973), pp 1046-1047.
type huffmanDecoder struct {
	// min, max code length
	min, max int

	// limit[i] = largest code word of length i
	// Given code v of length n,
	// need more bits if v > limit[n].
	limit [maxCodeLen + 1]int

	// base[i] = smallest code word of length i - seq number
	base [maxCodeLen + 1]int

	// codes[seq number] = output code.
	// Given code v of length n, value is
	// codes[v - base[n]].
	codes []int
}

// Initialize Huffman decoding tables from array of code lengths.
func (h *huffmanDecoder) init(bits []int) bool {
	// Count number of codes of each length,
	// compute min and max length.
	var count [maxCodeLen + 1]int
	var min, max int
	for _, n := range bits {
		if n == 0 {
			continue
		}
		if min == 0 || n < min {
			min = n
		}
		if n > max {
			max = n
		}
		count[n]++
	}
	if max == 0 {
		return false
	}

	h.min = min
	h.max = max

	// For each code range, compute
	// nextcode (first code of that length),
	// limit (last code of that length), and
	// base (offset from first code to sequence number).
	code := 0
	seq := 0
	var nextcode [maxCodeLen]int
	for i := min; i <= max; i++ {
		n := count[i]
		nextcode[i] = code
		h.base[i] = code - seq
		code += n
		seq += n
		h.limit[i] = code - 1
		code <<= 1
	}

	// Make array mapping sequence numbers to codes.
	if len(h.codes) < len(bits) {
		h.codes = make([]int, len(bits))
	}
	for i, n := range bits {
		if n == 0 {
			continue
		}
		code := nextcode[n]
		nextcode[n]++
		seq := code - h.base[n]
		h.codes[seq] = i
	}
	return true
}

func inflate(in string) (out []byte) {
	var d decompressor
	d.in = in
	for !d.err && !d.eof {
		d.nextBlock()
	}
	if len(d.in) != 0 {
		println("fs unzip: junk at end of compressed data")
		return nil
	}
	return d.out
}

// get4 returns the little-endian 32-bit value in b.
func zget4(b string) int {
	if len(b) < 4 {
		return 0
	}
	return int(b[0]) | int(b[1])<<8 | int(b[2])<<16 | int(b[3])<<24
}

// get2 returns the little-endian 16-bit value in b.
func zget2(b string) int {
	if len(b) < 2 {
		return 0
	}
	return int(b[0]) | int(b[1])<<8
}

func unzip(data string) {
	const (
		zecheader   = 0x06054b50
		zcheader    = 0x02014b50
		ztailsize   = 22
		zheadersize = 30
		zheader     = 0x04034b50
	)

	buf := data[len(data)-ztailsize:]
	n := zget2(buf[10:])
	size := zget4(buf[12:])
	off := zget4(buf[16:])

	hdr := data[off : off+size]
	for i := 0; i < n; i++ {
		// zip entry layout:
		//	0	magic[4]
		//	4	madevers[1]
		//	5	madeos[1]
		//	6	extvers[1]
		//	7	extos[1]
		//	8	flags[2]
		//	10	meth[2]
		//	12	modtime[2]
		//	14	moddate[2]
		//	16	crc[4]
		//	20	csize[4]
		//	24	uncsize[4]
		//	28	namelen[2]
		//	30	xlen[2]
		//	32	fclen[2]
		//	34	disknum[2]
		//	36	iattr[2]
		//	38	eattr[4]
		//	42	off[4]
		//	46	name[namelen]
		//	46+namelen+xlen+fclen - next header
		//
		if zget4(hdr) != zcheader {
			println("fs unzip: bad magic")
			break
		}
		meth := zget2(hdr[10:])
		mtime := zget2(hdr[12:])
		mdate := zget2(hdr[14:])
		csize := zget4(hdr[20:])
		size := zget4(hdr[24:])
		namelen := zget2(hdr[28:])
		xlen := zget2(hdr[30:])
		fclen := zget2(hdr[32:])
		xattr := uint32(zget4(hdr[38:])) >> 16
		off := zget4(hdr[42:])
		name := hdr[46 : 46+namelen]
		hdr = hdr[46+namelen+xlen+fclen:]

		// zip per-file header layout:
		//	0	magic[4]
		//	4	extvers[1]
		//	5	extos[1]
		//	6	flags[2]
		//	8	meth[2]
		//	10	modtime[2]
		//	12	moddate[2]
		//	14	crc[4]
		//	18	csize[4]
		//	22	uncsize[4]
		//	26	namelen[2]
		//	28	xlen[2]
		//	30	name[namelen]
		//	30+namelen+xlen - file data
		//
		buf := data[off : off+zheadersize+namelen]
		if zget4(buf) != zheader ||
			zget2(buf[8:]) != meth ||
			zget2(buf[26:]) != namelen ||
			buf[30:30+namelen] != name {
			println("fs unzip: inconsistent zip file")
			return
		}
		xlen = zget2(buf[28:])

		off += zheadersize + namelen + xlen

		var fdata []byte
		switch meth {
		case 0:
			// buf is uncompressed
			buf = data[off : off+size]
			fdata = []byte(buf)
		case 8:
			// buf is deflate-compressed
			buf = data[off : off+csize]
			fdata = inflate(buf)
			if len(fdata) != size {
				println("fs unzip: inconsistent size in zip file")
				return
			}
		}

		if xattr&S_IFMT == 0 {
			if xattr&0777 == 0 {
				xattr |= 0666
			}
			if len(name) > 0 && name[len(name)-1] == '/' {
				xattr |= S_IFDIR
				xattr |= 0111
			} else {
				xattr |= S_IFREG
			}
		}

		if err := create(name, xattr, zipToTime(mdate, mtime), fdata); err != nil {
			print("fs unzip: create ", name, ": ", err.Error(), "\n")
		}
	}

	chdirEnv()
}

func zipToTime(date, time int) int64 {
	dd := date & 0x1f
	mm := date >> 5 & 0xf
	yy := date >> 9 // since 1980

	sec := int64(315532800) // jan 1 1980
	sec += int64(yy) * 365 * 86400
	sec += int64(yy) / 4 * 86400
	if yy%4 > 0 || mm >= 3 {
		sec += 86400
	}
	sec += int64(daysBeforeMonth[mm]) * 86400
	sec += int64(dd-1) * 86400

	h := time >> 11
	m := time >> 5 & 0x3F
	s := time & 0x1f * 2
	sec += int64(h*3600 + m*60 + s)

	return sec
}

var daysBeforeMonth = [...]int32{
	0,
	0,
	31,
	31 + 28,
	31 + 28 + 31,
	31 + 28 + 31 + 30,
	31 + 28 + 31 + 30 + 31,
	31 + 28 + 31 + 30 + 31 + 30,
	31 + 28 + 31 + 30 + 31 + 30 + 31,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31,
}
