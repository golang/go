// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

// Constants to identify various tar formats.
const (
	// The format is unknown.
	formatUnknown = (1 << iota) / 2 // Sequence of 0, 1, 2, 4, 8, etc...

	// The format of the original Unix V7 tar tool prior to standardization.
	formatV7

	// The old and new GNU formats, which are incompatible with USTAR.
	// This does cover the old GNU sparse extension.
	// This does not cover the GNU sparse extensions using PAX headers,
	// versions 0.0, 0.1, and 1.0; these fall under the PAX format.
	formatGNU

	// Schily's tar format, which is incompatible with USTAR.
	// This does not cover STAR extensions to the PAX format; these fall under
	// the PAX format.
	formatSTAR

	// USTAR is the former standardization of tar defined in POSIX.1-1988.
	// This is incompatible with the GNU and STAR formats.
	formatUSTAR

	// PAX is the latest standardization of tar defined in POSIX.1-2001.
	// This is an extension of USTAR and is "backwards compatible" with it.
	//
	// Some newer formats add their own extensions to PAX, such as GNU sparse
	// files and SCHILY extended attributes. Since they are backwards compatible
	// with PAX, they will be labelled as "PAX".
	formatPAX
)

// Magics used to identify various formats.
const (
	magicGNU, versionGNU     = "ustar ", " \x00"
	magicUSTAR, versionUSTAR = "ustar\x00", "00"
	trailerSTAR              = "tar\x00"
)

// Size constants from various tar specifications.
const (
	blockSize  = 512 // Size of each block in a tar stream
	nameSize   = 100 // Max length of the name field in USTAR format
	prefixSize = 155 // Max length of the prefix field in USTAR format
)

var zeroBlock block

type block [blockSize]byte

// Convert block to any number of formats.
func (b *block) V7() *headerV7       { return (*headerV7)(b) }
func (b *block) GNU() *headerGNU     { return (*headerGNU)(b) }
func (b *block) STAR() *headerSTAR   { return (*headerSTAR)(b) }
func (b *block) USTAR() *headerUSTAR { return (*headerUSTAR)(b) }
func (b *block) Sparse() sparseArray { return (sparseArray)(b[:]) }

// GetFormat checks that the block is a valid tar header based on the checksum.
// It then attempts to guess the specific format based on magic values.
// If the checksum fails, then formatUnknown is returned.
func (b *block) GetFormat() (format int) {
	// Verify checksum.
	var p parser
	value := p.parseOctal(b.V7().Chksum())
	chksum1, chksum2 := b.ComputeChecksum()
	if p.err != nil || (value != chksum1 && value != chksum2) {
		return formatUnknown
	}

	// Guess the magic values.
	magic := string(b.USTAR().Magic())
	version := string(b.USTAR().Version())
	trailer := string(b.STAR().Trailer())
	switch {
	case magic == magicUSTAR && trailer == trailerSTAR:
		return formatSTAR
	case magic == magicUSTAR:
		return formatUSTAR
	case magic == magicGNU && version == versionGNU:
		return formatGNU
	default:
		return formatV7
	}
}

// SetFormat writes the magic values necessary for specified format
// and then updates the checksum accordingly.
func (b *block) SetFormat(format int) {
	// Set the magic values.
	switch format {
	case formatV7:
		// Do nothing.
	case formatGNU:
		copy(b.GNU().Magic(), magicGNU)
		copy(b.GNU().Version(), versionGNU)
	case formatSTAR:
		copy(b.STAR().Magic(), magicUSTAR)
		copy(b.STAR().Version(), versionUSTAR)
		copy(b.STAR().Trailer(), trailerSTAR)
	case formatUSTAR, formatPAX:
		copy(b.USTAR().Magic(), magicUSTAR)
		copy(b.USTAR().Version(), versionUSTAR)
	default:
		panic("invalid format")
	}

	// Update checksum.
	// This field is special in that it is terminated by a NULL then space.
	var f formatter
	field := b.V7().Chksum()
	chksum, _ := b.ComputeChecksum() // Possible values are 256..128776
	f.formatOctal(field[:7], chksum) // Never fails since 128776 < 262143
	field[7] = ' '
}

// ComputeChecksum computes the checksum for the header block.
// POSIX specifies a sum of the unsigned byte values, but the Sun tar used
// signed byte values.
// We compute and return both.
func (b *block) ComputeChecksum() (unsigned, signed int64) {
	for i, c := range b {
		if 148 <= i && i < 156 {
			c = ' ' // Treat the checksum field itself as all spaces.
		}
		unsigned += int64(uint8(c))
		signed += int64(int8(c))
	}
	return unsigned, signed
}

type headerV7 [blockSize]byte

func (h *headerV7) Name() []byte     { return h[000:][:100] }
func (h *headerV7) Mode() []byte     { return h[100:][:8] }
func (h *headerV7) UID() []byte      { return h[108:][:8] }
func (h *headerV7) GID() []byte      { return h[116:][:8] }
func (h *headerV7) Size() []byte     { return h[124:][:12] }
func (h *headerV7) ModTime() []byte  { return h[136:][:12] }
func (h *headerV7) Chksum() []byte   { return h[148:][:8] }
func (h *headerV7) TypeFlag() []byte { return h[156:][:1] }
func (h *headerV7) LinkName() []byte { return h[157:][:100] }

type headerGNU [blockSize]byte

func (h *headerGNU) V7() *headerV7       { return (*headerV7)(h) }
func (h *headerGNU) Magic() []byte       { return h[257:][:6] }
func (h *headerGNU) Version() []byte     { return h[263:][:2] }
func (h *headerGNU) UserName() []byte    { return h[265:][:32] }
func (h *headerGNU) GroupName() []byte   { return h[297:][:32] }
func (h *headerGNU) DevMajor() []byte    { return h[329:][:8] }
func (h *headerGNU) DevMinor() []byte    { return h[337:][:8] }
func (h *headerGNU) AccessTime() []byte  { return h[345:][:12] }
func (h *headerGNU) ChangeTime() []byte  { return h[357:][:12] }
func (h *headerGNU) Sparse() sparseArray { return (sparseArray)(h[386:][:24*4+1]) }
func (h *headerGNU) RealSize() []byte    { return h[483:][:12] }

type headerSTAR [blockSize]byte

func (h *headerSTAR) V7() *headerV7      { return (*headerV7)(h) }
func (h *headerSTAR) Magic() []byte      { return h[257:][:6] }
func (h *headerSTAR) Version() []byte    { return h[263:][:2] }
func (h *headerSTAR) UserName() []byte   { return h[265:][:32] }
func (h *headerSTAR) GroupName() []byte  { return h[297:][:32] }
func (h *headerSTAR) DevMajor() []byte   { return h[329:][:8] }
func (h *headerSTAR) DevMinor() []byte   { return h[337:][:8] }
func (h *headerSTAR) Prefix() []byte     { return h[345:][:131] }
func (h *headerSTAR) AccessTime() []byte { return h[476:][:12] }
func (h *headerSTAR) ChangeTime() []byte { return h[488:][:12] }
func (h *headerSTAR) Trailer() []byte    { return h[508:][:4] }

type headerUSTAR [blockSize]byte

func (h *headerUSTAR) V7() *headerV7     { return (*headerV7)(h) }
func (h *headerUSTAR) Magic() []byte     { return h[257:][:6] }
func (h *headerUSTAR) Version() []byte   { return h[263:][:2] }
func (h *headerUSTAR) UserName() []byte  { return h[265:][:32] }
func (h *headerUSTAR) GroupName() []byte { return h[297:][:32] }
func (h *headerUSTAR) DevMajor() []byte  { return h[329:][:8] }
func (h *headerUSTAR) DevMinor() []byte  { return h[337:][:8] }
func (h *headerUSTAR) Prefix() []byte    { return h[345:][:155] }

type sparseArray []byte

func (s sparseArray) Entry(i int) sparseNode { return (sparseNode)(s[i*24:]) }
func (s sparseArray) IsExtended() []byte     { return s[24*s.MaxEntries():][:1] }
func (s sparseArray) MaxEntries() int        { return len(s) / 24 }

type sparseNode []byte

func (s sparseNode) Offset() []byte   { return s[00:][:12] }
func (s sparseNode) NumBytes() []byte { return s[12:][:12] }
