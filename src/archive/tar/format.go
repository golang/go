// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import "strings"

// Format represents the tar archive format.
//
// The original tar format was introduced in Unix V7.
// Since then, there have been multiple competing formats attempting to
// standardize or extend the V7 format to overcome its limitations.
// The most common formats are the USTAR, PAX, and GNU formats,
// each with their own advantages and limitations.
//
// The following table captures the capabilities of each format:
//
//	                  |  USTAR |       PAX |       GNU
//	------------------+--------+-----------+----------
//	Name              |   256B | unlimited | unlimited
//	Linkname          |   100B | unlimited | unlimited
//	Size              | uint33 | unlimited |    uint89
//	Mode              | uint21 |    uint21 |    uint57
//	Uid/Gid           | uint21 | unlimited |    uint57
//	Uname/Gname       |    32B | unlimited |       32B
//	ModTime           | uint33 | unlimited |     int89
//	AccessTime        |    n/a | unlimited |     int89
//	ChangeTime        |    n/a | unlimited |     int89
//	Devmajor/Devminor | uint21 |    uint21 |    uint57
//	------------------+--------+-----------+----------
//	string encoding   |  ASCII |     UTF-8 |    binary
//	sub-second times  |     no |       yes |        no
//	sparse files      |     no |       yes |       yes
//
// The table's upper portion shows the Header fields, where each format reports
// the maximum number of bytes allowed for each string field and
// the integer type used to store each numeric field
// (where timestamps are stored as the number of seconds since the Unix epoch).
//
// The table's lower portion shows specialized features of each format,
// such as supported string encodings, support for sub-second timestamps,
// or support for sparse files.
//
// The Writer currently provides no support for sparse files.
type Format int

// Constants to identify various tar formats.
const (
	// Deliberately hide the meaning of constants from public API.
	_ Format = (1 << iota) / 4 // Sequence of 0, 0, 1, 2, 4, 8, etc...

	// FormatUnknown indicates that the format is unknown.
	FormatUnknown

	// The format of the original Unix V7 tar tool prior to standardization.
	formatV7

	// FormatUSTAR represents the USTAR header format defined in POSIX.1-1988.
	//
	// While this format is compatible with most tar readers,
	// the format has several limitations making it unsuitable for some usages.
	// Most notably, it cannot support sparse files, files larger than 8GiB,
	// filenames larger than 256 characters, and non-ASCII filenames.
	//
	// Reference:
	//	http://pubs.opengroup.org/onlinepubs/9699919799/utilities/pax.html#tag_20_92_13_06
	FormatUSTAR

	// FormatPAX represents the PAX header format defined in POSIX.1-2001.
	//
	// PAX extends USTAR by writing a special file with Typeflag TypeXHeader
	// preceding the original header. This file contains a set of key-value
	// records, which are used to overcome USTAR's shortcomings, in addition to
	// providing the ability to have sub-second resolution for timestamps.
	//
	// Some newer formats add their own extensions to PAX by defining their
	// own keys and assigning certain semantic meaning to the associated values.
	// For example, sparse file support in PAX is implemented using keys
	// defined by the GNU manual (e.g., "GNU.sparse.map").
	//
	// Reference:
	//	http://pubs.opengroup.org/onlinepubs/009695399/utilities/pax.html
	FormatPAX

	// FormatGNU represents the GNU header format.
	//
	// The GNU header format is older than the USTAR and PAX standards and
	// is not compatible with them. The GNU format supports
	// arbitrary file sizes, filenames of arbitrary encoding and length,
	// sparse files, and other features.
	//
	// It is recommended that PAX be chosen over GNU unless the target
	// application can only parse GNU formatted archives.
	//
	// Reference:
	//	http://www.gnu.org/software/tar/manual/html_node/Standard.html
	FormatGNU

	// Schily's tar format, which is incompatible with USTAR.
	// This does not cover STAR extensions to the PAX format; these fall under
	// the PAX format.
	formatSTAR

	formatMax
)

func (f Format) has(f2 Format) bool   { return f&f2 != 0 }
func (f *Format) mayBe(f2 Format)     { *f |= f2 }
func (f *Format) mayOnlyBe(f2 Format) { *f &= f2 }
func (f *Format) mustNotBe(f2 Format) { *f &^= f2 }

var formatNames = map[Format]string{
	formatV7: "V7", FormatUSTAR: "USTAR", FormatPAX: "PAX", FormatGNU: "GNU", formatSTAR: "STAR",
}

func (f Format) String() string {
	var ss []string
	for f2 := Format(1); f2 < formatMax; f2 <<= 1 {
		if f.has(f2) {
			ss = append(ss, formatNames[f2])
		}
	}
	switch len(ss) {
	case 0:
		return "<unknown>"
	case 1:
		return ss[0]
	default:
		return "(" + strings.Join(ss, " | ") + ")"
	}
}

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

// blockPadding computes the number of bytes needed to pad offset up to the
// nearest block edge where 0 <= n < blockSize.
func blockPadding(offset int64) (n int64) {
	return -offset & (blockSize - 1)
}

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
// If the checksum fails, then FormatUnknown is returned.
func (b *block) GetFormat() Format {
	// Verify checksum.
	var p parser
	value := p.parseOctal(b.V7().Chksum())
	chksum1, chksum2 := b.ComputeChecksum()
	if p.err != nil || (value != chksum1 && value != chksum2) {
		return FormatUnknown
	}

	// Guess the magic values.
	magic := string(b.USTAR().Magic())
	version := string(b.USTAR().Version())
	trailer := string(b.STAR().Trailer())
	switch {
	case magic == magicUSTAR && trailer == trailerSTAR:
		return formatSTAR
	case magic == magicUSTAR:
		return FormatUSTAR | FormatPAX
	case magic == magicGNU && version == versionGNU:
		return FormatGNU
	default:
		return formatV7
	}
}

// SetFormat writes the magic values necessary for specified format
// and then updates the checksum accordingly.
func (b *block) SetFormat(format Format) {
	// Set the magic values.
	switch {
	case format.has(formatV7):
		// Do nothing.
	case format.has(FormatGNU):
		copy(b.GNU().Magic(), magicGNU)
		copy(b.GNU().Version(), versionGNU)
	case format.has(formatSTAR):
		copy(b.STAR().Magic(), magicUSTAR)
		copy(b.STAR().Version(), versionUSTAR)
		copy(b.STAR().Trailer(), trailerSTAR)
	case format.has(FormatUSTAR | FormatPAX):
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
		unsigned += int64(c)
		signed += int64(int8(c))
	}
	return unsigned, signed
}

// Reset clears the block with all zeros.
func (b *block) Reset() {
	*b = block{}
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

func (s sparseArray) Entry(i int) sparseElem { return (sparseElem)(s[i*24:]) }
func (s sparseArray) IsExtended() []byte     { return s[24*s.MaxEntries():][:1] }
func (s sparseArray) MaxEntries() int        { return len(s) / 24 }

type sparseElem []byte

func (s sparseElem) Offset() []byte { return s[00:][:12] }
func (s sparseElem) Length() []byte { return s[12:][:12] }
