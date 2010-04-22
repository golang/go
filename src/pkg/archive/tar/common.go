// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The tar package implements access to tar archives.
// It aims to cover most of the variations, including those produced
// by GNU and BSD tars.
//
// References:
//   http://www.freebsd.org/cgi/man.cgi?query=tar&sektion=5
//   http://www.gnu.org/software/tar/manual/html_node/Standard.html
package tar

const (
	blockSize = 512

	// Types
	TypeReg           = '0'
	TypeRegA          = '\x00'
	TypeLink          = '1'
	TypeSymlink       = '2'
	TypeChar          = '3'
	TypeBlock         = '4'
	TypeDir           = '5'
	TypeFifo          = '6'
	TypeCont          = '7'
	TypeXHeader       = 'x'
	TypeXGlobalHeader = 'g'
)

// A Header represents a single header in a tar archive.
// Some fields may not be populated.
type Header struct {
	Name     string
	Mode     int64
	Uid      int
	Gid      int
	Size     int64
	Mtime    int64
	Typeflag byte
	Linkname string
	Uname    string
	Gname    string
	Devmajor int64
	Devminor int64
	Atime    int64
	Ctime    int64
}

var zeroBlock = make([]byte, blockSize)

// POSIX specifies a sum of the unsigned byte values, but the Sun tar uses signed byte values.
// We compute and return both.
func checksum(header []byte) (unsigned int64, signed int64) {
	for i := 0; i < len(header); i++ {
		if i == 148 {
			// The chksum field (header[148:156]) is special: it should be treated as space bytes.
			unsigned += ' ' * 8
			signed += ' ' * 8
			i += 7
			continue
		}
		unsigned += int64(header[i])
		signed += int64(int8(header[i]))
	}
	return
}

type slicer []byte

func (sp *slicer) next(n int) (b []byte) {
	s := *sp
	b, *sp = s[0:n], s[n:]
	return
}
