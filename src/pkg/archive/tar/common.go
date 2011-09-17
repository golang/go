// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tar implements access to tar archives.
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
	TypeReg           = '0'    // regular file.
	TypeRegA          = '\x00' // regular file.
	TypeLink          = '1'    // hard link.
	TypeSymlink       = '2'    // symbolic link.
	TypeChar          = '3'    // character device node.
	TypeBlock         = '4'    // block device node.
	TypeDir           = '5'    // directory.
	TypeFifo          = '6'    // fifo node.
	TypeCont          = '7'    // reserved.
	TypeXHeader       = 'x'    // extended header.
	TypeXGlobalHeader = 'g'    // global extended header.
)

// A Header represents a single header in a tar archive.
// Some fields may not be populated.
type Header struct {
	Name     string // name of header file entry.
	Mode     int64  // permission and mode bits.
	Uid      int    // user id of owner.
	Gid      int    // group id of owner.
	Size     int64  // length in bytes.
	Mtime    int64  // modified time; seconds since epoch.
	Typeflag byte   // type of header entry.
	Linkname string // target name of link.
	Uname    string // user name of owner.
	Gname    string // group name of owner.
	Devmajor int64  // major number of character or block device.
	Devminor int64  // minor number of character or block device.
	Atime    int64  // access time; seconds since epoch.
	Ctime    int64  // status change time; seconds since epoch.

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
