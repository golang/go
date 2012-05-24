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

import (
	"errors"
	"fmt"
	"os"
	"time"
)

const (
	blockSize = 512

	// Types
	TypeReg           = '0'    // regular file
	TypeRegA          = '\x00' // regular file
	TypeLink          = '1'    // hard link
	TypeSymlink       = '2'    // symbolic link
	TypeChar          = '3'    // character device node
	TypeBlock         = '4'    // block device node
	TypeDir           = '5'    // directory
	TypeFifo          = '6'    // fifo node
	TypeCont          = '7'    // reserved
	TypeXHeader       = 'x'    // extended header
	TypeXGlobalHeader = 'g'    // global extended header
)

// A Header represents a single header in a tar archive.
// Some fields may not be populated.
type Header struct {
	Name       string    // name of header file entry
	Mode       int64     // permission and mode bits
	Uid        int       // user id of owner
	Gid        int       // group id of owner
	Size       int64     // length in bytes
	ModTime    time.Time // modified time
	Typeflag   byte      // type of header entry
	Linkname   string    // target name of link
	Uname      string    // user name of owner
	Gname      string    // group name of owner
	Devmajor   int64     // major number of character or block device
	Devminor   int64     // minor number of character or block device
	AccessTime time.Time // access time
	ChangeTime time.Time // status change time
}

// sysStat, if non-nil, populates h from system-dependent fields of fi.
var sysStat func(fi os.FileInfo, h *Header) error

// Mode constants from the tar spec.
const (
	c_ISDIR  = 040000
	c_ISFIFO = 010000
	c_ISREG  = 0100000
	c_ISLNK  = 0120000
	c_ISBLK  = 060000
	c_ISCHR  = 020000
	c_ISSOCK = 0140000
)

// FileInfoHeader creates a partially-populated Header from fi.
// If fi describes a symlink, FileInfoHeader records link as the link target.
func FileInfoHeader(fi os.FileInfo, link string) (*Header, error) {
	if fi == nil {
		return nil, errors.New("tar: FileInfo is nil")
	}
	h := &Header{
		Name:    fi.Name(),
		ModTime: fi.ModTime(),
		Mode:    int64(fi.Mode().Perm()), // or'd with c_IS* constants later
	}
	switch {
	case fi.Mode()&os.ModeType == 0:
		h.Mode |= c_ISREG
		h.Typeflag = TypeReg
		h.Size = fi.Size()
	case fi.IsDir():
		h.Typeflag = TypeDir
		h.Mode |= c_ISDIR
	case fi.Mode()&os.ModeSymlink != 0:
		h.Typeflag = TypeSymlink
		h.Mode |= c_ISLNK
		h.Linkname = link
	case fi.Mode()&os.ModeDevice != 0:
		if fi.Mode()&os.ModeCharDevice != 0 {
			h.Mode |= c_ISCHR
			h.Typeflag = TypeChar
		} else {
			h.Mode |= c_ISBLK
			h.Typeflag = TypeBlock
		}
	case fi.Mode()&os.ModeSocket != 0:
		h.Mode |= c_ISSOCK
	default:
		return nil, fmt.Errorf("archive/tar: unknown file mode %v", fi.Mode())
	}
	if sysStat != nil {
		return h, sysStat(fi, h)
	}
	return h, nil
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
