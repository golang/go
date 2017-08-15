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
//   http://pubs.opengroup.org/onlinepubs/9699919799/utilities/pax.html
package tar

import (
	"errors"
	"fmt"
	"os"
	"path"
	"strconv"
	"time"
)

// BUG: Use of the Uid and Gid fields in Header could overflow on 32-bit
// architectures. If a large value is encountered when decoding, the result
// stored in Header will be the truncated version.

var (
	ErrHeader          = errors.New("tar: invalid tar header")
	ErrWriteTooLong    = errors.New("tar: write too long")
	ErrFieldTooLong    = errors.New("tar: header field too long")
	ErrWriteAfterClose = errors.New("tar: write after close")
)

// Header type flags.
const (
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
	TypeGNULongName   = 'L'    // Next file has a long name
	TypeGNULongLink   = 'K'    // Next file symlinks to a file w/ a long name
	TypeGNUSparse     = 'S'    // sparse file
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
	Xattrs     map[string]string
}

// FileInfo returns an os.FileInfo for the Header.
func (h *Header) FileInfo() os.FileInfo {
	return headerFileInfo{h}
}

// allowedFormats determines which formats can be used. The value returned
// is the logical OR of multiple possible formats. If the value is
// formatUnknown, then the input Header cannot be encoded.
//
// As a by-product of checking the fields, this function returns paxHdrs, which
// contain all fields that could not be directly encoded.
func (h *Header) allowedFormats() (format int, paxHdrs map[string]string) {
	format = formatUSTAR | formatPAX | formatGNU
	paxHdrs = make(map[string]string)

	verifyString := func(s string, size int, gnuLong bool, paxKey string) {
		// NUL-terminator is optional for path and linkpath.
		// Technically, it is required for uname and gname,
		// but neither GNU nor BSD tar checks for it.
		tooLong := len(s) > size
		if !isASCII(s) || (tooLong && !gnuLong) {
			// TODO(dsnet): GNU supports UTF-8 (without NUL) for strings.
			format &^= formatGNU // No GNU
		}
		if !isASCII(s) || tooLong {
			// TODO(dsnet): If the path is splittable, it is possible to still
			// use the USTAR format.
			format &^= formatUSTAR // No USTAR
			if paxKey == paxNone {
				format &^= formatPAX // No PAX
			} else {
				paxHdrs[paxKey] = s
			}
		}
	}
	verifyNumeric := func(n int64, size int, paxKey string) {
		if !fitsInBase256(size, n) {
			format &^= formatGNU // No GNU
		}
		if !fitsInOctal(size, n) {
			format &^= formatUSTAR // No USTAR
			if paxKey == paxNone {
				format &^= formatPAX // No PAX
			} else {
				paxHdrs[paxKey] = strconv.FormatInt(n, 10)
			}
		}
	}
	verifyTime := func(ts time.Time, size int, ustarField bool, paxKey string) {
		if ts.IsZero() {
			return // Always okay
		}
		needsNano := ts.Nanosecond() != 0
		if !fitsInBase256(size, ts.Unix()) || needsNano {
			format &^= formatGNU // No GNU
		}
		if !fitsInOctal(size, ts.Unix()) || needsNano || !ustarField {
			format &^= formatUSTAR // No USTAR
			if paxKey == paxNone {
				format &^= formatPAX // No PAX
			} else {
				paxHdrs[paxKey] = formatPAXTime(ts)
			}
		}
	}

	// TODO(dsnet): Add GNU long name support.
	const supportGNULong = false

	var blk block
	v7 := blk.V7()
	ustar := blk.USTAR()
	gnu := blk.GNU()
	verifyString(h.Name, len(v7.Name()), supportGNULong, paxPath)
	verifyString(h.Linkname, len(v7.LinkName()), supportGNULong, paxLinkpath)
	verifyString(h.Uname, len(ustar.UserName()), false, paxUname)
	verifyString(h.Gname, len(ustar.GroupName()), false, paxGname)
	verifyNumeric(h.Mode, len(v7.Mode()), paxNone)
	verifyNumeric(int64(h.Uid), len(v7.UID()), paxUid)
	verifyNumeric(int64(h.Gid), len(v7.GID()), paxGid)
	verifyNumeric(h.Size, len(v7.Size()), paxSize)
	verifyNumeric(h.Devmajor, len(ustar.DevMajor()), paxNone)
	verifyNumeric(h.Devminor, len(ustar.DevMinor()), paxNone)
	verifyTime(h.ModTime, len(v7.ModTime()), true, paxMtime)
	verifyTime(h.AccessTime, len(gnu.AccessTime()), false, paxAtime)
	verifyTime(h.ChangeTime, len(gnu.ChangeTime()), false, paxCtime)

	if !isHeaderOnlyType(h.Typeflag) && h.Size < 0 {
		return formatUnknown, nil
	}
	if len(h.Xattrs) > 0 {
		for k, v := range h.Xattrs {
			paxHdrs[paxXattr+k] = v
		}
		format &= formatPAX // PAX only
	}
	for k, v := range paxHdrs {
		// Forbid empty values (which represent deletion) since usage of
		// them are non-sensible without global PAX record support.
		if !validPAXRecord(k, v) || v == "" {
			return formatUnknown, nil // Invalid PAX key
		}
	}
	return format, paxHdrs
}

// headerFileInfo implements os.FileInfo.
type headerFileInfo struct {
	h *Header
}

func (fi headerFileInfo) Size() int64        { return fi.h.Size }
func (fi headerFileInfo) IsDir() bool        { return fi.Mode().IsDir() }
func (fi headerFileInfo) ModTime() time.Time { return fi.h.ModTime }
func (fi headerFileInfo) Sys() interface{}   { return fi.h }

// Name returns the base name of the file.
func (fi headerFileInfo) Name() string {
	if fi.IsDir() {
		return path.Base(path.Clean(fi.h.Name))
	}
	return path.Base(fi.h.Name)
}

// Mode returns the permission and mode bits for the headerFileInfo.
func (fi headerFileInfo) Mode() (mode os.FileMode) {
	// Set file permission bits.
	mode = os.FileMode(fi.h.Mode).Perm()

	// Set setuid, setgid and sticky bits.
	if fi.h.Mode&c_ISUID != 0 {
		// setuid
		mode |= os.ModeSetuid
	}
	if fi.h.Mode&c_ISGID != 0 {
		// setgid
		mode |= os.ModeSetgid
	}
	if fi.h.Mode&c_ISVTX != 0 {
		// sticky
		mode |= os.ModeSticky
	}

	// Set file mode bits.
	// clear perm, setuid, setgid and sticky bits.
	m := os.FileMode(fi.h.Mode) &^ 07777
	if m == c_ISDIR {
		// directory
		mode |= os.ModeDir
	}
	if m == c_ISFIFO {
		// named pipe (FIFO)
		mode |= os.ModeNamedPipe
	}
	if m == c_ISLNK {
		// symbolic link
		mode |= os.ModeSymlink
	}
	if m == c_ISBLK {
		// device file
		mode |= os.ModeDevice
	}
	if m == c_ISCHR {
		// Unix character device
		mode |= os.ModeDevice
		mode |= os.ModeCharDevice
	}
	if m == c_ISSOCK {
		// Unix domain socket
		mode |= os.ModeSocket
	}

	switch fi.h.Typeflag {
	case TypeSymlink:
		// symbolic link
		mode |= os.ModeSymlink
	case TypeChar:
		// character device node
		mode |= os.ModeDevice
		mode |= os.ModeCharDevice
	case TypeBlock:
		// block device node
		mode |= os.ModeDevice
	case TypeDir:
		// directory
		mode |= os.ModeDir
	case TypeFifo:
		// fifo node
		mode |= os.ModeNamedPipe
	}

	return mode
}

// sysStat, if non-nil, populates h from system-dependent fields of fi.
var sysStat func(fi os.FileInfo, h *Header) error

const (
	// Mode constants from the USTAR spec:
	// See http://pubs.opengroup.org/onlinepubs/9699919799/utilities/pax.html#tag_20_92_13_06
	c_ISUID = 04000 // Set uid
	c_ISGID = 02000 // Set gid
	c_ISVTX = 01000 // Save text (sticky bit)

	// Common Unix mode constants; these are not defined in any common tar standard.
	// Header.FileInfo understands these, but FileInfoHeader will never produce these.
	c_ISDIR  = 040000  // Directory
	c_ISFIFO = 010000  // FIFO
	c_ISREG  = 0100000 // Regular file
	c_ISLNK  = 0120000 // Symbolic link
	c_ISBLK  = 060000  // Block special file
	c_ISCHR  = 020000  // Character special file
	c_ISSOCK = 0140000 // Socket
)

// Keywords for the PAX Extended Header
const (
	paxAtime    = "atime"
	paxCharset  = "charset"
	paxComment  = "comment"
	paxCtime    = "ctime" // please note that ctime is not a valid pax header.
	paxGid      = "gid"
	paxGname    = "gname"
	paxLinkpath = "linkpath"
	paxMtime    = "mtime"
	paxPath     = "path"
	paxSize     = "size"
	paxUid      = "uid"
	paxUname    = "uname"
	paxXattr    = "SCHILY.xattr."
	paxNone     = ""
)

// FileInfoHeader creates a partially-populated Header from fi.
// If fi describes a symlink, FileInfoHeader records link as the link target.
// If fi describes a directory, a slash is appended to the name.
// Because os.FileInfo's Name method returns only the base name of
// the file it describes, it may be necessary to modify the Name field
// of the returned header to provide the full path name of the file.
func FileInfoHeader(fi os.FileInfo, link string) (*Header, error) {
	if fi == nil {
		return nil, errors.New("tar: FileInfo is nil")
	}
	fm := fi.Mode()
	h := &Header{
		Name:    fi.Name(),
		ModTime: fi.ModTime(),
		Mode:    int64(fm.Perm()), // or'd with c_IS* constants later
	}
	switch {
	case fm.IsRegular():
		h.Typeflag = TypeReg
		h.Size = fi.Size()
	case fi.IsDir():
		h.Typeflag = TypeDir
		h.Name += "/"
	case fm&os.ModeSymlink != 0:
		h.Typeflag = TypeSymlink
		h.Linkname = link
	case fm&os.ModeDevice != 0:
		if fm&os.ModeCharDevice != 0 {
			h.Typeflag = TypeChar
		} else {
			h.Typeflag = TypeBlock
		}
	case fm&os.ModeNamedPipe != 0:
		h.Typeflag = TypeFifo
	case fm&os.ModeSocket != 0:
		return nil, fmt.Errorf("tar: sockets not supported")
	default:
		return nil, fmt.Errorf("tar: unknown file mode %v", fm)
	}
	if fm&os.ModeSetuid != 0 {
		h.Mode |= c_ISUID
	}
	if fm&os.ModeSetgid != 0 {
		h.Mode |= c_ISGID
	}
	if fm&os.ModeSticky != 0 {
		h.Mode |= c_ISVTX
	}
	// If possible, populate additional fields from OS-specific
	// FileInfo fields.
	if sys, ok := fi.Sys().(*Header); ok {
		// This FileInfo came from a Header (not the OS). Use the
		// original Header to populate all remaining fields.
		h.Uid = sys.Uid
		h.Gid = sys.Gid
		h.Uname = sys.Uname
		h.Gname = sys.Gname
		h.AccessTime = sys.AccessTime
		h.ChangeTime = sys.ChangeTime
		if sys.Xattrs != nil {
			h.Xattrs = make(map[string]string)
			for k, v := range sys.Xattrs {
				h.Xattrs[k] = v
			}
		}
		if sys.Typeflag == TypeLink {
			// hard link
			h.Typeflag = TypeLink
			h.Size = 0
			h.Linkname = sys.Linkname
		}
	}
	if sysStat != nil {
		return h, sysStat(fi, h)
	}
	return h, nil
}

// isHeaderOnlyType checks if the given type flag is of the type that has no
// data section even if a size is specified.
func isHeaderOnlyType(flag byte) bool {
	switch flag {
	case TypeLink, TypeSymlink, TypeChar, TypeBlock, TypeDir, TypeFifo:
		return true
	default:
		return false
	}
}
