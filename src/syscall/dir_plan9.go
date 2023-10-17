// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9 directory marshaling. See intro(5).

package syscall

import "errors"

var (
	ErrShortStat = errors.New("stat buffer too short")
	ErrBadStat   = errors.New("malformed stat buffer")
	ErrBadName   = errors.New("bad character in file name")
)

// A Qid represents a 9P server's unique identification for a file.
type Qid struct {
	Path uint64 // the file server's unique identification for the file
	Vers uint32 // version number for given Path
	Type uint8  // the type of the file (syscall.QTDIR for example)
}

// A Dir contains the metadata for a file.
type Dir struct {
	// system-modified data
	Type uint16 // server type
	Dev  uint32 // server subtype

	// file data
	Qid    Qid    // unique id from server
	Mode   uint32 // permissions
	Atime  uint32 // last read time
	Mtime  uint32 // last write time
	Length int64  // file length
	Name   string // last element of path
	Uid    string // owner name
	Gid    string // group name
	Muid   string // last modifier name
}

var nullDir = Dir{
	Type: ^uint16(0),
	Dev:  ^uint32(0),
	Qid: Qid{
		Path: ^uint64(0),
		Vers: ^uint32(0),
		Type: ^uint8(0),
	},
	Mode:   ^uint32(0),
	Atime:  ^uint32(0),
	Mtime:  ^uint32(0),
	Length: ^int64(0),
}

// Null assigns special "don't touch" values to members of d to
// avoid modifying them during syscall.Wstat.
func (d *Dir) Null() { *d = nullDir }

// Marshal encodes a 9P stat message corresponding to d into b
//
// If there isn't enough space in b for a stat message, ErrShortStat is returned.
func (d *Dir) Marshal(b []byte) (n int, err error) {
	n = STATFIXLEN + len(d.Name) + len(d.Uid) + len(d.Gid) + len(d.Muid)
	if n > len(b) {
		return n, ErrShortStat
	}

	for _, c := range d.Name {
		if c == '/' {
			return n, ErrBadName
		}
	}

	b = pbit16(b, uint16(n)-2)
	b = pbit16(b, d.Type)
	b = pbit32(b, d.Dev)
	b = pbit8(b, d.Qid.Type)
	b = pbit32(b, d.Qid.Vers)
	b = pbit64(b, d.Qid.Path)
	b = pbit32(b, d.Mode)
	b = pbit32(b, d.Atime)
	b = pbit32(b, d.Mtime)
	b = pbit64(b, uint64(d.Length))
	b = pstring(b, d.Name)
	b = pstring(b, d.Uid)
	b = pstring(b, d.Gid)
	b = pstring(b, d.Muid)

	return n, nil
}

// UnmarshalDir decodes a single 9P stat message from b and returns the resulting Dir.
//
// If b is too small to hold a valid stat message, ErrShortStat is returned.
//
// If the stat message itself is invalid, ErrBadStat is returned.
func UnmarshalDir(b []byte) (*Dir, error) {
	if len(b) < STATFIXLEN {
		return nil, ErrShortStat
	}
	size, buf := gbit16(b)
	if len(b) != int(size)+2 {
		return nil, ErrBadStat
	}
	b = buf

	var d Dir
	d.Type, b = gbit16(b)
	d.Dev, b = gbit32(b)
	d.Qid.Type, b = gbit8(b)
	d.Qid.Vers, b = gbit32(b)
	d.Qid.Path, b = gbit64(b)
	d.Mode, b = gbit32(b)
	d.Atime, b = gbit32(b)
	d.Mtime, b = gbit32(b)

	n, b := gbit64(b)
	d.Length = int64(n)

	var ok bool
	if d.Name, b, ok = gstring(b); !ok {
		return nil, ErrBadStat
	}
	if d.Uid, b, ok = gstring(b); !ok {
		return nil, ErrBadStat
	}
	if d.Gid, b, ok = gstring(b); !ok {
		return nil, ErrBadStat
	}
	if d.Muid, b, ok = gstring(b); !ok {
		return nil, ErrBadStat
	}

	return &d, nil
}

// pbit8 copies the 8-bit number v to b and returns the remaining slice of b.
func pbit8(b []byte, v uint8) []byte {
	b[0] = byte(v)
	return b[1:]
}

// pbit16 copies the 16-bit number v to b in little-endian order and returns the remaining slice of b.
func pbit16(b []byte, v uint16) []byte {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	return b[2:]
}

// pbit32 copies the 32-bit number v to b in little-endian order and returns the remaining slice of b.
func pbit32(b []byte, v uint32) []byte {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	return b[4:]
}

// pbit64 copies the 64-bit number v to b in little-endian order and returns the remaining slice of b.
func pbit64(b []byte, v uint64) []byte {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
	return b[8:]
}

// pstring copies the string s to b, prepending it with a 16-bit length in little-endian order, and
// returning the remaining slice of b..
func pstring(b []byte, s string) []byte {
	b = pbit16(b, uint16(len(s)))
	n := copy(b, s)
	return b[n:]
}

// gbit8 reads an 8-bit number from b and returns it with the remaining slice of b.
func gbit8(b []byte) (uint8, []byte) {
	return uint8(b[0]), b[1:]
}

// gbit16 reads a 16-bit number in little-endian order from b and returns it with the remaining slice of b.
//
//go:nosplit
func gbit16(b []byte) (uint16, []byte) {
	return uint16(b[0]) | uint16(b[1])<<8, b[2:]
}

// gbit32 reads a 32-bit number in little-endian order from b and returns it with the remaining slice of b.
func gbit32(b []byte) (uint32, []byte) {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24, b[4:]
}

// gbit64 reads a 64-bit number in little-endian order from b and returns it with the remaining slice of b.
func gbit64(b []byte) (uint64, []byte) {
	lo := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	hi := uint32(b[4]) | uint32(b[5])<<8 | uint32(b[6])<<16 | uint32(b[7])<<24
	return uint64(lo) | uint64(hi)<<32, b[8:]
}

// gstring reads a string from b, prefixed with a 16-bit length in little-endian order.
// It returns the string with the remaining slice of b and a boolean. If the length is
// greater than the number of bytes in b, the boolean will be false.
func gstring(b []byte) (string, []byte, bool) {
	n, b := gbit16(b)
	if int(n) > len(b) {
		return "", b, false
	}
	return string(b[:n]), b[n:], true
}
