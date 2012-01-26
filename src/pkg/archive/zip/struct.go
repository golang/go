// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package zip provides support for reading and writing ZIP archives.

See: http://www.pkware.com/documents/casestudies/APPNOTE.TXT

This package does not support ZIP64 or disk spanning.
*/
package zip

import (
	"errors"
	"os"
	"time"
)

// Compression methods.
const (
	Store   uint16 = 0
	Deflate uint16 = 8
)

const (
	fileHeaderSignature      = 0x04034b50
	directoryHeaderSignature = 0x02014b50
	directoryEndSignature    = 0x06054b50
	fileHeaderLen            = 30 // + filename + extra
	directoryHeaderLen       = 46 // + filename + extra + comment
	directoryEndLen          = 22 // + comment
	dataDescriptorLen        = 12

	// Constants for the first byte in CreatorVersion
	creatorFAT    = 0
	creatorUnix   = 3
	creatorNTFS   = 11
	creatorVFAT   = 14
	creatorMacOSX = 19
)

type FileHeader struct {
	Name             string
	CreatorVersion   uint16
	ReaderVersion    uint16
	Flags            uint16
	Method           uint16
	ModifiedTime     uint16 // MS-DOS time
	ModifiedDate     uint16 // MS-DOS date
	CRC32            uint32
	CompressedSize   uint32
	UncompressedSize uint32
	Extra            []byte
	ExternalAttrs    uint32 // Meaning depends on CreatorVersion
	Comment          string
}

// FileInfo returns an os.FileInfo for the FileHeader.
func (fh *FileHeader) FileInfo() os.FileInfo {
	return headerFileInfo{fh}
}

// headerFileInfo implements os.FileInfo.
type headerFileInfo struct {
	fh *FileHeader
}

func (fi headerFileInfo) Name() string       { return fi.fh.Name }
func (fi headerFileInfo) Size() int64        { return int64(fi.fh.UncompressedSize) }
func (fi headerFileInfo) IsDir() bool        { return fi.Mode().IsDir() }
func (fi headerFileInfo) ModTime() time.Time { return fi.fh.ModTime() }
func (fi headerFileInfo) Mode() os.FileMode  { return fi.fh.Mode() }

// FileInfoHeader creates a partially-populated FileHeader from an
// os.FileInfo.
func FileInfoHeader(fi os.FileInfo) (*FileHeader, error) {
	size := fi.Size()
	if size > (1<<32 - 1) {
		return nil, errors.New("zip: file over 4GB")
	}
	fh := &FileHeader{
		Name:             fi.Name(),
		UncompressedSize: uint32(size),
	}
	fh.SetModTime(fi.ModTime())
	fh.SetMode(fi.Mode())
	return fh, nil
}

type directoryEnd struct {
	diskNbr            uint16 // unused
	dirDiskNbr         uint16 // unused
	dirRecordsThisDisk uint16 // unused
	directoryRecords   uint16
	directorySize      uint32
	directoryOffset    uint32 // relative to file
	commentLen         uint16
	comment            string
}

func recoverError(errp *error) {
	if e := recover(); e != nil {
		if err, ok := e.(error); ok {
			*errp = err
			return
		}
		panic(e)
	}
}

// msDosTimeToTime converts an MS-DOS date and time into a time.Time.
// The resolution is 2s.
// See: http://msdn.microsoft.com/en-us/library/ms724247(v=VS.85).aspx
func msDosTimeToTime(dosDate, dosTime uint16) time.Time {
	return time.Date(
		// date bits 0-4: day of month; 5-8: month; 9-15: years since 1980
		int(dosDate>>9+1980),
		time.Month(dosDate>>5&0xf),
		int(dosDate&0x1f),

		// time bits 0-4: second/2; 5-10: minute; 11-15: hour
		int(dosTime>>11),
		int(dosTime>>5&0x3f),
		int(dosTime&0x1f*2),
		0, // nanoseconds

		time.UTC,
	)
}

// timeToMsDosTime converts a time.Time to an MS-DOS date and time.
// The resolution is 2s.
// See: http://msdn.microsoft.com/en-us/library/ms724274(v=VS.85).aspx
func timeToMsDosTime(t time.Time) (fDate uint16, fTime uint16) {
	t = t.In(time.UTC)
	fDate = uint16(t.Day() + int(t.Month())<<5 + (t.Year()-1980)<<9)
	fTime = uint16(t.Second()/2 + t.Minute()<<5 + t.Hour()<<11)
	return
}

// ModTime returns the modification time.
// The resolution is 2s.
func (h *FileHeader) ModTime() time.Time {
	return msDosTimeToTime(h.ModifiedDate, h.ModifiedTime)
}

// SetModTime sets the ModifiedTime and ModifiedDate fields to the given time.
// The resolution is 2s.
func (h *FileHeader) SetModTime(t time.Time) {
	h.ModifiedDate, h.ModifiedTime = timeToMsDosTime(t)
}

// traditional names for Unix constants
const (
	s_IFMT  = 0xf000
	s_IFDIR = 0x4000
	s_IFREG = 0x8000
	s_ISUID = 0x800
	s_ISGID = 0x400

	msdosDir      = 0x10
	msdosReadOnly = 0x01
)

// Mode returns the permission and mode bits for the FileHeader.
func (h *FileHeader) Mode() (mode os.FileMode) {
	switch h.CreatorVersion >> 8 {
	case creatorUnix, creatorMacOSX:
		mode = unixModeToFileMode(h.ExternalAttrs >> 16)
	case creatorNTFS, creatorVFAT, creatorFAT:
		mode = msdosModeToFileMode(h.ExternalAttrs)
	}
	if len(h.Name) > 0 && h.Name[len(h.Name)-1] == '/' {
		mode |= os.ModeDir
	}
	return mode
}

// SetMode changes the permission and mode bits for the FileHeader.
func (h *FileHeader) SetMode(mode os.FileMode) {
	h.CreatorVersion = h.CreatorVersion&0xff | creatorUnix<<8
	h.ExternalAttrs = fileModeToUnixMode(mode) << 16

	// set MSDOS attributes too, as the original zip does.
	if mode&os.ModeDir != 0 {
		h.ExternalAttrs |= msdosDir
	}
	if mode&0200 == 0 {
		h.ExternalAttrs |= msdosReadOnly
	}
}

func msdosModeToFileMode(m uint32) (mode os.FileMode) {
	if m&msdosDir != 0 {
		mode = os.ModeDir | 0777
	} else {
		mode = 0666
	}
	if m&msdosReadOnly != 0 {
		mode &^= 0222
	}
	return mode
}

func fileModeToUnixMode(mode os.FileMode) uint32 {
	var m uint32
	if mode&os.ModeDir != 0 {
		m = s_IFDIR
	} else {
		m = s_IFREG
	}
	if mode&os.ModeSetuid != 0 {
		m |= s_ISUID
	}
	if mode&os.ModeSetgid != 0 {
		m |= s_ISGID
	}
	return m | uint32(mode&0777)
}

func unixModeToFileMode(m uint32) os.FileMode {
	var mode os.FileMode
	if m&s_IFMT == s_IFDIR {
		mode |= os.ModeDir
	}
	if m&s_ISGID != 0 {
		mode |= os.ModeSetgid
	}
	if m&s_ISUID != 0 {
		mode |= os.ModeSetuid
	}
	return mode | os.FileMode(m&0777)
}
