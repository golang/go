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
	creatorUnix = 3
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

// ModTime returns the modification time.
// The resolution is 2s.
func (h *FileHeader) ModTime() time.Time {
	return msDosTimeToTime(h.ModifiedDate, h.ModifiedTime)
}

// Mode returns the permission and mode bits for the FileHeader.
// An error is returned in case the information is not available.
func (h *FileHeader) Mode() (mode uint32, err error) {
	if h.CreatorVersion>>8 == creatorUnix {
		return h.ExternalAttrs >> 16, nil
	}
	return 0, errors.New("file mode not available")
}

// SetMode changes the permission and mode bits for the FileHeader.
func (h *FileHeader) SetMode(mode uint32) {
	h.CreatorVersion = h.CreatorVersion&0xff | creatorUnix<<8
	h.ExternalAttrs = mode << 16
}
