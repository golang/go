// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package zip provides support for reading and writing ZIP archives.

See: http://www.pkware.com/documents/casestudies/APPNOTE.TXT

This package does not support ZIP64 or disk spanning.
*/
package zip

import "os"

// Compression methods.
const (
	Store   uint16 = 0
	Deflate uint16 = 8
)

const (
	fileHeaderSignature      = 0x04034b50
	directoryHeaderSignature = 0x02014b50
	directoryEndSignature    = 0x06054b50
	dataDescriptorLen        = 12
)

type FileHeader struct {
	Name             string
	CreatorVersion   uint16
	ReaderVersion    uint16
	Flags            uint16
	Method           uint16
	ModifiedTime     uint16
	ModifiedDate     uint16
	CRC32            uint32
	CompressedSize   uint32
	UncompressedSize uint32
	Extra            []byte
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

func recoverError(err *os.Error) {
	if e := recover(); e != nil {
		if osErr, ok := e.(os.Error); ok {
			*err = osErr
			return
		}
		panic(e)
	}
}
