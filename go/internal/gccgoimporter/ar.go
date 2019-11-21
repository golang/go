// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Except for this comment, this file is a verbatim copy of the file
// with the same name in $GOROOT/src/go/internal/gccgoimporter.

package gccgoimporter

import (
	"bytes"
	"debug/elf"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Magic strings for different archive file formats.
const (
	armag  = "!<arch>\n"
	armagt = "!<thin>\n"
	armagb = "<bigaf>\n"
)

// Offsets and sizes for fields in a standard archive header.
const (
	arNameOff  = 0
	arNameSize = 16
	arDateOff  = arNameOff + arNameSize
	arDateSize = 12
	arUIDOff   = arDateOff + arDateSize
	arUIDSize  = 6
	arGIDOff   = arUIDOff + arUIDSize
	arGIDSize  = 6
	arModeOff  = arGIDOff + arGIDSize
	arModeSize = 8
	arSizeOff  = arModeOff + arModeSize
	arSizeSize = 10
	arFmagOff  = arSizeOff + arSizeSize
	arFmagSize = 2

	arHdrSize = arFmagOff + arFmagSize
)

// The contents of the fmag field of a standard archive header.
const arfmag = "`\n"

// arExportData takes an archive file and returns a ReadSeeker for the
// export data in that file. This assumes that there is only one
// object in the archive containing export data, which is not quite
// what gccgo does; gccgo concatenates together all the export data
// for all the objects in the file.  In practice that case does not arise.
func arExportData(archive io.ReadSeeker) (io.ReadSeeker, error) {
	if _, err := archive.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	var buf [len(armag)]byte
	if _, err := archive.Read(buf[:]); err != nil {
		return nil, err
	}

	switch string(buf[:]) {
	case armag:
		return standardArExportData(archive)
	case armagt:
		return nil, errors.New("unsupported thin archive")
	case armagb:
		return nil, errors.New("unsupported AIX big archive")
	default:
		return nil, fmt.Errorf("unrecognized archive file format %q", buf[:])
	}
}

// standardArExportData returns export data form a standard archive.
func standardArExportData(archive io.ReadSeeker) (io.ReadSeeker, error) {
	off := int64(len(armag))
	for {
		var hdrBuf [arHdrSize]byte
		if _, err := archive.Read(hdrBuf[:]); err != nil {
			return nil, err
		}
		off += arHdrSize

		if !bytes.Equal(hdrBuf[arFmagOff:arFmagOff+arFmagSize], []byte(arfmag)) {
			return nil, fmt.Errorf("archive header format header (%q)", hdrBuf[:])
		}

		size, err := strconv.ParseInt(strings.TrimSpace(string(hdrBuf[arSizeOff:arSizeOff+arSizeSize])), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing size in archive header (%q): %v", hdrBuf[:], err)
		}

		fn := hdrBuf[arNameOff : arNameOff+arNameSize]
		if fn[0] == '/' && (fn[1] == ' ' || fn[1] == '/' || bytes.Equal(fn[:8], []byte("/SYM64/ "))) {
			// Archive symbol table or extended name table,
			// which we don't care about.
		} else {
			archiveAt := readerAtFromSeeker(archive)
			ret, err := elfFromAr(io.NewSectionReader(archiveAt, off, size))
			if ret != nil || err != nil {
				return ret, err
			}
		}

		if size&1 != 0 {
			size++
		}
		off += size
		if _, err := archive.Seek(off, io.SeekStart); err != nil {
			return nil, err
		}
	}
}

// elfFromAr tries to get export data from an archive member as an ELF file.
// If there is no export data, this returns nil, nil.
func elfFromAr(member *io.SectionReader) (io.ReadSeeker, error) {
	ef, err := elf.NewFile(member)
	if err != nil {
		return nil, err
	}
	sec := ef.Section(".go_export")
	if sec == nil {
		return nil, nil
	}
	return sec.Open(), nil
}

// readerAtFromSeeker turns an io.ReadSeeker into an io.ReaderAt.
// This is only safe because there won't be any concurrent seeks
// while this code is executing.
func readerAtFromSeeker(rs io.ReadSeeker) io.ReaderAt {
	if ret, ok := rs.(io.ReaderAt); ok {
		return ret
	}
	return seekerReadAt{rs}
}

type seekerReadAt struct {
	seeker io.ReadSeeker
}

func (sra seekerReadAt) ReadAt(p []byte, off int64) (int, error) {
	if _, err := sra.seeker.Seek(off, io.SeekStart); err != nil {
		return 0, err
	}
	return sra.seeker.Read(p)
}
