// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xcoff

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

const (
	SAIAMAG   = 0x8
	AIAFMAG   = "`\n"
	AIAMAG    = "<aiaff>\n"
	AIAMAGBIG = "<bigaf>\n"

	// Sizeof
	FL_HSZ_BIG = 0x80
	AR_HSZ_BIG = 0x70
)

type bigarFileHeader struct {
	Flmagic    [SAIAMAG]byte // Archive magic string
	Flmemoff   [20]byte      // Member table offset
	Flgstoff   [20]byte      // 32-bits global symtab offset
	Flgst64off [20]byte      // 64-bits global symtab offset
	Flfstmoff  [20]byte      // First member offset
	Fllstmoff  [20]byte      // Last member offset
	Flfreeoff  [20]byte      // First member on free list offset
}

type bigarMemberHeader struct {
	Arsize   [20]byte // File member size
	Arnxtmem [20]byte // Next member pointer
	Arprvmem [20]byte // Previous member pointer
	Ardate   [12]byte // File member date
	Aruid    [12]byte // File member uid
	Argid    [12]byte // File member gid
	Armode   [12]byte // File member mode (octal)
	Arnamlen [4]byte  // File member name length
	// _ar_nam is removed because it's easier to get name without it.
}

// Archive represents an open AIX big archive.
type Archive struct {
	ArchiveHeader
	Members []*Member

	closer io.Closer
}

// MemberHeader holds information about a big archive file header
type ArchiveHeader struct {
	magic string
}

// Member represents a member of an AIX big archive.
type Member struct {
	MemberHeader
	sr *io.SectionReader
}

// MemberHeader holds information about a big archive member
type MemberHeader struct {
	Name string
	Size uint64
}

// OpenArchive opens the named archive using os.Open and prepares it for use
// as an AIX big archive.
func OpenArchive(name string) (*Archive, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	arch, err := NewArchive(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	arch.closer = f
	return arch, nil
}

// Close closes the Archive.
// If the Archive was created using NewArchive directly instead of OpenArchive,
// Close has no effect.
func (a *Archive) Close() error {
	var err error
	if a.closer != nil {
		err = a.closer.Close()
		a.closer = nil
	}
	return err
}

// NewArchive creates a new Archive for accessing an AIX big archive in an underlying reader.
func NewArchive(r io.ReaderAt) (*Archive, error) {
	parseDecimalBytes := func(b []byte) (int64, error) {
		return strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	}
	sr := io.NewSectionReader(r, 0, 1<<63-1)

	// Read File Header
	var magic [SAIAMAG]byte
	if _, err := sr.ReadAt(magic[:], 0); err != nil {
		return nil, err
	}

	arch := new(Archive)
	switch string(magic[:]) {
	case AIAMAGBIG:
		arch.magic = string(magic[:])
	case AIAMAG:
		return nil, fmt.Errorf("small AIX archive not supported")
	default:
		return nil, fmt.Errorf("unrecognised archive magic: 0x%x", magic)
	}

	var fhdr bigarFileHeader
	if _, err := sr.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	if err := binary.Read(sr, binary.BigEndian, &fhdr); err != nil {
		return nil, err
	}

	off, err := parseDecimalBytes(fhdr.Flfstmoff[:])
	if err != nil {
		return nil, fmt.Errorf("error parsing offset of first member in archive header(%q); %v", fhdr, err)
	}

	if off == 0 {
		// Occurs if the archive is empty.
		return arch, nil
	}

	lastoff, err := parseDecimalBytes(fhdr.Fllstmoff[:])
	if err != nil {
		return nil, fmt.Errorf("error parsing offset of first member in archive header(%q); %v", fhdr, err)
	}

	// Read members
	for {
		// Read Member Header
		// The member header is normally 2 bytes larger. But it's easier
		// to read the name if the header is read without _ar_nam.
		// However, AIAFMAG must be read afterward.
		if _, err := sr.Seek(off, io.SeekStart); err != nil {
			return nil, err
		}

		var mhdr bigarMemberHeader
		if err := binary.Read(sr, binary.BigEndian, &mhdr); err != nil {
			return nil, err
		}

		member := new(Member)
		arch.Members = append(arch.Members, member)

		size, err := parseDecimalBytes(mhdr.Arsize[:])
		if err != nil {
			return nil, fmt.Errorf("error parsing size in member header(%q); %v", mhdr, err)
		}
		member.Size = uint64(size)

		// Read name
		namlen, err := parseDecimalBytes(mhdr.Arnamlen[:])
		if err != nil {
			return nil, fmt.Errorf("error parsing name length in member header(%q); %v", mhdr, err)
		}
		name := make([]byte, namlen)
		if err := binary.Read(sr, binary.BigEndian, name); err != nil {
			return nil, err
		}
		member.Name = string(name)

		fileoff := off + AR_HSZ_BIG + namlen
		if fileoff&1 != 0 {
			fileoff++
			if _, err := sr.Seek(1, io.SeekCurrent); err != nil {
				return nil, err
			}
		}

		// Read AIAFMAG string
		var fmag [2]byte
		if err := binary.Read(sr, binary.BigEndian, &fmag); err != nil {
			return nil, err
		}
		if string(fmag[:]) != AIAFMAG {
			return nil, fmt.Errorf("AIAFMAG not found after member header")
		}

		fileoff += 2 // Add the two bytes of AIAFMAG
		member.sr = io.NewSectionReader(sr, fileoff, size)

		if off == lastoff {
			break
		}
		off, err = parseDecimalBytes(mhdr.Arnxtmem[:])
		if err != nil {
			return nil, fmt.Errorf("error parsing offset of first member in archive header(%q); %v", fhdr, err)
		}

	}

	return arch, nil
}

// GetFile returns the XCOFF file defined by member name.
// FIXME: This doesn't work if an archive has two members with the same
// name which can occur if an archive has both 32-bits and 64-bits files.
func (arch *Archive) GetFile(name string) (*File, error) {
	for _, mem := range arch.Members {
		if mem.Name == name {
			return NewFile(mem.sr)
		}
	}
	return nil, fmt.Errorf("unknown member %s in archive", name)
}
