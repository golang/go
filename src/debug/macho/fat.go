// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package macho

import (
	"encoding/binary"
	"fmt"
	"internal/saferio"
	"io"
	"os"
)

// A FatFile is a Mach-O universal binary that contains at least one architecture.
type FatFile struct {
	Magic  uint32
	Arches []FatArch
	closer io.Closer
}

// A FatArchHeader represents a fat header for a specific image architecture.
type FatArchHeader struct {
	Cpu    Cpu
	SubCpu uint32
	Offset uint32
	Size   uint32
	Align  uint32
}

const fatArchHeaderSize = 5 * 4

// A FatArch is a Mach-O File inside a FatFile.
type FatArch struct {
	FatArchHeader
	*File
}

// ErrNotFat is returned from [NewFatFile] or [OpenFat] when the file is not a
// universal binary but may be a thin binary, based on its magic number.
var ErrNotFat = &FormatError{0, "not a fat Mach-O file", nil}

// NewFatFile creates a new [FatFile] for accessing all the Mach-O images in a
// universal binary. The Mach-O binary is expected to start at position 0 in
// the ReaderAt.
func NewFatFile(r io.ReaderAt) (*FatFile, error) {
	var ff FatFile
	sr := io.NewSectionReader(r, 0, 1<<63-1)

	// Read the fat_header struct, which is always in big endian.
	// Start with the magic number.
	err := binary.Read(sr, binary.BigEndian, &ff.Magic)
	if err != nil {
		return nil, &FormatError{0, "error reading magic number", nil}
	} else if ff.Magic != MagicFat {
		// See if this is a Mach-O file via its magic number. The magic
		// must be converted to little endian first though.
		var buf [4]byte
		binary.BigEndian.PutUint32(buf[:], ff.Magic)
		leMagic := binary.LittleEndian.Uint32(buf[:])
		if leMagic == Magic32 || leMagic == Magic64 {
			return nil, ErrNotFat
		} else {
			return nil, &FormatError{0, "invalid magic number", nil}
		}
	}
	offset := int64(4)

	// Read the number of FatArchHeaders that come after the fat_header.
	var narch uint32
	err = binary.Read(sr, binary.BigEndian, &narch)
	if err != nil {
		return nil, &FormatError{offset, "invalid fat_header", nil}
	}
	offset += 4

	if narch < 1 {
		return nil, &FormatError{offset, "file contains no images", nil}
	}

	// Combine the Cpu and SubCpu (both uint32) into a uint64 to make sure
	// there are not duplicate architectures.
	seenArches := make(map[uint64]bool)
	// Make sure that all images are for the same MH_ type.
	var machoType Type

	// Following the fat_header comes narch fat_arch structs that index
	// Mach-O images further in the file.
	c := saferio.SliceCap[FatArch](uint64(narch))
	if c < 0 {
		return nil, &FormatError{offset, "too many images", nil}
	}
	ff.Arches = make([]FatArch, 0, c)
	for i := uint32(0); i < narch; i++ {
		var fa FatArch
		err = binary.Read(sr, binary.BigEndian, &fa.FatArchHeader)
		if err != nil {
			return nil, &FormatError{offset, "invalid fat_arch header", nil}
		}
		offset += fatArchHeaderSize

		fr := io.NewSectionReader(r, int64(fa.Offset), int64(fa.Size))
		fa.File, err = NewFile(fr)
		if err != nil {
			return nil, err
		}

		// Make sure the architecture for this image is not duplicate.
		seenArch := (uint64(fa.Cpu) << 32) | uint64(fa.SubCpu)
		if o, k := seenArches[seenArch]; o || k {
			return nil, &FormatError{offset, fmt.Sprintf("duplicate architecture cpu=%v, subcpu=%#x", fa.Cpu, fa.SubCpu), nil}
		}
		seenArches[seenArch] = true

		// Make sure the Mach-O type matches that of the first image.
		if i == 0 {
			machoType = fa.Type
		} else {
			if fa.Type != machoType {
				return nil, &FormatError{offset, fmt.Sprintf("Mach-O type for architecture #%d (type=%#x) does not match first (type=%#x)", i, fa.Type, machoType), nil}
			}
		}

		ff.Arches = append(ff.Arches, fa)
	}

	return &ff, nil
}

// OpenFat opens the named file using [os.Open] and prepares it for use as a Mach-O
// universal binary.
func OpenFat(name string) (*FatFile, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	ff, err := NewFatFile(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	ff.closer = f
	return ff, nil
}

func (ff *FatFile) Close() error {
	var err error
	if ff.closer != nil {
		err = ff.closer.Close()
		ff.closer = nil
	}
	return err
}
