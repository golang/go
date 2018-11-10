// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

var (
	errBadELF    = errors.New("malformed ELF binary")
	errNoBuildID = errors.New("no NT_GNU_BUILD_ID found in ELF binary")
)

// elfBuildID returns the GNU build ID of the named ELF binary,
// without introducing a dependency on debug/elf and its dependencies.
func elfBuildID(file string) (string, error) {
	buf := make([]byte, 256)
	f, err := os.Open(file)
	if err != nil {
		return "", err
	}
	defer f.Close()

	if _, err := f.ReadAt(buf[:64], 0); err != nil {
		return "", err
	}

	// ELF file begins with \x7F E L F.
	if buf[0] != 0x7F || buf[1] != 'E' || buf[2] != 'L' || buf[3] != 'F' {
		return "", errBadELF
	}

	var byteOrder binary.ByteOrder
	switch buf[5] {
	default:
		return "", errBadELF
	case 1: // little-endian
		byteOrder = binary.LittleEndian
	case 2: // big-endian
		byteOrder = binary.BigEndian
	}

	var shnum int
	var shoff, shentsize int64
	switch buf[4] {
	default:
		return "", errBadELF
	case 1: // 32-bit file header
		shoff = int64(byteOrder.Uint32(buf[32:]))
		shentsize = int64(byteOrder.Uint16(buf[46:]))
		if shentsize != 40 {
			return "", errBadELF
		}
		shnum = int(byteOrder.Uint16(buf[48:]))
	case 2: // 64-bit file header
		shoff = int64(byteOrder.Uint64(buf[40:]))
		shentsize = int64(byteOrder.Uint16(buf[58:]))
		if shentsize != 64 {
			return "", errBadELF
		}
		shnum = int(byteOrder.Uint16(buf[60:]))
	}

	for i := 0; i < shnum; i++ {
		if _, err := f.ReadAt(buf[:shentsize], shoff+int64(i)*shentsize); err != nil {
			return "", err
		}
		if typ := byteOrder.Uint32(buf[4:]); typ != 7 { // SHT_NOTE
			continue
		}
		var off, size int64
		if shentsize == 40 {
			// 32-bit section header
			off = int64(byteOrder.Uint32(buf[16:]))
			size = int64(byteOrder.Uint32(buf[20:]))
		} else {
			// 64-bit section header
			off = int64(byteOrder.Uint64(buf[24:]))
			size = int64(byteOrder.Uint64(buf[32:]))
		}
		size += off
		for off < size {
			if _, err := f.ReadAt(buf[:16], off); err != nil { // room for header + name GNU\x00
				return "", err
			}
			nameSize := int(byteOrder.Uint32(buf[0:]))
			descSize := int(byteOrder.Uint32(buf[4:]))
			noteType := int(byteOrder.Uint32(buf[8:]))
			descOff := off + int64(12+(nameSize+3)&^3)
			off = descOff + int64((descSize+3)&^3)
			if nameSize != 4 || noteType != 3 || buf[12] != 'G' || buf[13] != 'N' || buf[14] != 'U' || buf[15] != '\x00' { // want name GNU\x00 type 3 (NT_GNU_BUILD_ID)
				continue
			}
			if descSize > len(buf) {
				return "", errBadELF
			}
			if _, err := f.ReadAt(buf[:descSize], descOff); err != nil {
				return "", err
			}
			return fmt.Sprintf("%x", buf[:descSize]), nil
		}
	}
	return "", errNoBuildID
}
