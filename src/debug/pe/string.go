// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"internal/saferio"
	"io"
)

// cstring converts ASCII byte sequence b to string.
// It stops once it finds 0 or reaches end of b.
func cstring(b []byte) string {
	i := bytes.IndexByte(b, 0)
	if i == -1 {
		i = len(b)
	}
	return string(b[:i])
}

// StringTable is a COFF string table.
type StringTable []byte

func readStringTable(fh *FileHeader, r io.ReadSeeker) (StringTable, error) {
	// COFF string table is located right after COFF symbol table.
	if fh.PointerToSymbolTable <= 0 {
		return nil, nil
	}
	offset := fh.PointerToSymbolTable + COFFSymbolSize*fh.NumberOfSymbols
	_, err := r.Seek(int64(offset), io.SeekStart)
	if err != nil {
		return nil, fmt.Errorf("fail to seek to string table: %v", err)
	}
	var l uint32
	err = binary.Read(r, binary.LittleEndian, &l)
	if err != nil {
		return nil, fmt.Errorf("fail to read string table length: %v", err)
	}
	// string table length includes itself
	if l <= 4 {
		return nil, nil
	}
	l -= 4

	buf, err := saferio.ReadData(r, uint64(l))
	if err != nil {
		return nil, fmt.Errorf("fail to read string table: %v", err)
	}
	return StringTable(buf), nil
}

// TODO(brainman): decide if start parameter should be int instead of uint32

// String extracts string from COFF string table st at offset start.
func (st StringTable) String(start uint32) (string, error) {
	// start includes 4 bytes of string table length
	if start < 4 {
		return "", fmt.Errorf("offset %d is before the start of string table", start)
	}
	start -= 4
	if int(start) > len(st) {
		return "", fmt.Errorf("offset %d is beyond the end of string table", start)
	}
	return cstring(st[start:]), nil
}
