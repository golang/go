// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"encoding/binary"
	"fmt"
	"io"
)

const COFFSymbolSize = 18

// COFFSymbol represents single COFF symbol table record.
type COFFSymbol struct {
	Name               [8]uint8
	Value              uint32
	SectionNumber      int16
	Type               uint16
	StorageClass       uint8
	NumberOfAuxSymbols uint8
}

func readCOFFSymbols(fh *FileHeader, r io.ReadSeeker) ([]COFFSymbol, error) {
	if fh.NumberOfSymbols <= 0 {
		return nil, nil
	}
	_, err := r.Seek(int64(fh.PointerToSymbolTable), io.SeekStart)
	if err != nil {
		return nil, fmt.Errorf("fail to seek to symbol table: %v", err)
	}
	syms := make([]COFFSymbol, fh.NumberOfSymbols)
	err = binary.Read(r, binary.LittleEndian, syms)
	if err != nil {
		return nil, fmt.Errorf("fail to read symbol table: %v", err)
	}
	return syms, nil
}

// isSymNameOffset checks symbol name if it is encoded as offset into string table.
func isSymNameOffset(name [8]byte) (bool, uint32) {
	if name[0] == 0 && name[1] == 0 && name[2] == 0 && name[3] == 0 {
		return true, binary.LittleEndian.Uint32(name[4:])
	}
	return false, 0
}

// _FullName finds real name of symbol sym. Normally name is stored
// in sym.Name, but if it is longer then 8 characters, it is stored
// in COFF string table st instead.
func (sym *COFFSymbol) _FullName(st _StringTable) (string, error) {
	if ok, offset := isSymNameOffset(sym.Name); ok {
		return st.String(offset)
	}
	return cstring(sym.Name[:]), nil
}

func removeAuxSymbols(allsyms []COFFSymbol, st _StringTable) ([]*Symbol, error) {
	if len(allsyms) == 0 {
		return nil, nil
	}
	syms := make([]*Symbol, 0)
	aux := uint8(0)
	for _, sym := range allsyms {
		if aux > 0 {
			aux--
			continue
		}
		name, err := sym._FullName(st)
		if err != nil {
			return nil, err
		}
		aux = sym.NumberOfAuxSymbols
		s := &Symbol{
			Name:          name,
			Value:         sym.Value,
			SectionNumber: sym.SectionNumber,
			Type:          sym.Type,
			StorageClass:  sym.StorageClass,
		}
		syms = append(syms, s)
	}
	return syms, nil
}

// Symbol is similar to COFFSymbol with Name field replaced
// by Go string. Symbol also does not have NumberOfAuxSymbols.
type Symbol struct {
	Name          string
	Value         uint32
	SectionNumber int16
	Type          uint16
	StorageClass  uint8
}
