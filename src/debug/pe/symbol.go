// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"encoding/binary"
	"errors"
	"fmt"
	"internal/saferio"
	"io"
	"unsafe"
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

// readCOFFSymbols reads in the symbol table for a PE file, returning
// a slice of COFFSymbol objects. The PE format includes both primary
// symbols (whose fields are described by COFFSymbol above) and
// auxiliary symbols; all symbols are 18 bytes in size. The auxiliary
// symbols for a given primary symbol are placed following it in the
// array, e.g.
//
//	...
//	k+0:  regular sym k
//	k+1:    1st aux symbol for k
//	k+2:    2nd aux symbol for k
//	k+3:  regular sym k+3
//	k+4:    1st aux symbol for k+3
//	k+5:  regular sym k+5
//	k+6:  regular sym k+6
//
// The PE format allows for several possible aux symbol formats. For
// more info see:
//
//	https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#auxiliary-symbol-records
//
// At the moment this package only provides APIs for looking at
// aux symbols of format 5 (associated with section definition symbols).
func readCOFFSymbols(fh *FileHeader, r io.ReadSeeker) ([]COFFSymbol, error) {
	if fh.PointerToSymbolTable == 0 {
		return nil, nil
	}
	if fh.NumberOfSymbols <= 0 {
		return nil, nil
	}
	_, err := r.Seek(int64(fh.PointerToSymbolTable), io.SeekStart)
	if err != nil {
		return nil, fmt.Errorf("fail to seek to symbol table: %v", err)
	}
	c := saferio.SliceCap[COFFSymbol](uint64(fh.NumberOfSymbols))
	if c < 0 {
		return nil, errors.New("too many symbols; file may be corrupt")
	}
	syms := make([]COFFSymbol, 0, c)
	naux := 0
	for k := uint32(0); k < fh.NumberOfSymbols; k++ {
		var sym COFFSymbol
		if naux == 0 {
			// Read a primary symbol.
			err = binary.Read(r, binary.LittleEndian, &sym)
			if err != nil {
				return nil, fmt.Errorf("fail to read symbol table: %v", err)
			}
			// Record how many auxiliary symbols it has.
			naux = int(sym.NumberOfAuxSymbols)
		} else {
			// Read an aux symbol. At the moment we assume all
			// aux symbols are format 5 (obviously this doesn't always
			// hold; more cases will be needed below if more aux formats
			// are supported in the future).
			naux--
			aux := (*COFFSymbolAuxFormat5)(unsafe.Pointer(&sym))
			err = binary.Read(r, binary.LittleEndian, aux)
			if err != nil {
				return nil, fmt.Errorf("fail to read symbol table: %v", err)
			}
		}
		syms = append(syms, sym)
	}
	if naux != 0 {
		return nil, fmt.Errorf("fail to read symbol table: %d aux symbols unread", naux)
	}
	return syms, nil
}

// isSymNameOffset checks symbol name if it is encoded as offset into string table.
func isSymNameOffset(name [8]byte) (bool, uint32) {
	if name[0] == 0 && name[1] == 0 && name[2] == 0 && name[3] == 0 {
		offset := binary.LittleEndian.Uint32(name[4:])
		if offset == 0 {
			// symbol has no name
			return false, 0
		}
		return true, offset
	}
	return false, 0
}

// FullName finds real name of symbol sym. Normally name is stored
// in sym.Name, but if it is longer then 8 characters, it is stored
// in COFF string table st instead.
func (sym *COFFSymbol) FullName(st StringTable) (string, error) {
	if ok, offset := isSymNameOffset(sym.Name); ok {
		return st.String(offset)
	}
	return cstring(sym.Name[:]), nil
}

func removeAuxSymbols(allsyms []COFFSymbol, st StringTable) ([]*Symbol, error) {
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
		name, err := sym.FullName(st)
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

// Symbol is similar to [COFFSymbol] with Name field replaced
// by Go string. Symbol also does not have NumberOfAuxSymbols.
type Symbol struct {
	Name          string
	Value         uint32
	SectionNumber int16
	Type          uint16
	StorageClass  uint8
}

// COFFSymbolAuxFormat5 describes the expected form of an aux symbol
// attached to a section definition symbol. The PE format defines a
// number of different aux symbol formats: format 1 for function
// definitions, format 2 for .be and .ef symbols, and so on. Format 5
// holds extra info associated with a section definition, including
// number of relocations + line numbers, as well as COMDAT info. See
// https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#auxiliary-format-5-section-definitions
// for more on what's going on here.
type COFFSymbolAuxFormat5 struct {
	Size           uint32
	NumRelocs      uint16
	NumLineNumbers uint16
	Checksum       uint32
	SecNum         uint16
	Selection      uint8
	_              [3]uint8 // padding
}

// These constants make up the possible values for the 'Selection'
// field in an AuxFormat5.
const (
	IMAGE_COMDAT_SELECT_NODUPLICATES = 1
	IMAGE_COMDAT_SELECT_ANY          = 2
	IMAGE_COMDAT_SELECT_SAME_SIZE    = 3
	IMAGE_COMDAT_SELECT_EXACT_MATCH  = 4
	IMAGE_COMDAT_SELECT_ASSOCIATIVE  = 5
	IMAGE_COMDAT_SELECT_LARGEST      = 6
)

// COFFSymbolReadSectionDefAux returns a blob of auxiliary information
// (including COMDAT info) for a section definition symbol. Here 'idx'
// is the index of a section symbol in the main [COFFSymbol] array for
// the File. Return value is a pointer to the appropriate aux symbol
// struct. For more info, see:
//
// auxiliary symbols: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#auxiliary-symbol-records
// COMDAT sections: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#comdat-sections-object-only
// auxiliary info for section definitions: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#auxiliary-format-5-section-definitions
func (f *File) COFFSymbolReadSectionDefAux(idx int) (*COFFSymbolAuxFormat5, error) {
	var rv *COFFSymbolAuxFormat5
	if idx < 0 || idx >= len(f.COFFSymbols) {
		return rv, fmt.Errorf("invalid symbol index")
	}
	pesym := &f.COFFSymbols[idx]
	const IMAGE_SYM_CLASS_STATIC = 3
	if pesym.StorageClass != uint8(IMAGE_SYM_CLASS_STATIC) {
		return rv, fmt.Errorf("incorrect symbol storage class")
	}
	if pesym.NumberOfAuxSymbols == 0 || idx+1 >= len(f.COFFSymbols) {
		return rv, fmt.Errorf("aux symbol unavailable")
	}
	// Locate and return a pointer to the successor aux symbol.
	pesymn := &f.COFFSymbols[idx+1]
	rv = (*COFFSymbolAuxFormat5)(unsafe.Pointer(pesymn))
	return rv, nil
}
