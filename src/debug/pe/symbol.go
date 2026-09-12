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
const BigObjSymbolSize = 20

// COFFSymbol represents single COFF symbol table record.
// NOTE: This is actually the format of a bigobj COFF symbol.
// The only difference between a bigobj symbol and regular symbol
// is that the SectionNumber is 32-bits.
type COFFSymbol struct {
	Name               [8]uint8
	Value              uint32
	SectionNumber      int32 // bigobj format (this field is 32 bits, rather than 16)
	Type               uint16
	StorageClass       uint8
	NumberOfAuxSymbols uint8
}

// rawCOFFSymbol represents a COFF symbol as stored in a regular COFF file
type rawCOFFSymbol struct {
	Name               [8]uint8
	Value              uint32
	SectionNumber      int16 // 16-bit in regular COFF
	Type               uint16
	StorageClass       uint8
	NumberOfAuxSymbols uint8
}

// readBigObjSymbol reads a single symbol from bigobj COFF format (20 bytes)
func readBigObjSymbol(r io.ReadSeeker) (COFFSymbol, error) {
	var sym COFFSymbol
	err := binary.Read(r, binary.LittleEndian, &sym)
	return sym, err
}

// readRegularSymbol reads a single symbol from regular COFF format (18 bytes) and converts to COFFSymbol
func readRegularSymbol(r io.ReadSeeker) (COFFSymbol, error) {
	var rawSym rawCOFFSymbol
	err := binary.Read(r, binary.LittleEndian, &rawSym)
	if err != nil {
		return COFFSymbol{}, err
	}

	sym := COFFSymbol{
		Name:               rawSym.Name,
		Value:              rawSym.Value,
		SectionNumber:      int32(rawSym.SectionNumber), // Extend to 32 bits
		Type:               rawSym.Type,
		StorageClass:       rawSym.StorageClass,
		NumberOfAuxSymbols: rawSym.NumberOfAuxSymbols,
	}
	return sym, nil
}

// readCOFFSymbols reads in the symbol table for a PE file, returning
// a slice of COFFSymbol objects containing both primary and auxiliary symbols.
// In a regular COFF file, each symbol is 18 bytes, with a 16-bit SectionNumber field.
// In a bigobj COFF file, each symbol is 20 bytes, with a 32-bit SectionNumber field.
//
// The COFF symbol table contains both primary symbols and auxiliary symbols.
// Auxiliary symbols immediately follow their associated primary symbol in both
// the binary data and the returned slice.
// In the binary format, symbols are arranged like this:
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
func readCOFFSymbols(f *File, r io.ReadSeeker) ([]COFFSymbol, error) {
	if f.GetPointerToSymbolTable() == 0 {
		return nil, nil
	}
	if f.GetNumberOfSymbols() <= 0 {
		return nil, nil
	}
	_, err := r.Seek(int64(f.GetPointerToSymbolTable()), io.SeekStart)
	if err != nil {
		return nil, fmt.Errorf("fail to seek to symbol table: %v", err)
	}
	c := saferio.SliceCap[COFFSymbol](uint64(f.GetNumberOfSymbols()))
	if c < 0 {
		return nil, errors.New("too many symbols; file may be corrupt")
	}
	syms := make([]COFFSymbol, 0, c)
	naux := 0

	isBigObj := f.IsBigObj()

	for k := uint32(0); k < f.GetNumberOfSymbols(); k++ {
		var sym COFFSymbol

		if isBigObj {
			sym, err = readBigObjSymbol(r)
		} else {
			sym, err = readRegularSymbol(r)
		}
		if err != nil {
			return nil, fmt.Errorf("fail to read symbol table: %v", err)
		}

		if naux == 0 {
			// This is a primary symbol
			// Record how many auxilliary symbols it has
			naux = int(sym.NumberOfAuxSymbols)
		} else {
			// This is an auxilliary symbol
			naux--
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
	SectionNumber int32
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
