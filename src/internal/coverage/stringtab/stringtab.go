// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stringtab

import (
	"fmt"
	"internal/coverage/uleb128"
	"io"
)

// This package implements a string table writer utility for use in
// emitting coverage meta-data and counter-data files.

type Writer struct {
	stab   map[string]uint32
	strs   []string
	tmp    []byte
	frozen bool
}

// InitWriter initializes a stringtab.Writer.
func (stw *Writer) InitWriter() {
	stw.stab = make(map[string]uint32)
	stw.tmp = make([]byte, 64)
}

// Nentries returns the number of strings interned so far.
func (stw *Writer) Nentries() uint32 {
	return uint32(len(stw.strs))
}

// Lookup looks up string 's' in the writer's table, adding
// a new entry if need be, and returning an index into the table.
func (stw *Writer) Lookup(s string) uint32 {
	if idx, ok := stw.stab[s]; ok {
		return idx
	}
	idx := uint32(len(stw.strs))
	stw.stab[s] = idx
	stw.strs = append(stw.strs, s)
	return idx
}

// Size computes the memory in bytes needed for the serialized
// version of a stringtab.Writer.
func (stw *Writer) Size() uint32 {
	rval := uint32(0)
	stw.tmp = stw.tmp[:0]
	stw.tmp = uleb128.AppendUleb128(stw.tmp, uint(len(stw.strs)))
	rval += uint32(len(stw.tmp))
	for _, s := range stw.strs {
		stw.tmp = stw.tmp[:0]
		slen := uint(len(s))
		stw.tmp = uleb128.AppendUleb128(stw.tmp, slen)
		rval += uint32(len(stw.tmp)) + uint32(slen)
	}
	return rval
}

// Write writes the string table in serialized form to the specified
// io.Writer.
func (stw *Writer) Write(w io.Writer) error {
	wr128 := func(v uint) error {
		stw.tmp = stw.tmp[:0]
		stw.tmp = uleb128.AppendUleb128(stw.tmp, v)
		if nw, err := w.Write(stw.tmp); err != nil {
			return fmt.Errorf("writing string table: %v", err)
		} else if nw != len(stw.tmp) {
			return fmt.Errorf("short write emitting stringtab uleb")
		}
		return nil
	}
	if err := wr128(uint(len(stw.strs))); err != nil {
		return err
	}
	for _, s := range stw.strs {
		if err := wr128(uint(len(s))); err != nil {
			return err
		}
		if nw, err := w.Write([]byte(s)); err != nil {
			return fmt.Errorf("writing string table: %v\n", err)
		} else if nw != len([]byte(s)) {
			return fmt.Errorf("short write emitting stringtab")
		}
	}
	return nil
}
