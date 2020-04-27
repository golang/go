// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/sym"
	"strings"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in dodata(), still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted.

func Addstring(s *sym.Symbol, str string) int64 {
	if s.Type == 0 {
		s.Type = sym.SNOPTRDATA
	}
	s.Attr |= sym.AttrReachable
	r := s.Size
	if s.Name == ".shstrtab" {
		elfsetstring(s, str, int(r))
	}
	s.P = append(s.P, str...)
	s.P = append(s.P, 0)
	s.Size = int64(len(s.P))
	return r
}

// symalign returns the required alignment for the given symbol s.
func symalign(s *sym.Symbol) int32 {
	min := int32(thearch.Minalign)
	if s.Align >= min {
		return s.Align
	} else if s.Align != 0 {
		return min
	}
	if strings.HasPrefix(s.Name, "go.string.") || strings.HasPrefix(s.Name, "type..namedata.") {
		// String data is just bytes.
		// If we align it, we waste a lot of space to padding.
		return min
	}
	align := int32(thearch.Maxalign)
	for int64(align) > s.Size && align > min {
		align >>= 1
	}
	s.Align = align
	return align
}
