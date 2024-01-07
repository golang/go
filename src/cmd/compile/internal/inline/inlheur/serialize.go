// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import "strings"

func (funcProps *FuncProps) SerializeToString() string {
	if funcProps == nil {
		return ""
	}
	var sb strings.Builder
	writeUleb128(&sb, uint64(funcProps.Flags))
	writeUleb128(&sb, uint64(len(funcProps.ParamFlags)))
	for _, pf := range funcProps.ParamFlags {
		writeUleb128(&sb, uint64(pf))
	}
	writeUleb128(&sb, uint64(len(funcProps.ResultFlags)))
	for _, rf := range funcProps.ResultFlags {
		writeUleb128(&sb, uint64(rf))
	}
	return sb.String()
}

func DeserializeFromString(s string) *FuncProps {
	if len(s) == 0 {
		return nil
	}
	var funcProps FuncProps
	var v uint64
	sl := []byte(s)
	v, sl = readULEB128(sl)
	funcProps.Flags = FuncPropBits(v)
	v, sl = readULEB128(sl)
	funcProps.ParamFlags = make([]ParamPropBits, v)
	for i := range funcProps.ParamFlags {
		v, sl = readULEB128(sl)
		funcProps.ParamFlags[i] = ParamPropBits(v)
	}
	v, sl = readULEB128(sl)
	funcProps.ResultFlags = make([]ResultPropBits, v)
	for i := range funcProps.ResultFlags {
		v, sl = readULEB128(sl)
		funcProps.ResultFlags[i] = ResultPropBits(v)
	}
	return &funcProps
}

func readULEB128(sl []byte) (value uint64, rsl []byte) {
	var shift uint

	for {
		b := sl[0]
		sl = sl[1:]
		value |= (uint64(b&0x7F) << shift)
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	return value, sl
}

func writeUleb128(sb *strings.Builder, v uint64) {
	if v < 128 {
		sb.WriteByte(uint8(v))
		return
	}
	more := true
	for more {
		c := uint8(v & 0x7f)
		v >>= 7
		more = v != 0
		if more {
			c |= 0x80
		}
		sb.WriteByte(c)
	}
}
