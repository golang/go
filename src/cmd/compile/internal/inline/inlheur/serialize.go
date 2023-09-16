// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import "strings"

func (fp *FuncProps) SerializeToString() string {
	if fp == nil {
		return ""
	}
	var sb strings.Builder
	writeUleb128(&sb, uint64(fp.Flags))
	writeUleb128(&sb, uint64(len(fp.ParamFlags)))
	for _, pf := range fp.ParamFlags {
		writeUleb128(&sb, uint64(pf))
	}
	writeUleb128(&sb, uint64(len(fp.ResultFlags)))
	for _, rf := range fp.ResultFlags {
		writeUleb128(&sb, uint64(rf))
	}
	return sb.String()
}

func DeserializeFromString(s string) *FuncProps {
	if len(s) == 0 {
		return nil
	}
	var fp FuncProps
	var v uint64
	sl := []byte(s)
	v, sl = readULEB128(sl)
	fp.Flags = FuncPropBits(v)
	v, sl = readULEB128(sl)
	fp.ParamFlags = make([]ParamPropBits, v)
	for i := range fp.ParamFlags {
		v, sl = readULEB128(sl)
		fp.ParamFlags[i] = ParamPropBits(v)
	}
	v, sl = readULEB128(sl)
	fp.ResultFlags = make([]ResultPropBits, v)
	for i := range fp.ResultFlags {
		v, sl = readULEB128(sl)
		fp.ResultFlags[i] = ResultPropBits(v)
	}
	return &fp
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
