// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"maps"
	"math"
	"slices"
	"strconv"
	"unsafe"
)

// This file holds the runtime half of the generated binary encoding
// (SPEC Sec 12.3, T-Sec12.1): the append helpers and the BinaryReader
// that the generated MarshalBinary, AppendBinary, and UnmarshalBinary
// call. The layout is version byte 0x01, then the fields in declaration
// order: integers as 8 bytes little-endian, float32 and float64 as their
// IEEE bits little-endian, bool as one byte, strings and byte slices as a
// uvarint length plus the bytes, slices and arrays as a uvarint count plus
// the elements, string-keyed maps as a uvarint count plus the entries
// sorted by key, structs inline, and an optional as a presence byte plus
// the payload when present.

// BinaryVersion is the first byte of every generated binary encoding.
const BinaryVersion = 0x01

// BinaryMaxPrealloc is the byte budget of the memory a count header may
// reserve before the elements arrive (SPEC Resolution 139).
const BinaryMaxPrealloc = 1 << 20

// BinaryErrorKind tells why a binary encode or decode failed.
type BinaryErrorKind uint8

// The kinds of BinaryError.
const (
	BinaryTruncated BinaryErrorKind = iota + 1
	BinaryBadPresence
	BinaryBadBool
	BinaryBadVersion
	BinaryTrailing
	BinaryOverflow
	BinaryBadLength
	BinaryNilPointer
	BinaryInvalid
)

// BinaryError reports a failed binary encode or decode. Type is the
// struct being coded, Field the path of the failing datum (`Items[].Name`
// for an element, `(version)` and `(end)` for the framing), and Offset
// the byte offset where that datum starts: in the input for a decode, in
// the output for an encode. Cause carries the error of a field's own
// MarshalBinary or UnmarshalBinary.
type BinaryError struct {
	Type   string
	Field  string
	Offset int
	Kind   BinaryErrorKind
	Detail string
	Cause  error
}

// Error spells the type, the field path, the offset, and the reason.
func (e *BinaryError) Error() string {
	name := e.Type
	if e.Field != "" {
		name += "." + e.Field
	}
	return "logos: binary " + name + " at offset " + strconv.Itoa(e.Offset) + ": " + e.Detail
}

// Unwrap returns the error of the field's own coder, if any.
func (e *BinaryError) Unwrap() error { return e.Cause }

// BinNil returns the error of an encode that met a nil pointer in a
// non-nil field. off is the length of the output so far.
func BinNil(typ, field string, off int) error {
	return &BinaryError{Type: typ, Field: field, Offset: off, Kind: BinaryNilPointer, Detail: "nil pointer in non-nil field"}
}

// BinInvalid returns the error of an encode whose field coder failed.
// off is the length of the output so far.
func BinInvalid(typ, field string, off int, cause error) error {
	return &BinaryError{Type: typ, Field: field, Offset: off, Kind: BinaryInvalid, Detail: cause.Error(), Cause: cause}
}

// BinAppendUvarint appends x as a uvarint.
func BinAppendUvarint(b []byte, x uint64) []byte {
	for x >= 0x80 {
		b = append(b, byte(x)|0x80)
		x >>= 7
	}
	return append(b, byte(x))
}

// BinAppendUint64 appends x as 8 bytes little-endian.
func BinAppendUint64(b []byte, x uint64) []byte {
	return append(b, byte(x), byte(x>>8), byte(x>>16), byte(x>>24), byte(x>>32), byte(x>>40), byte(x>>48), byte(x>>56))
}

// BinAppendUint32 appends x as 4 bytes little-endian.
func BinAppendUint32(b []byte, x uint32) []byte {
	return append(b, byte(x), byte(x>>8), byte(x>>16), byte(x>>24))
}

// BinAppendFloat32 appends the bits of f as 4 bytes little-endian.
func BinAppendFloat32(b []byte, f float32) []byte {
	return BinAppendUint32(b, math.Float32bits(f))
}

// BinAppendFloat64 appends the bits of f as 8 bytes little-endian.
func BinAppendFloat64(b []byte, f float64) []byte {
	return BinAppendUint64(b, math.Float64bits(f))
}

// BinAppendBool appends 0x01 for true and 0x00 for false.
func BinAppendBool(b []byte, v bool) []byte {
	if v {
		return append(b, 1)
	}
	return append(b, 0)
}

// BinAppendString appends the uvarint length of s, then its bytes.
func BinAppendString(b []byte, s string) []byte {
	b = BinAppendUvarint(b, uint64(len(s)))
	return append(b, s...)
}

// BinAppendBytes appends the uvarint length of p, then its bytes.
func BinAppendBytes(b []byte, p []byte) []byte {
	b = BinAppendUvarint(b, uint64(len(p)))
	return append(b, p...)
}

// BinSortedKeys returns the keys of m in ascending byte order.
func BinSortedKeys[M ~map[K]V, K ~string, V any](m M) []K {
	return slices.Sorted(maps.Keys(m))
}

// BinaryReader decodes one generated binary encoding. The first failure
// lands in Err; every later read returns the zero value and records
// nothing, so generated straight-line code checks Err once at its end and
// loops stop on it.
type BinaryReader struct {
	Type string
	Buf  []byte
	Off  int
	Err  error
}

// fail records the first error of the reader.
func (r *BinaryReader) fail(kind BinaryErrorKind, field string, off int, detail string) {
	if r.Err == nil {
		r.Err = &BinaryError{Type: r.Type, Field: field, Offset: off, Kind: kind, Detail: detail}
	}
}

// take returns the next n bytes, or fails with a truncation error naming
// field and the offset where the datum started.
func (r *BinaryReader) take(field string, n int) []byte {
	if r.Err != nil {
		return nil
	}
	if have := len(r.Buf) - r.Off; have < n {
		r.fail(BinaryTruncated, field, r.Off, "truncated: need "+strconv.Itoa(n)+" bytes, have "+strconv.Itoa(have))
		return nil
	}
	s := r.Buf[r.Off : r.Off+n]
	r.Off += n
	return s
}

// Version reads the leading version byte and rejects any other value.
func (r *BinaryReader) Version() {
	b := r.take("(version)", 1)
	if b != nil && b[0] != BinaryVersion {
		r.fail(BinaryBadVersion, "(version)", 0, "version "+hexByte(b[0])+", want "+hexByte(BinaryVersion))
	}
}

// Done returns the first error, or a trailing-bytes error when input
// remains after the last field.
func (r *BinaryReader) Done() error {
	if r.Err == nil && r.Off != len(r.Buf) {
		r.fail(BinaryTrailing, "(end)", r.Off, strconv.Itoa(len(r.Buf)-r.Off)+" trailing bytes")
	}
	return r.Err
}

// hexByte spells b as 0xNN.
func hexByte(b byte) string {
	const digits = "0123456789abcdef"
	return "0x" + string([]byte{digits[b>>4], digits[b&15]})
}

// Uint64 reads 8 bytes little-endian.
func (r *BinaryReader) Uint64(field string) uint64 {
	b := r.take(field, 8)
	if b == nil {
		return 0
	}
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// Float32 reads 4 bytes little-endian as float32 bits.
func (r *BinaryReader) Float32(field string) float32 {
	b := r.take(field, 4)
	if b == nil {
		return 0
	}
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
}

// Float64 reads 8 bytes little-endian as float64 bits.
func (r *BinaryReader) Float64(field string) float64 {
	return math.Float64frombits(r.Uint64(field))
}

// Bool reads one byte and rejects any value except 0x00 and 0x01.
func (r *BinaryReader) Bool(field string) bool {
	start := r.Off
	b := r.take(field, 1)
	if b == nil {
		return false
	}
	if b[0] > 1 {
		r.fail(BinaryBadBool, field, start, "bad bool byte "+hexByte(b[0]))
	}
	return b[0] == 1
}

// Presence reads an optional's presence byte: 0x00 absent, 0x01 present.
func (r *BinaryReader) Presence(field string) bool {
	start := r.Off
	b := r.take(field, 1)
	if b == nil {
		return false
	}
	if b[0] > 1 {
		r.fail(BinaryBadPresence, field, start, "bad presence byte "+hexByte(b[0]))
	}
	return b[0] == 1
}

// uvarint reads a uvarint that started at start.
func (r *BinaryReader) uvarint(field string) (uint64, bool) {
	start := r.Off
	var x uint64
	for shift := uint(0); r.Err == nil; shift += 7 {
		b := r.take(field, 1)
		if b == nil {
			r.Err.(*BinaryError).Offset = start
			return 0, false
		}
		if shift == 63 && b[0] > 1 {
			r.fail(BinaryOverflow, field, start, "varint overflows 64 bits")
			return 0, false
		}
		x |= uint64(b[0]&0x7f) << shift
		if b[0] < 0x80 {
			return x, true
		}
	}
	return 0, false
}

// Count reads a slice or map count header. Every element takes at least
// width bytes, so a count above the remaining input divided by width
// fails at once as a truncation, before any element is read or any
// memory is reserved (SPEC Resolution 139).
func (r *BinaryReader) Count(field string, width int) int {
	start := r.Off
	n, ok := r.uvarint(field)
	if !ok {
		return 0
	}
	rest := uint64(len(r.Buf) - r.Off)
	if width > 0 && n > rest/uint64(width) {
		r.fail(BinaryTruncated, field, start, "truncated: count "+strconv.FormatUint(n, 10)+" of "+strconv.Itoa(width)+"-byte elements, "+strconv.FormatUint(rest, 10)+" bytes left")
		return 0
	}
	return int(n)
}

// ArrayLen reads an array count header and requires it to equal want.
func (r *BinaryReader) ArrayLen(field string, want int) {
	start := r.Off
	n, ok := r.uvarint(field)
	if ok && n != uint64(want) {
		r.fail(BinaryBadLength, field, start, "array length "+strconv.FormatUint(n, 10)+", want "+strconv.Itoa(want))
	}
}

// bytesField reads a length-prefixed byte run without copying it.
func (r *BinaryReader) bytesField(field string) []byte {
	start := r.Off
	n, ok := r.uvarint(field)
	if !ok {
		return nil
	}
	rest := uint64(len(r.Buf) - r.Off)
	if n > rest {
		r.fail(BinaryTruncated, field, start, "truncated: need "+strconv.FormatUint(n, 10)+" bytes, have "+strconv.FormatUint(rest, 10))
		return nil
	}
	return r.take(field, int(n))
}

// String reads a length-prefixed string and copies it out of the input.
func (r *BinaryReader) String(field string) string {
	return string(r.bytesField(field))
}

// Bytes reads a length-prefixed byte slice and copies it out of the
// input; the result is never nil after a successful read.
func (r *BinaryReader) Bytes(field string) []byte {
	raw := r.bytesField(field)
	out := make([]byte, len(raw))
	copy(out, raw)
	return out
}

// Unmarshal reads a length-prefixed run and hands it to the field's own
// UnmarshalBinary, which must copy what it keeps.
func (r *BinaryReader) Unmarshal(field string, u interface{ UnmarshalBinary([]byte) error }) {
	start := r.Off
	raw := r.bytesField(field)
	if r.Err != nil {
		return
	}
	if err := u.UnmarshalBinary(raw); err != nil {
		r.Err = &BinaryError{Type: r.Type, Field: field, Offset: start, Kind: BinaryInvalid, Detail: err.Error(), Cause: err}
	}
}

// BinSigned reads a signed integer of type T from its 8-byte form and
// rejects a value T cannot hold.
func BinSigned[T int | int8 | int16 | int32 | int64](r *BinaryReader, field string) T {
	start := r.Off
	u := int64(r.Uint64(field))
	x := T(u)
	if r.Err == nil && int64(x) != u {
		r.fail(BinaryOverflow, field, start, "value "+strconv.FormatInt(u, 10)+" does not fit in "+strconv.Itoa(int(unsafe.Sizeof(x))*8)+" bits")
	}
	return x
}

// BinUnsigned reads an unsigned integer of type T from its 8-byte form
// and rejects a value T cannot hold.
func BinUnsigned[T uint | uint8 | uint16 | uint32 | uint64 | uintptr](r *BinaryReader, field string) T {
	start := r.Off
	u := r.Uint64(field)
	x := T(u)
	if r.Err == nil && uint64(x) != u {
		r.fail(BinaryOverflow, field, start, "value "+strconv.FormatUint(u, 10)+" does not fit in "+strconv.Itoa(int(unsafe.Sizeof(x))*8)+" bits")
	}
	return x
}

// binCap bounds the elements a count header may pre-allocate: the count
// itself, half the remaining input, and the byte budget over the element
// size (SPEC Resolution 139).
func binCap(r *BinaryReader, n int, size uintptr) int {
	c := min(n, (len(r.Buf)-r.Off)/2)
	if size > 0 {
		c = min(c, BinaryMaxPrealloc/int(size))
	}
	return max(c, 0)
}

// BinSlice returns an empty slice with the pre-allocation the count
// header n may claim; the caller appends the decoded elements.
func BinSlice[E any](r *BinaryReader, n int) []E {
	return make([]E, 0, binCap(r, n, unsafe.Sizeof(*new(E))))
}

// BinMap returns an empty map with the pre-allocation the count header n
// may claim; the caller stores the decoded entries.
func BinMap[M ~map[K]V, K comparable, V any](r *BinaryReader, n int) M {
	return make(M, binCap(r, n, unsafe.Sizeof(*new(K))+unsafe.Sizeof(*new(V))))
}
