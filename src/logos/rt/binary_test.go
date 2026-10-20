// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"bytes"
	"errors"
	"math"
	"testing"
)

func newBinaryReader(data ...byte) *BinaryReader {
	return &BinaryReader{Type: "T", Buf: data}
}

func binaryErrorOf(t *testing.T, err error) *BinaryError {
	t.Helper()
	be, ok := errors.AsType[*BinaryError](err)
	if !ok {
		t.Fatalf("error %v is not a *BinaryError", err)
	}
	return be
}

func TestBinUvarintRoundTrip(t *testing.T) {
	for _, x := range []uint64{0, 1, 127, 128, 255, 16383, 16384, 1 << 32, 1<<63 - 1, 1 << 63, math.MaxUint64} {
		r := newBinaryReader(BinAppendUvarint(nil, x)...)
		got, ok := r.uvarint("x")
		if !ok || got != x || r.Off != len(r.Buf) {
			t.Errorf("uvarint %d decoded as %d (ok %v, off %d of %d)", x, got, ok, r.Off, len(r.Buf))
		}
	}
}

func TestBinUvarintMalformed(t *testing.T) {
	over := bytes.Repeat([]byte{0xff}, 10)
	over = append(over, 0x01)
	r := newBinaryReader(over...)
	if _, ok := r.uvarint("x"); ok {
		t.Fatal("11-byte varint accepted")
	}
	if be := binaryErrorOf(t, r.Err); be.Kind != BinaryOverflow || be.Offset != 0 {
		t.Errorf("overflow error: %+v", be)
	}
	r = newBinaryReader(0x80, 0x80)
	if _, ok := r.uvarint("x"); ok {
		t.Fatal("unterminated varint accepted")
	}
	if be := binaryErrorOf(t, r.Err); be.Kind != BinaryTruncated || be.Offset != 0 {
		t.Errorf("truncated error: %+v", be)
	}
}

func TestBinScalarRoundTrip(t *testing.T) {
	var b []byte
	b = BinAppendUint64(b, 0x0102030405060708)
	b = BinAppendFloat32(b, -1.5)
	b = BinAppendFloat64(b, math.Pi)
	b = BinAppendBool(b, true)
	b = BinAppendString(b, "héllo")
	b = BinAppendBytes(b, []byte{1, 2, 3})
	r := newBinaryReader(b...)
	if got := r.Uint64("a"); got != 0x0102030405060708 {
		t.Errorf("Uint64 = %#x", got)
	}
	if got := r.Float32("b"); got != -1.5 {
		t.Errorf("Float32 = %v", got)
	}
	if got := r.Float64("c"); got != math.Pi {
		t.Errorf("Float64 = %v", got)
	}
	if !r.Bool("d") {
		t.Error("Bool = false")
	}
	if got := r.String("e"); got != "héllo" {
		t.Errorf("String = %q", got)
	}
	if got := r.Bytes("f"); !bytes.Equal(got, []byte{1, 2, 3}) {
		t.Errorf("Bytes = %v", got)
	}
	if err := r.Done(); err != nil {
		t.Errorf("Done: %v", err)
	}
}

func TestBinReaderStickyFirstError(t *testing.T) {
	r := newBinaryReader(1, 2)
	r.Uint64("first")
	r.Uint64("second")
	be := binaryErrorOf(t, r.Err)
	if be.Field != "first" || be.Offset != 0 || be.Kind != BinaryTruncated {
		t.Errorf("first error lost: %+v", be)
	}
	if got := r.String("later"); got != "" || r.Err != error(be) {
		t.Errorf("read after failure: %q, err %v", got, r.Err)
	}
}

func TestBinReaderFraming(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		kind BinaryErrorKind
		off  int
	}{
		{"empty", nil, BinaryTruncated, 0},
		{"wrong version", []byte{0x02}, BinaryBadVersion, 0},
		{"trailing", []byte{BinaryVersion, 0xaa, 0xbb}, BinaryTrailing, 1},
	}
	for _, c := range cases {
		r := &BinaryReader{Type: "T", Buf: c.data}
		r.Version()
		be := binaryErrorOf(t, r.Done())
		if be.Kind != c.kind || be.Offset != c.off || be.Type != "T" || be.Field == "" {
			t.Errorf("%s: %+v", c.name, be)
		}
	}
}

func TestBinBoolAndPresenceRejectOtherBytes(t *testing.T) {
	r := newBinaryReader(2)
	r.Bool("flag")
	if be := binaryErrorOf(t, r.Err); be.Kind != BinaryBadBool || be.Offset != 0 || be.Field != "flag" {
		t.Errorf("bool: %+v", be)
	}
	r = newBinaryReader(0, 3)
	if r.Presence("a") {
		t.Error("presence 0 reported present")
	}
	r.Presence("b")
	if be := binaryErrorOf(t, r.Err); be.Kind != BinaryBadPresence || be.Offset != 1 || be.Field != "b" {
		t.Errorf("presence: %+v", be)
	}
}

func TestBinSignedUnsignedOverflow(t *testing.T) {
	r := newBinaryReader(BinAppendUint64(nil, 300)...)
	if got := BinUnsigned[uint8](r, "u"); got != 44 || binaryErrorOf(t, r.Err).Kind != BinaryOverflow {
		t.Errorf("uint8 of 300: %d, %v", got, r.Err)
	}
	r = newBinaryReader(BinAppendUint64(nil, uint64(math.MaxUint64))...)
	if got := BinSigned[int8](r, "s"); got != -1 || r.Err != nil {
		t.Errorf("int8 of -1: %d, %v", got, r.Err)
	}
	r = newBinaryReader(BinAppendUint64(nil, uint64(math.MaxInt64))...)
	BinSigned[int32](r, "s")
	if be := binaryErrorOf(t, r.Err); be.Kind != BinaryOverflow || be.Field != "s" {
		t.Errorf("int32 of MaxInt64: %+v", be)
	}
}

func TestBinCountBoundsByRemainingInput(t *testing.T) {
	data := append(BinAppendUvarint(nil, 1<<40), make([]byte, 10)...)
	r := newBinaryReader(data...)
	if n := r.Count("Items", 1); n != 0 {
		t.Errorf("count 2^40 over 10 bytes returned %d", n)
	}
	if be := binaryErrorOf(t, r.Err); be.Kind != BinaryTruncated || be.Offset != 0 || be.Field != "Items" {
		t.Errorf("count error: %+v", be)
	}
	for _, c := range []struct {
		count uint64
		width int
		ok    bool
	}{{10, 1, true}, {11, 1, false}, {5, 2, true}, {6, 2, false}, {1, 10, true}, {1, 11, false}} {
		r = newBinaryReader(append(BinAppendUvarint(nil, c.count), make([]byte, 10)...)...)
		got := r.Count("c", c.width)
		if (r.Err == nil) != c.ok || (c.ok && got != int(c.count)) {
			t.Errorf("count %d width %d: got %d, err %v", c.count, c.width, got, r.Err)
		}
	}
}

func TestBinSliceCapIsBounded(t *testing.T) {
	r := newBinaryReader(make([]byte, 20)...)
	if got := cap(BinSlice[uint64](r, 1000)); got != 10 {
		t.Errorf("cap with 20 bytes left = %d, want len(rest)/2 = 10", got)
	}
	if got := cap(BinSlice[uint64](r, 4)); got != 4 {
		t.Errorf("cap for count 4 = %d", got)
	}
	big := newBinaryReader(make([]byte, 4<<20)...)
	if got := cap(BinSlice[uint64](big, 3<<20)); got != BinaryMaxPrealloc/8 {
		t.Errorf("cap under the 1 MB budget = %d, want %d", got, BinaryMaxPrealloc/8)
	}
	if got := cap(BinSlice[struct{}](big, 5)); got != 5 {
		t.Errorf("zero-size element cap = %d", got)
	}
	if got := BinSlice[byte](big, 0); got == nil || len(got) != 0 {
		t.Errorf("empty slice = %#v, want non-nil", got)
	}
}

func TestBinMapIsNonNilAndBounded(t *testing.T) {
	r := newBinaryReader(make([]byte, 8)...)
	m := BinMap[map[string]int64](r, 1<<40)
	if m == nil {
		t.Fatal("BinMap returned nil")
	}
}

func TestBinSortedKeys(t *testing.T) {
	type key string
	got := BinSortedKeys(map[key]int{"b": 1, "a": 2, "": 3, "ab": 4})
	want := []key{"", "a", "ab", "b"}
	if len(got) != len(want) {
		t.Fatalf("keys = %q", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("keys = %q, want %q", got, want)
		}
	}
}

type failingCoder struct{ err error }

func (f *failingCoder) UnmarshalBinary([]byte) error { return f.err }

func TestBinUnmarshalWrapsFieldError(t *testing.T) {
	cause := errors.New("bad time")
	r := newBinaryReader(BinAppendBytes(nil, []byte{1, 2})...)
	r.Unmarshal("At", &failingCoder{cause})
	be := binaryErrorOf(t, r.Err)
	if be.Kind != BinaryInvalid || be.Field != "At" || be.Offset != 0 || !errors.Is(r.Err, cause) {
		t.Errorf("wrapped error: %+v", be)
	}
}

func TestBinaryErrorMessageNamesFieldAndOffset(t *testing.T) {
	e := BinNil("User", "Profile.Name", 17)
	const want = "logos: binary User.Profile.Name at offset 17: nil pointer in non-nil field"
	if e.Error() != want {
		t.Errorf("message = %q, want %q", e.Error(), want)
	}
}
