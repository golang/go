// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"go/constant"
	"io"
	"math/big"
	"runtime"
	"strings"
)

// A PkgEncoder provides methods for encoding a package's Unified IR
// export data.
type PkgEncoder struct {
	// version of the bitstream.
	version Version

	// elems holds the bitstream for previously encoded elements.
	elems [numRelocs][]string

	// stringsIdx maps previously encoded strings to their index within
	// the RelocString section, to allow deduplication. That is,
	// elems[RelocString][stringsIdx[s]] == s (if present).
	stringsIdx map[string]RelIndex

	// syncFrames is the number of frames to write at each sync
	// marker. A negative value means sync markers are omitted.
	syncFrames int
}

// SyncMarkers reports whether pw uses sync markers.
func (pw *PkgEncoder) SyncMarkers() bool { return pw.syncFrames >= 0 }

// NewPkgEncoder returns an initialized PkgEncoder.
//
// syncFrames is the number of caller frames that should be serialized
// at Sync points. Serializing additional frames results in larger
// export data files, but can help diagnosing desync errors in
// higher-level Unified IR reader/writer code. If syncFrames is
// negative, then sync markers are omitted entirely.
func NewPkgEncoder(version Version, syncFrames int) PkgEncoder {
	return PkgEncoder{
		version:    version,
		stringsIdx: make(map[string]RelIndex),
		syncFrames: syncFrames,
	}
}

// DumpTo writes the package's encoded data to out0 and returns the
// package fingerprint.
func (pw *PkgEncoder) DumpTo(out0 io.Writer) (fingerprint [8]byte) {
	h := sha256.New()
	out := io.MultiWriter(out0, h)

	writeUint32 := func(x uint32) {
		assert(binary.Write(out, binary.LittleEndian, x) == nil)
	}

	writeUint32(uint32(pw.version))

	if pw.version.Has(Flags) {
		var flags uint32
		if pw.SyncMarkers() {
			flags |= flagSyncMarkers
		}
		writeUint32(flags)
	}

	// Write elemEndsEnds.
	var sum uint32
	for _, elems := range &pw.elems {
		sum += uint32(len(elems))
		writeUint32(sum)
	}

	// Write elemEnds.
	sum = 0
	for _, elems := range &pw.elems {
		for _, elem := range elems {
			sum += uint32(len(elem))
			writeUint32(sum)
		}
	}

	// Write elemData.
	for _, elems := range &pw.elems {
		for _, elem := range elems {
			_, err := io.WriteString(out, elem)
			assert(err == nil)
		}
	}

	// Write fingerprint.
	copy(fingerprint[:], h.Sum(nil))
	_, err := out0.Write(fingerprint[:])
	assert(err == nil)

	return
}

// StringIdx adds a string value to the strings section, if not
// already present, and returns its index.
func (pw *PkgEncoder) StringIdx(s string) RelIndex {
	if idx, ok := pw.stringsIdx[s]; ok {
		assert(pw.elems[SectionString][idx] == s)
		return idx
	}

	idx := RelIndex(len(pw.elems[SectionString]))
	pw.elems[SectionString] = append(pw.elems[SectionString], s)
	pw.stringsIdx[s] = idx
	return idx
}

// NewEncoder returns an Encoder for a new element within the given
// section, and encodes the given SyncMarker as the start of the
// element bitstream.
func (pw *PkgEncoder) NewEncoder(k SectionKind, marker SyncMarker) Encoder {
	e := pw.NewEncoderRaw(k)
	e.Sync(marker)
	return e
}

// NewEncoderRaw returns an Encoder for a new element within the given
// section.
//
// Most callers should use NewEncoder instead.
func (pw *PkgEncoder) NewEncoderRaw(k SectionKind) Encoder {
	idx := RelIndex(len(pw.elems[k]))
	pw.elems[k] = append(pw.elems[k], "") // placeholder

	return Encoder{
		p:   pw,
		k:   k,
		Idx: idx,
	}
}

// An Encoder provides methods for encoding an individual element's
// bitstream data.
type Encoder struct {
	p *PkgEncoder

	Relocs   []RelocEnt
	RelocMap map[RelocEnt]uint32
	Data     bytes.Buffer // accumulated element bitstream data

	encodingRelocHeader bool

	k   SectionKind
	Idx RelIndex // index within relocation section
}

// Flush finalizes the element's bitstream and returns its [RelIndex].
func (w *Encoder) Flush() RelIndex {
	var sb strings.Builder

	// Backup the data so we write the relocations at the front.
	var tmp bytes.Buffer
	io.Copy(&tmp, &w.Data)

	// TODO(mdempsky): Consider writing these out separately so they're
	// easier to strip, along with function bodies, so that we can prune
	// down to just the data that's relevant to go/types.
	if w.encodingRelocHeader {
		panic("encodingRelocHeader already true; recursive flush?")
	}
	w.encodingRelocHeader = true
	w.Sync(SyncRelocs)
	w.Len(len(w.Relocs))
	for _, rEnt := range w.Relocs {
		w.Sync(SyncReloc)
		w.Len(int(rEnt.Kind))
		w.Len(int(rEnt.Idx))
	}

	io.Copy(&sb, &w.Data)
	io.Copy(&sb, &tmp)
	w.p.elems[w.k][w.Idx] = sb.String()

	return w.Idx
}

func (w *Encoder) checkErr(err error) {
	if err != nil {
		panicf("unexpected encoding error: %v", err)
	}
}

func (w *Encoder) rawUvarint(x uint64) {
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(buf[:], x)
	_, err := w.Data.Write(buf[:n])
	w.checkErr(err)
}

func (w *Encoder) rawVarint(x int64) {
	// Zig-zag encode.
	ux := uint64(x) << 1
	if x < 0 {
		ux = ^ux
	}

	w.rawUvarint(ux)
}

func (w *Encoder) rawReloc(k SectionKind, idx RelIndex) int {
	e := RelocEnt{k, idx}
	if w.RelocMap != nil {
		if i, ok := w.RelocMap[e]; ok {
			return int(i)
		}
	} else {
		w.RelocMap = make(map[RelocEnt]uint32)
	}

	i := len(w.Relocs)
	w.RelocMap[e] = uint32(i)
	w.Relocs = append(w.Relocs, e)
	return i
}

func (w *Encoder) Sync(m SyncMarker) {
	if !w.p.SyncMarkers() {
		return
	}

	// Writing out stack frame string references requires working
	// relocations, but writing out the relocations themselves involves
	// sync markers. To prevent infinite recursion, we simply trim the
	// stack frame for sync markers within the relocation header.
	var frames []string
	if !w.encodingRelocHeader && w.p.syncFrames > 0 {
		pcs := make([]uintptr, w.p.syncFrames)
		n := runtime.Callers(2, pcs)
		frames = fmtFrames(pcs[:n]...)
	}

	// TODO(mdempsky): Save space by writing out stack frames as a
	// linked list so we can share common stack frames.
	w.rawUvarint(uint64(m))
	w.rawUvarint(uint64(len(frames)))
	for _, frame := range frames {
		w.rawUvarint(uint64(w.rawReloc(SectionString, w.p.StringIdx(frame))))
	}
}

// Bool encodes and writes a bool value into the element bitstream,
// and then returns the bool value.
//
// For simple, 2-alternative encodings, the idiomatic way to call Bool
// is something like:
//
//	if w.Bool(x != 0) {
//		// alternative #1
//	} else {
//		// alternative #2
//	}
//
// For multi-alternative encodings, use Code instead.
func (w *Encoder) Bool(b bool) bool {
	w.Sync(SyncBool)
	var x byte
	if b {
		x = 1
	}
	err := w.Data.WriteByte(x)
	w.checkErr(err)
	return b
}

// Int64 encodes and writes an int64 value into the element bitstream.
func (w *Encoder) Int64(x int64) {
	w.Sync(SyncInt64)
	w.rawVarint(x)
}

// Uint64 encodes and writes a uint64 value into the element bitstream.
func (w *Encoder) Uint64(x uint64) {
	w.Sync(SyncUint64)
	w.rawUvarint(x)
}

// Len encodes and writes a non-negative int value into the element bitstream.
func (w *Encoder) Len(x int) { assert(x >= 0); w.Uint64(uint64(x)) }

// Int encodes and writes an int value into the element bitstream.
func (w *Encoder) Int(x int) { w.Int64(int64(x)) }

// Uint encodes and writes a uint value into the element bitstream.
func (w *Encoder) Uint(x uint) { w.Uint64(uint64(x)) }

// Reloc encodes and writes a relocation for the given (section,
// index) pair into the element bitstream.
//
// Note: Only the index is formally written into the element
// bitstream, so bitstream decoders must know from context which
// section an encoded relocation refers to.
func (w *Encoder) Reloc(k SectionKind, idx RelIndex) {
	w.Sync(SyncUseReloc)
	w.Len(w.rawReloc(k, idx))
}

// Code encodes and writes a Code value into the element bitstream.
func (w *Encoder) Code(c Code) {
	w.Sync(c.Marker())
	w.Len(c.Value())
}

// String encodes and writes a string value into the element
// bitstream.
//
// Internally, strings are deduplicated by adding them to the strings
// section (if not already present), and then writing a relocation
// into the element bitstream.
func (w *Encoder) String(s string) {
	w.StringRef(w.p.StringIdx(s))
}

// StringRef writes a reference to the given index, which must be a
// previously encoded string value.
func (w *Encoder) StringRef(idx RelIndex) {
	w.Sync(SyncString)
	w.Reloc(SectionString, idx)
}

// Strings encodes and writes a variable-length slice of strings into
// the element bitstream.
func (w *Encoder) Strings(ss []string) {
	w.Len(len(ss))
	for _, s := range ss {
		w.String(s)
	}
}

// Value encodes and writes a constant.Value into the element
// bitstream.
func (w *Encoder) Value(val constant.Value) {
	w.Sync(SyncValue)
	if w.Bool(val.Kind() == constant.Complex) {
		w.scalar(constant.Real(val))
		w.scalar(constant.Imag(val))
	} else {
		w.scalar(val)
	}
}

func (w *Encoder) scalar(val constant.Value) {
	switch v := constant.Val(val).(type) {
	default:
		panicf("unhandled %v (%v)", val, val.Kind())
	case bool:
		w.Code(ValBool)
		w.Bool(v)
	case string:
		w.Code(ValString)
		w.String(v)
	case int64:
		w.Code(ValInt64)
		w.Int64(v)
	case *big.Int:
		w.Code(ValBigInt)
		w.bigInt(v)
	case *big.Rat:
		w.Code(ValBigRat)
		w.bigInt(v.Num())
		w.bigInt(v.Denom())
	case *big.Float:
		w.Code(ValBigFloat)
		w.bigFloat(v)
	}
}

func (w *Encoder) bigInt(v *big.Int) {
	b := v.Bytes()
	w.String(string(b)) // TODO: More efficient encoding.
	w.Bool(v.Sign() < 0)
}

func (w *Encoder) bigFloat(v *big.Float) {
	b := v.Append(nil, 'p', -1)
	w.String(string(b)) // TODO: More efficient encoding.
}

// Version reports the version of the bitstream.
func (w *Encoder) Version() Version { return w.p.version }
