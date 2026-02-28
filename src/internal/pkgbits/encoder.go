// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"go/constant"
	"io"
	"math/big"
	"runtime"
)

type PkgEncoder struct {
	elems [numRelocs][]string

	stringsIdx map[string]int

	syncFrames int
}

func NewPkgEncoder(syncFrames int) PkgEncoder {
	return PkgEncoder{
		stringsIdx: make(map[string]int),
		syncFrames: syncFrames,
	}
}

func (pw *PkgEncoder) DumpTo(out0 io.Writer) (fingerprint [8]byte) {
	h := md5.New()
	out := io.MultiWriter(out0, h)

	writeUint32 := func(x uint32) {
		assert(binary.Write(out, binary.LittleEndian, x) == nil)
	}

	writeUint32(0) // version

	var sum uint32
	for _, elems := range &pw.elems {
		sum += uint32(len(elems))
		writeUint32(sum)
	}

	sum = 0
	for _, elems := range &pw.elems {
		for _, elem := range elems {
			sum += uint32(len(elem))
			writeUint32(sum)
		}
	}

	for _, elems := range &pw.elems {
		for _, elem := range elems {
			_, err := io.WriteString(out, elem)
			assert(err == nil)
		}
	}

	copy(fingerprint[:], h.Sum(nil))
	_, err := out0.Write(fingerprint[:])
	assert(err == nil)

	return
}

func (pw *PkgEncoder) StringIdx(s string) int {
	if idx, ok := pw.stringsIdx[s]; ok {
		assert(pw.elems[RelocString][idx] == s)
		return idx
	}

	idx := len(pw.elems[RelocString])
	pw.elems[RelocString] = append(pw.elems[RelocString], s)
	pw.stringsIdx[s] = idx
	return idx
}

func (pw *PkgEncoder) NewEncoder(k RelocKind, marker SyncMarker) Encoder {
	e := pw.NewEncoderRaw(k)
	e.Sync(marker)
	return e
}

func (pw *PkgEncoder) NewEncoderRaw(k RelocKind) Encoder {
	idx := len(pw.elems[k])
	pw.elems[k] = append(pw.elems[k], "") // placeholder

	return Encoder{
		p:   pw,
		k:   k,
		Idx: idx,
	}
}

// Encoders

type Encoder struct {
	p *PkgEncoder

	Relocs []RelocEnt
	Data   bytes.Buffer

	encodingRelocHeader bool

	k   RelocKind
	Idx int
}

func (w *Encoder) Flush() int {
	var sb bytes.Buffer // TODO(mdempsky): strings.Builder after #44505 is resolved

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
	for _, rent := range w.Relocs {
		w.Sync(SyncReloc)
		w.Len(int(rent.Kind))
		w.Len(rent.Idx)
	}

	io.Copy(&sb, &w.Data)
	io.Copy(&sb, &tmp)
	w.p.elems[w.k][w.Idx] = sb.String()

	return w.Idx
}

func (w *Encoder) checkErr(err error) {
	if err != nil {
		errorf("unexpected encoding error: %v", err)
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

func (w *Encoder) rawReloc(r RelocKind, idx int) int {
	// TODO(mdempsky): Use map for lookup.
	for i, rent := range w.Relocs {
		if rent.Kind == r && rent.Idx == idx {
			return i
		}
	}

	i := len(w.Relocs)
	w.Relocs = append(w.Relocs, RelocEnt{r, idx})
	return i
}

func (w *Encoder) Sync(m SyncMarker) {
	if !EnableSync {
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
		w.rawUvarint(uint64(w.rawReloc(RelocString, w.p.StringIdx(frame))))
	}
}

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

func (w *Encoder) Int64(x int64) {
	w.Sync(SyncInt64)
	w.rawVarint(x)
}

func (w *Encoder) Uint64(x uint64) {
	w.Sync(SyncUint64)
	w.rawUvarint(x)
}

func (w *Encoder) Len(x int)   { assert(x >= 0); w.Uint64(uint64(x)) }
func (w *Encoder) Int(x int)   { w.Int64(int64(x)) }
func (w *Encoder) Uint(x uint) { w.Uint64(uint64(x)) }

func (w *Encoder) Reloc(r RelocKind, idx int) {
	w.Sync(SyncUseReloc)
	w.Len(w.rawReloc(r, idx))
}

func (w *Encoder) Code(c Code) {
	w.Sync(c.Marker())
	w.Len(c.Value())
}

func (w *Encoder) String(s string) {
	w.Sync(SyncString)
	w.Reloc(RelocString, w.p.StringIdx(s))
}

func (w *Encoder) Strings(ss []string) {
	w.Len(len(ss))
	for _, s := range ss {
		w.String(s)
	}
}

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
		errorf("unhandled %v (%v)", val, val.Kind())
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
