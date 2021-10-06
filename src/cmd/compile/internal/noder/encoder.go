// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/constant"
	"io"
	"math/big"
	"runtime"

	"cmd/compile/internal/base"
)

type pkgEncoder struct {
	elems [numRelocs][]string

	stringsIdx map[string]int
}

func newPkgEncoder() pkgEncoder {
	return pkgEncoder{
		stringsIdx: make(map[string]int),
	}
}

func (pw *pkgEncoder) dump(out io.Writer) {
	writeUint32 := func(x uint32) {
		assert(binary.Write(out, binary.LittleEndian, x) == nil)
	}

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
}

func (pw *pkgEncoder) stringIdx(s string) int {
	if idx, ok := pw.stringsIdx[s]; ok {
		assert(pw.elems[relocString][idx] == s)
		return idx
	}

	idx := len(pw.elems[relocString])
	pw.elems[relocString] = append(pw.elems[relocString], s)
	pw.stringsIdx[s] = idx
	return idx
}

func (pw *pkgEncoder) newEncoder(k reloc, marker syncMarker) encoder {
	e := pw.newEncoderRaw(k)
	e.sync(marker)
	return e
}

func (pw *pkgEncoder) newEncoderRaw(k reloc) encoder {
	idx := len(pw.elems[k])
	pw.elems[k] = append(pw.elems[k], "") // placeholder

	return encoder{
		p:   pw,
		k:   k,
		idx: idx,
	}
}

// Encoders

type encoder struct {
	p *pkgEncoder

	relocs []relocEnt
	data   bytes.Buffer

	encodingRelocHeader bool

	k   reloc
	idx int
}

func (w *encoder) flush() int {
	var sb bytes.Buffer // TODO(mdempsky): strings.Builder after #44505 is resolved

	// Backup the data so we write the relocations at the front.
	var tmp bytes.Buffer
	io.Copy(&tmp, &w.data)

	// TODO(mdempsky): Consider writing these out separately so they're
	// easier to strip, along with function bodies, so that we can prune
	// down to just the data that's relevant to go/types.
	if w.encodingRelocHeader {
		base.Fatalf("encodingRelocHeader already true; recursive flush?")
	}
	w.encodingRelocHeader = true
	w.sync(syncRelocs)
	w.len(len(w.relocs))
	for _, rent := range w.relocs {
		w.sync(syncReloc)
		w.len(int(rent.kind))
		w.len(rent.idx)
	}

	io.Copy(&sb, &w.data)
	io.Copy(&sb, &tmp)
	w.p.elems[w.k][w.idx] = sb.String()

	return w.idx
}

func (w *encoder) checkErr(err error) {
	if err != nil {
		base.Fatalf("unexpected error: %v", err)
	}
}

func (w *encoder) rawUvarint(x uint64) {
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(buf[:], x)
	_, err := w.data.Write(buf[:n])
	w.checkErr(err)
}

func (w *encoder) rawVarint(x int64) {
	// Zig-zag encode.
	ux := uint64(x) << 1
	if x < 0 {
		ux = ^ux
	}

	w.rawUvarint(ux)
}

func (w *encoder) rawReloc(r reloc, idx int) int {
	// TODO(mdempsky): Use map for lookup.
	for i, rent := range w.relocs {
		if rent.kind == r && rent.idx == idx {
			return i
		}
	}

	i := len(w.relocs)
	w.relocs = append(w.relocs, relocEnt{r, idx})
	return i
}

func (w *encoder) sync(m syncMarker) {
	if !enableSync {
		return
	}

	// Writing out stack frame string references requires working
	// relocations, but writing out the relocations themselves involves
	// sync markers. To prevent infinite recursion, we simply trim the
	// stack frame for sync markers within the relocation header.
	var frames []string
	if !w.encodingRelocHeader && base.Debug.SyncFrames > 0 {
		pcs := make([]uintptr, base.Debug.SyncFrames)
		n := runtime.Callers(2, pcs)
		frames = fmtFrames(pcs[:n]...)
	}

	// TODO(mdempsky): Save space by writing out stack frames as a
	// linked list so we can share common stack frames.
	w.rawUvarint(uint64(m))
	w.rawUvarint(uint64(len(frames)))
	for _, frame := range frames {
		w.rawUvarint(uint64(w.rawReloc(relocString, w.p.stringIdx(frame))))
	}
}

func (w *encoder) bool(b bool) bool {
	w.sync(syncBool)
	var x byte
	if b {
		x = 1
	}
	err := w.data.WriteByte(x)
	w.checkErr(err)
	return b
}

func (w *encoder) int64(x int64) {
	w.sync(syncInt64)
	w.rawVarint(x)
}

func (w *encoder) uint64(x uint64) {
	w.sync(syncUint64)
	w.rawUvarint(x)
}

func (w *encoder) len(x int)   { assert(x >= 0); w.uint64(uint64(x)) }
func (w *encoder) int(x int)   { w.int64(int64(x)) }
func (w *encoder) uint(x uint) { w.uint64(uint64(x)) }

func (w *encoder) reloc(r reloc, idx int) {
	w.sync(syncUseReloc)
	w.len(w.rawReloc(r, idx))
}

func (w *encoder) code(c code) {
	w.sync(c.marker())
	w.len(c.value())
}

func (w *encoder) string(s string) {
	w.sync(syncString)
	w.reloc(relocString, w.p.stringIdx(s))
}

func (w *encoder) strings(ss []string) {
	w.len(len(ss))
	for _, s := range ss {
		w.string(s)
	}
}

func (w *encoder) value(val constant.Value) {
	w.sync(syncValue)
	if w.bool(val.Kind() == constant.Complex) {
		w.scalar(constant.Real(val))
		w.scalar(constant.Imag(val))
	} else {
		w.scalar(val)
	}
}

func (w *encoder) scalar(val constant.Value) {
	switch v := constant.Val(val).(type) {
	default:
		panic(fmt.Sprintf("unhandled %v (%v)", val, val.Kind()))
	case bool:
		w.code(valBool)
		w.bool(v)
	case string:
		w.code(valString)
		w.string(v)
	case int64:
		w.code(valInt64)
		w.int64(v)
	case *big.Int:
		w.code(valBigInt)
		w.bigInt(v)
	case *big.Rat:
		w.code(valBigRat)
		w.bigInt(v.Num())
		w.bigInt(v.Denom())
	case *big.Float:
		w.code(valBigFloat)
		w.bigFloat(v)
	}
}

func (w *encoder) bigInt(v *big.Int) {
	b := v.Bytes()
	w.string(string(b)) // TODO: More efficient encoding.
	w.bool(v.Sign() < 0)
}

func (w *encoder) bigFloat(v *big.Float) {
	b := v.Append(nil, 'p', -1)
	w.string(string(b)) // TODO: More efficient encoding.
}
