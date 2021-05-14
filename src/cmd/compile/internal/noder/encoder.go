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

func (w *encoder) sync(m syncMarker) {
	if debug {
		err := w.data.WriteByte(byte(m))
		w.checkErr(err)
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
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutVarint(buf[:], x)
	_, err := w.data.Write(buf[:n])
	w.checkErr(err)
}

func (w *encoder) uint64(x uint64) {
	w.sync(syncUint64)
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(buf[:], x)
	_, err := w.data.Write(buf[:n])
	w.checkErr(err)
}

func (w *encoder) len(x int)   { assert(x >= 0); w.uint64(uint64(x)) }
func (w *encoder) int(x int)   { w.int64(int64(x)) }
func (w *encoder) uint(x uint) { w.uint64(uint64(x)) }

func (w *encoder) reloc(r reloc, idx int) {
	w.sync(syncUseReloc)

	// TODO(mdempsky): Use map for lookup.
	for i, rent := range w.relocs {
		if rent.kind == r && rent.idx == idx {
			w.len(i)
			return
		}
	}

	w.len(len(w.relocs))
	w.relocs = append(w.relocs, relocEnt{r, idx})
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

func (w *encoder) rawValue(val constant.Value) {
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
