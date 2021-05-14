// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"math/big"
	"os"
	"strings"

	"cmd/compile/internal/base"
)

type pkgDecoder struct {
	pkgPath string

	elemEndsEnds [numRelocs]uint32
	elemEnds     []uint32
	elemData     string
}

func newPkgDecoder(pkgPath, input string) pkgDecoder {
	pr := pkgDecoder{
		pkgPath: pkgPath,
	}

	// TODO(mdempsky): Implement direct indexing of input string to
	// avoid copying the position information.

	r := strings.NewReader(input)

	assert(binary.Read(r, binary.LittleEndian, pr.elemEndsEnds[:]) == nil)

	pr.elemEnds = make([]uint32, pr.elemEndsEnds[len(pr.elemEndsEnds)-1])
	assert(binary.Read(r, binary.LittleEndian, pr.elemEnds[:]) == nil)

	pos, err := r.Seek(0, os.SEEK_CUR)
	assert(err == nil)

	pr.elemData = input[pos:]
	assert(len(pr.elemData) == int(pr.elemEnds[len(pr.elemEnds)-1]))

	return pr
}

func (pr *pkgDecoder) numElems(k reloc) int {
	count := int(pr.elemEndsEnds[k])
	if k > 0 {
		count -= int(pr.elemEndsEnds[k-1])
	}
	return count
}

func (pr *pkgDecoder) totalElems() int {
	return len(pr.elemEnds)
}

func (pr *pkgDecoder) absIdx(k reloc, idx int) int {
	absIdx := idx
	if k > 0 {
		absIdx += int(pr.elemEndsEnds[k-1])
	}
	if absIdx >= int(pr.elemEndsEnds[k]) {
		base.Fatalf("%v:%v is out of bounds; %v", k, idx, pr.elemEndsEnds)
	}
	return absIdx
}

func (pr *pkgDecoder) dataIdx(k reloc, idx int) string {
	absIdx := pr.absIdx(k, idx)

	var start uint32
	if absIdx > 0 {
		start = pr.elemEnds[absIdx-1]
	}
	end := pr.elemEnds[absIdx]

	return pr.elemData[start:end]
}

func (pr *pkgDecoder) stringIdx(idx int) string {
	return pr.dataIdx(relocString, idx)
}

func (pr *pkgDecoder) newDecoder(k reloc, idx int, marker syncMarker) decoder {
	r := pr.newDecoderRaw(k, idx)
	r.sync(marker)
	return r
}

func (pr *pkgDecoder) newDecoderRaw(k reloc, idx int) decoder {
	r := decoder{
		common: pr,
		k:      k,
		idx:    idx,
	}

	// TODO(mdempsky) r.data.Reset(...) after #44505 is resolved.
	r.data = *strings.NewReader(pr.dataIdx(k, idx))

	r.sync(syncRelocs)
	r.relocs = make([]relocEnt, r.len())
	for i := range r.relocs {
		r.sync(syncReloc)
		r.relocs[i] = relocEnt{reloc(r.len()), r.len()}
	}

	return r
}

type decoder struct {
	common *pkgDecoder

	relocs []relocEnt
	data   strings.Reader

	k   reloc
	idx int
}

func (r *decoder) checkErr(err error) {
	if err != nil {
		base.Fatalf("unexpected error: %v", err)
	}
}

func (r *decoder) sync(m syncMarker) {
	if debug {
		pos, err0 := r.data.Seek(0, os.SEEK_CUR)
		x, err := r.data.ReadByte()
		r.checkErr(err)
		if x != byte(m) {
			// TODO(mdempsky): Revisit this error message, and make it more
			// useful (e.g., include r.p.pkgPath).
			base.Fatalf("data sync error: found %v at %v (%v) in (%v:%v), but expected %v", syncMarker(x), pos, err0, r.k, r.idx, m)
		}
	}
}

func (r *decoder) bool() bool {
	r.sync(syncBool)
	x, err := r.data.ReadByte()
	r.checkErr(err)
	assert(x < 2)
	return x != 0
}

func (r *decoder) int64() int64 {
	r.sync(syncInt64)
	x, err := binary.ReadVarint(&r.data)
	r.checkErr(err)
	return x
}

func (r *decoder) uint64() uint64 {
	r.sync(syncUint64)
	x, err := binary.ReadUvarint(&r.data)
	r.checkErr(err)
	return x
}

func (r *decoder) len() int   { x := r.uint64(); v := int(x); assert(uint64(v) == x); return v }
func (r *decoder) int() int   { x := r.int64(); v := int(x); assert(int64(v) == x); return v }
func (r *decoder) uint() uint { x := r.uint64(); v := uint(x); assert(uint64(v) == x); return v }

func (r *decoder) code(mark syncMarker) int {
	r.sync(mark)
	return r.len()
}

func (r *decoder) reloc(k reloc) int {
	r.sync(syncUseReloc)
	idx := r.len()

	e := r.relocs[idx]
	assert(e.kind == k)
	return e.idx
}

func (r *decoder) string() string {
	r.sync(syncString)
	return r.common.stringIdx(r.reloc(relocString))
}

func (r *decoder) strings() []string {
	res := make([]string, r.len())
	for i := range res {
		res[i] = r.string()
	}
	return res
}

func (r *decoder) rawValue() constant.Value {
	isComplex := r.bool()
	val := r.scalar()
	if isComplex {
		val = constant.BinaryOp(val, token.ADD, constant.MakeImag(r.scalar()))
	}
	return val
}

func (r *decoder) scalar() constant.Value {
	switch tag := codeVal(r.code(syncVal)); tag {
	default:
		panic(fmt.Sprintf("unexpected scalar tag: %v", tag))

	case valBool:
		return constant.MakeBool(r.bool())
	case valString:
		return constant.MakeString(r.string())
	case valInt64:
		return constant.MakeInt64(r.int64())
	case valBigInt:
		return constant.Make(r.bigInt())
	case valBigRat:
		num := r.bigInt()
		denom := r.bigInt()
		return constant.Make(new(big.Rat).SetFrac(num, denom))
	case valBigFloat:
		return constant.Make(r.bigFloat())
	}
}

func (r *decoder) bigInt() *big.Int {
	v := new(big.Int).SetBytes([]byte(r.string()))
	if r.bool() {
		v.Neg(v)
	}
	return v
}

func (r *decoder) bigFloat() *big.Float {
	v := new(big.Float).SetPrec(512)
	assert(v.UnmarshalText([]byte(r.string())) == nil)
	return v
}
