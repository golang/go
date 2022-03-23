// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

import (
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"math/big"
	"os"
	"runtime"
	"strings"
)

type PkgDecoder struct {
	pkgPath string

	elemEndsEnds [numRelocs]uint32
	elemEnds     []uint32
	elemData     string // last 8 bytes are fingerprint
}

func (pr *PkgDecoder) PkgPath() string { return pr.pkgPath }

func NewPkgDecoder(pkgPath, input string) PkgDecoder {
	pr := PkgDecoder{
		pkgPath: pkgPath,
	}

	// TODO(mdempsky): Implement direct indexing of input string to
	// avoid copying the position information.

	r := strings.NewReader(input)

	var version uint32
	assert(binary.Read(r, binary.LittleEndian, &version) == nil)
	assert(version == 0)

	assert(binary.Read(r, binary.LittleEndian, pr.elemEndsEnds[:]) == nil)

	pr.elemEnds = make([]uint32, pr.elemEndsEnds[len(pr.elemEndsEnds)-1])
	assert(binary.Read(r, binary.LittleEndian, pr.elemEnds[:]) == nil)

	pos, err := r.Seek(0, os.SEEK_CUR)
	assert(err == nil)

	pr.elemData = input[pos:]
	assert(len(pr.elemData)-8 == int(pr.elemEnds[len(pr.elemEnds)-1]))

	return pr
}

func (pr *PkgDecoder) NumElems(k RelocKind) int {
	count := int(pr.elemEndsEnds[k])
	if k > 0 {
		count -= int(pr.elemEndsEnds[k-1])
	}
	return count
}

func (pr *PkgDecoder) TotalElems() int {
	return len(pr.elemEnds)
}

func (pr *PkgDecoder) Fingerprint() [8]byte {
	var fp [8]byte
	copy(fp[:], pr.elemData[len(pr.elemData)-8:])
	return fp
}

func (pr *PkgDecoder) AbsIdx(k RelocKind, idx int) int {
	absIdx := idx
	if k > 0 {
		absIdx += int(pr.elemEndsEnds[k-1])
	}
	if absIdx >= int(pr.elemEndsEnds[k]) {
		errorf("%v:%v is out of bounds; %v", k, idx, pr.elemEndsEnds)
	}
	return absIdx
}

func (pr *PkgDecoder) DataIdx(k RelocKind, idx int) string {
	absIdx := pr.AbsIdx(k, idx)

	var start uint32
	if absIdx > 0 {
		start = pr.elemEnds[absIdx-1]
	}
	end := pr.elemEnds[absIdx]

	return pr.elemData[start:end]
}

func (pr *PkgDecoder) StringIdx(idx int) string {
	return pr.DataIdx(RelocString, idx)
}

func (pr *PkgDecoder) NewDecoder(k RelocKind, idx int, marker SyncMarker) Decoder {
	r := pr.NewDecoderRaw(k, idx)
	r.Sync(marker)
	return r
}

func (pr *PkgDecoder) NewDecoderRaw(k RelocKind, idx int) Decoder {
	r := Decoder{
		common: pr,
		k:      k,
		Idx:    idx,
	}

	// TODO(mdempsky) r.data.Reset(...) after #44505 is resolved.
	r.Data = *strings.NewReader(pr.DataIdx(k, idx))

	r.Sync(SyncRelocs)
	r.Relocs = make([]RelocEnt, r.Len())
	for i := range r.Relocs {
		r.Sync(SyncReloc)
		r.Relocs[i] = RelocEnt{RelocKind(r.Len()), r.Len()}
	}

	return r
}

type Decoder struct {
	common *PkgDecoder

	Relocs []RelocEnt
	Data   strings.Reader

	k   RelocKind
	Idx int
}

func (r *Decoder) checkErr(err error) {
	if err != nil {
		errorf("unexpected decoding error: %w", err)
	}
}

func (r *Decoder) rawUvarint() uint64 {
	x, err := binary.ReadUvarint(&r.Data)
	r.checkErr(err)
	return x
}

func (r *Decoder) rawVarint() int64 {
	ux := r.rawUvarint()

	// Zig-zag decode.
	x := int64(ux >> 1)
	if ux&1 != 0 {
		x = ^x
	}
	return x
}

func (r *Decoder) rawReloc(k RelocKind, idx int) int {
	e := r.Relocs[idx]
	assert(e.Kind == k)
	return e.Idx
}

func (r *Decoder) Sync(mWant SyncMarker) {
	if !EnableSync {
		return
	}

	pos, _ := r.Data.Seek(0, os.SEEK_CUR) // TODO(mdempsky): io.SeekCurrent after #44505 is resolved
	mHave := SyncMarker(r.rawUvarint())
	writerPCs := make([]int, r.rawUvarint())
	for i := range writerPCs {
		writerPCs[i] = int(r.rawUvarint())
	}

	if mHave == mWant {
		return
	}

	// There's some tension here between printing:
	//
	// (1) full file paths that tools can recognize (e.g., so emacs
	//     hyperlinks the "file:line" text for easy navigation), or
	//
	// (2) short file paths that are easier for humans to read (e.g., by
	//     omitting redundant or irrelevant details, so it's easier to
	//     focus on the useful bits that remain).
	//
	// The current formatting favors the former, as it seems more
	// helpful in practice. But perhaps the formatting could be improved
	// to better address both concerns. For example, use relative file
	// paths if they would be shorter, or rewrite file paths to contain
	// "$GOROOT" (like objabi.AbsFile does) if tools can be taught how
	// to reliably expand that again.

	fmt.Printf("export data desync: package %q, section %v, index %v, offset %v\n", r.common.pkgPath, r.k, r.Idx, pos)

	fmt.Printf("\nfound %v, written at:\n", mHave)
	if len(writerPCs) == 0 {
		fmt.Printf("\t[stack trace unavailable; recompile package %q with -d=syncframes]\n", r.common.pkgPath)
	}
	for _, pc := range writerPCs {
		fmt.Printf("\t%s\n", r.common.StringIdx(r.rawReloc(RelocString, pc)))
	}

	fmt.Printf("\nexpected %v, reading at:\n", mWant)
	var readerPCs [32]uintptr // TODO(mdempsky): Dynamically size?
	n := runtime.Callers(2, readerPCs[:])
	for _, pc := range fmtFrames(readerPCs[:n]...) {
		fmt.Printf("\t%s\n", pc)
	}

	// We already printed a stack trace for the reader, so now we can
	// simply exit. Printing a second one with panic or base.Fatalf
	// would just be noise.
	os.Exit(1)
}

func (r *Decoder) Bool() bool {
	r.Sync(SyncBool)
	x, err := r.Data.ReadByte()
	r.checkErr(err)
	assert(x < 2)
	return x != 0
}

func (r *Decoder) Int64() int64 {
	r.Sync(SyncInt64)
	return r.rawVarint()
}

func (r *Decoder) Uint64() uint64 {
	r.Sync(SyncUint64)
	return r.rawUvarint()
}

func (r *Decoder) Len() int   { x := r.Uint64(); v := int(x); assert(uint64(v) == x); return v }
func (r *Decoder) Int() int   { x := r.Int64(); v := int(x); assert(int64(v) == x); return v }
func (r *Decoder) Uint() uint { x := r.Uint64(); v := uint(x); assert(uint64(v) == x); return v }

// TODO(mdempsky): Ideally this method would have signature "Code[T
// Code] T" instead, but we don't allow generic methods and the
// compiler can't depend on generics yet anyway.
func (r *Decoder) Code(mark SyncMarker) int {
	r.Sync(mark)
	return r.Len()
}

func (r *Decoder) Reloc(k RelocKind) int {
	r.Sync(SyncUseReloc)
	return r.rawReloc(k, r.Len())
}

func (r *Decoder) String() string {
	r.Sync(SyncString)
	return r.common.StringIdx(r.Reloc(RelocString))
}

func (r *Decoder) Strings() []string {
	res := make([]string, r.Len())
	for i := range res {
		res[i] = r.String()
	}
	return res
}

func (r *Decoder) Value() constant.Value {
	r.Sync(SyncValue)
	isComplex := r.Bool()
	val := r.scalar()
	if isComplex {
		val = constant.BinaryOp(val, token.ADD, constant.MakeImag(r.scalar()))
	}
	return val
}

func (r *Decoder) scalar() constant.Value {
	switch tag := CodeVal(r.Code(SyncVal)); tag {
	default:
		panic(fmt.Errorf("unexpected scalar tag: %v", tag))

	case ValBool:
		return constant.MakeBool(r.Bool())
	case ValString:
		return constant.MakeString(r.String())
	case ValInt64:
		return constant.MakeInt64(r.Int64())
	case ValBigInt:
		return constant.Make(r.bigInt())
	case ValBigRat:
		num := r.bigInt()
		denom := r.bigInt()
		return constant.Make(new(big.Rat).SetFrac(num, denom))
	case ValBigFloat:
		return constant.Make(r.bigFloat())
	}
}

func (r *Decoder) bigInt() *big.Int {
	v := new(big.Int).SetBytes([]byte(r.String()))
	if r.Bool() {
		v.Neg(v)
	}
	return v
}

func (r *Decoder) bigFloat() *big.Float {
	v := new(big.Float).SetPrec(512)
	assert(v.UnmarshalText([]byte(r.String())) == nil)
	return v
}

// @@@ Helpers

// TODO(mdempsky): These should probably be removed. I think they're a
// smell that the export data format is not yet quite right.

func (pr *PkgDecoder) PeekPkgPath(idx int) string {
	r := pr.NewDecoder(RelocPkg, idx, SyncPkgDef)
	path := r.String()
	if path == "" {
		path = pr.pkgPath
	}
	return path
}

func (pr *PkgDecoder) PeekObj(idx int) (string, string, CodeObj) {
	r := pr.NewDecoder(RelocName, idx, SyncObject1)
	r.Sync(SyncSym)
	r.Sync(SyncPkg)
	path := pr.PeekPkgPath(r.Reloc(RelocPkg))
	name := r.String()
	assert(name != "")

	tag := CodeObj(r.Code(SyncCodeObj))

	return path, name, tag
}
