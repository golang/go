// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

import (
	"encoding/binary"
	"errors"
	"fmt"
	"go/constant"
	"go/token"
	"io"
	"math/big"
	"os"
	"runtime"
	"strings"
)

// A PkgDecoder provides methods for decoding a package's Unified IR
// export data.
type PkgDecoder struct {
	// version is the file format version.
	version uint32

	// sync indicates whether the file uses sync markers.
	sync bool

	// pkgPath is the package path for the package to be decoded.
	//
	// TODO(mdempsky): Remove; unneeded since CL 391014.
	pkgPath string

	// elemData is the full data payload of the encoded package.
	// Elements are densely and contiguously packed together.
	//
	// The last 8 bytes of elemData are the package fingerprint.
	elemData string

	// elemEnds stores the byte-offset end positions of element
	// bitstreams within elemData.
	//
	// For example, element I's bitstream data starts at elemEnds[I-1]
	// (or 0, if I==0) and ends at elemEnds[I].
	//
	// Note: elemEnds is indexed by absolute indices, not
	// section-relative indices.
	elemEnds []uint32

	// elemEndsEnds stores the index-offset end positions of relocation
	// sections within elemEnds.
	//
	// For example, section K's end positions start at elemEndsEnds[K-1]
	// (or 0, if K==0) and end at elemEndsEnds[K].
	elemEndsEnds [numRelocs]uint32

	scratchRelocEnt []RelocEnt
}

// PkgPath returns the package path for the package
//
// TODO(mdempsky): Remove; unneeded since CL 391014.
func (pr *PkgDecoder) PkgPath() string { return pr.pkgPath }

// SyncMarkers reports whether pr uses sync markers.
func (pr *PkgDecoder) SyncMarkers() bool { return pr.sync }

// NewPkgDecoder returns a PkgDecoder initialized to read the Unified
// IR export data from input. pkgPath is the package path for the
// compilation unit that produced the export data.
//
// TODO(mdempsky): Remove pkgPath parameter; unneeded since CL 391014.
func NewPkgDecoder(pkgPath, input string) PkgDecoder {
	pr := PkgDecoder{
		pkgPath: pkgPath,
	}

	// TODO(mdempsky): Implement direct indexing of input string to
	// avoid copying the position information.

	r := strings.NewReader(input)

	assert(binary.Read(r, binary.LittleEndian, &pr.version) == nil)

	switch pr.version {
	default:
		panic(fmt.Errorf("unsupported version: %v", pr.version))
	case 0:
		// no flags
	case 1:
		var flags uint32
		assert(binary.Read(r, binary.LittleEndian, &flags) == nil)
		pr.sync = flags&flagSyncMarkers != 0
	}

	assert(binary.Read(r, binary.LittleEndian, pr.elemEndsEnds[:]) == nil)

	pr.elemEnds = make([]uint32, pr.elemEndsEnds[len(pr.elemEndsEnds)-1])
	assert(binary.Read(r, binary.LittleEndian, pr.elemEnds[:]) == nil)

	pos, err := r.Seek(0, io.SeekCurrent)
	assert(err == nil)

	pr.elemData = input[pos:]
	assert(len(pr.elemData)-8 == int(pr.elemEnds[len(pr.elemEnds)-1]))

	return pr
}

// NumElems returns the number of elements in section k.
func (pr *PkgDecoder) NumElems(k RelocKind) int {
	count := int(pr.elemEndsEnds[k])
	if k > 0 {
		count -= int(pr.elemEndsEnds[k-1])
	}
	return count
}

// TotalElems returns the total number of elements across all sections.
func (pr *PkgDecoder) TotalElems() int {
	return len(pr.elemEnds)
}

// Fingerprint returns the package fingerprint.
func (pr *PkgDecoder) Fingerprint() [8]byte {
	var fp [8]byte
	copy(fp[:], pr.elemData[len(pr.elemData)-8:])
	return fp
}

// AbsIdx returns the absolute index for the given (section, index)
// pair.
func (pr *PkgDecoder) AbsIdx(k RelocKind, idx Index) int {
	absIdx := int(idx)
	if k > 0 {
		absIdx += int(pr.elemEndsEnds[k-1])
	}
	if absIdx >= int(pr.elemEndsEnds[k]) {
		errorf("%v:%v is out of bounds; %v", k, idx, pr.elemEndsEnds)
	}
	return absIdx
}

// DataIdx returns the raw element bitstream for the given (section,
// index) pair.
func (pr *PkgDecoder) DataIdx(k RelocKind, idx Index) string {
	absIdx := pr.AbsIdx(k, idx)

	var start uint32
	if absIdx > 0 {
		start = pr.elemEnds[absIdx-1]
	}
	end := pr.elemEnds[absIdx]

	return pr.elemData[start:end]
}

// StringIdx returns the string value for the given string index.
func (pr *PkgDecoder) StringIdx(idx Index) string {
	return pr.DataIdx(RelocString, idx)
}

// NewDecoder returns a Decoder for the given (section, index) pair,
// and decodes the given SyncMarker from the element bitstream.
func (pr *PkgDecoder) NewDecoder(k RelocKind, idx Index, marker SyncMarker) Decoder {
	r := pr.NewDecoderRaw(k, idx)
	r.Sync(marker)
	return r
}

// TempDecoder returns a Decoder for the given (section, index) pair,
// and decodes the given SyncMarker from the element bitstream.
// If possible the Decoder should be RetireDecoder'd when it is no longer
// needed, this will avoid heap allocations.
func (pr *PkgDecoder) TempDecoder(k RelocKind, idx Index, marker SyncMarker) Decoder {
	r := pr.TempDecoderRaw(k, idx)
	r.Sync(marker)
	return r
}

func (pr *PkgDecoder) RetireDecoder(d *Decoder) {
	pr.scratchRelocEnt = d.Relocs
	d.Relocs = nil
}

// NewDecoderRaw returns a Decoder for the given (section, index) pair.
//
// Most callers should use NewDecoder instead.
func (pr *PkgDecoder) NewDecoderRaw(k RelocKind, idx Index) Decoder {
	r := Decoder{
		common: pr,
		k:      k,
		Idx:    idx,
	}

	r.Data.Reset(pr.DataIdx(k, idx))
	r.Sync(SyncRelocs)
	r.Relocs = make([]RelocEnt, r.Len())
	for i := range r.Relocs {
		r.Sync(SyncReloc)
		r.Relocs[i] = RelocEnt{RelocKind(r.Len()), Index(r.Len())}
	}

	return r
}

func (pr *PkgDecoder) TempDecoderRaw(k RelocKind, idx Index) Decoder {
	r := Decoder{
		common: pr,
		k:      k,
		Idx:    idx,
	}

	r.Data.Reset(pr.DataIdx(k, idx))
	r.Sync(SyncRelocs)
	l := r.Len()
	if cap(pr.scratchRelocEnt) >= l {
		r.Relocs = pr.scratchRelocEnt[:l]
		pr.scratchRelocEnt = nil
	} else {
		r.Relocs = make([]RelocEnt, l)
	}
	for i := range r.Relocs {
		r.Sync(SyncReloc)
		r.Relocs[i] = RelocEnt{RelocKind(r.Len()), Index(r.Len())}
	}

	return r
}

// A Decoder provides methods for decoding an individual element's
// bitstream data.
type Decoder struct {
	common *PkgDecoder

	Relocs []RelocEnt
	Data   strings.Reader

	k   RelocKind
	Idx Index
}

func (r *Decoder) checkErr(err error) {
	if err != nil {
		errorf("unexpected decoding error: %w", err)
	}
}

func (r *Decoder) rawUvarint() uint64 {
	x, err := readUvarint(&r.Data)
	r.checkErr(err)
	return x
}

// readUvarint is a type-specialized copy of encoding/binary.ReadUvarint.
// This avoids the interface conversion and thus has better escape properties,
// which flows up the stack.
func readUvarint(r *strings.Reader) (uint64, error) {
	var x uint64
	var s uint
	for i := 0; i < binary.MaxVarintLen64; i++ {
		b, err := r.ReadByte()
		if err != nil {
			if i > 0 && err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return x, err
		}
		if b < 0x80 {
			if i == binary.MaxVarintLen64-1 && b > 1 {
				return x, overflow
			}
			return x | uint64(b)<<s, nil
		}
		x |= uint64(b&0x7f) << s
		s += 7
	}
	return x, overflow
}

var overflow = errors.New("pkgbits: readUvarint overflows a 64-bit integer")

func (r *Decoder) rawVarint() int64 {
	ux := r.rawUvarint()

	// Zig-zag decode.
	x := int64(ux >> 1)
	if ux&1 != 0 {
		x = ^x
	}
	return x
}

func (r *Decoder) rawReloc(k RelocKind, idx int) Index {
	e := r.Relocs[idx]
	assert(e.Kind == k)
	return e.Idx
}

// Sync decodes a sync marker from the element bitstream and asserts
// that it matches the expected marker.
//
// If EnableSync is false, then Sync is a no-op.
func (r *Decoder) Sync(mWant SyncMarker) {
	if !r.common.sync {
		return
	}

	pos, _ := r.Data.Seek(0, io.SeekCurrent)
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

// Bool decodes and returns a bool value from the element bitstream.
func (r *Decoder) Bool() bool {
	r.Sync(SyncBool)
	x, err := r.Data.ReadByte()
	r.checkErr(err)
	assert(x < 2)
	return x != 0
}

// Int64 decodes and returns an int64 value from the element bitstream.
func (r *Decoder) Int64() int64 {
	r.Sync(SyncInt64)
	return r.rawVarint()
}

// Int64 decodes and returns a uint64 value from the element bitstream.
func (r *Decoder) Uint64() uint64 {
	r.Sync(SyncUint64)
	return r.rawUvarint()
}

// Len decodes and returns a non-negative int value from the element bitstream.
func (r *Decoder) Len() int { x := r.Uint64(); v := int(x); assert(uint64(v) == x); return v }

// Int decodes and returns an int value from the element bitstream.
func (r *Decoder) Int() int { x := r.Int64(); v := int(x); assert(int64(v) == x); return v }

// Uint decodes and returns a uint value from the element bitstream.
func (r *Decoder) Uint() uint { x := r.Uint64(); v := uint(x); assert(uint64(v) == x); return v }

// Code decodes a Code value from the element bitstream and returns
// its ordinal value. It's the caller's responsibility to convert the
// result to an appropriate Code type.
//
// TODO(mdempsky): Ideally this method would have signature "Code[T
// Code] T" instead, but we don't allow generic methods and the
// compiler can't depend on generics yet anyway.
func (r *Decoder) Code(mark SyncMarker) int {
	r.Sync(mark)
	return r.Len()
}

// Reloc decodes a relocation of expected section k from the element
// bitstream and returns an index to the referenced element.
func (r *Decoder) Reloc(k RelocKind) Index {
	r.Sync(SyncUseReloc)
	return r.rawReloc(k, r.Len())
}

// String decodes and returns a string value from the element
// bitstream.
func (r *Decoder) String() string {
	r.Sync(SyncString)
	return r.common.StringIdx(r.Reloc(RelocString))
}

// Strings decodes and returns a variable-length slice of strings from
// the element bitstream.
func (r *Decoder) Strings() []string {
	res := make([]string, r.Len())
	for i := range res {
		res[i] = r.String()
	}
	return res
}

// Value decodes and returns a constant.Value from the element
// bitstream.
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

// PeekPkgPath returns the package path for the specified package
// index.
func (pr *PkgDecoder) PeekPkgPath(idx Index) string {
	var path string
	{
		r := pr.TempDecoder(RelocPkg, idx, SyncPkgDef)
		path = r.String()
		pr.RetireDecoder(&r)
	}
	if path == "" {
		path = pr.pkgPath
	}
	return path
}

// PeekObj returns the package path, object name, and CodeObj for the
// specified object index.
func (pr *PkgDecoder) PeekObj(idx Index) (string, string, CodeObj) {
	var ridx Index
	var name string
	var rcode int
	{
		r := pr.TempDecoder(RelocName, idx, SyncObject1)
		r.Sync(SyncSym)
		r.Sync(SyncPkg)
		ridx = r.Reloc(RelocPkg)
		name = r.String()
		rcode = r.Code(SyncCodeObj)
		pr.RetireDecoder(&r)
	}

	path := pr.PeekPkgPath(ridx)
	assert(name != "")

	tag := CodeObj(rcode)

	return path, name, tag
}
