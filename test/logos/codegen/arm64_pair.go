// asmcheck -goexperiment simd

package codegen

// Pairing checks for the 64-bit ARM target (SPEC 15.1, row 111).
// The 8-byte cases fuse in the SSA pair pass and pin upstream
// behavior, including the call, loop-join, and atomic barriers and the
// chase that must stay separate.
//
// The 16-byte cases reach the same pass through the simd/archsimd
// Load<T>Array and StoreArray intrinsics, which lower to FMOVQ loads
// and stores on the caller's own memory chain. Plain [16]byte values
// cannot serve: an array is not SSA-able, so every copy is a load
// followed by its own store, and the next load already sees that new
// memory instead of the one the pass needs it to share.
// A check sits on the source line of the first load or store of a pair,
// because the fused instruction takes the position of the lower-offset
// member and the second member emits no instruction of its own. A pair
// that must stay split pins both FMOVQ offsets and the absence of
// FLDPQ and FSTPQ.

import (
	"simd/archsimd"
	"sync/atomic"
	"unsafe"
)

type pairLoad08 struct {
	A int64
	B int64
}

// Adjacent 8-byte loads fuse into one paired load.
func loadPair08(p *pairLoad08) (int64, int64) {
	// arm64:"LDP \\(R"
	return p.A, p.B
}

type pairLoad504 struct {
	Pad [504]byte
	A   int64
	B   int64
}

// The pair at the top of the 64-bit range uses the lower offset.
func loadPair504(p *pairLoad504) (int64, int64) {
	// arm64:"LDP 504\\(" -"LDP 512\\("
	return p.A, p.B
}

type pairLoad512 struct {
	Pad [512]byte
	A   int64
	B   int64
}

// Past the top of the 64-bit range the pair still forms through the
// assembler fixup, unlike the rejected 16-byte pairs below.
func loadPair512(p *pairLoad512) (int64, int64) {
	// arm64:"LDP 512\\("
	return p.A, p.B
}

type pairChase struct {
	Next *pairChase
	V    int64
}

// A load that feeds the base of the next load stays separate.
func loadChase(n *pairChase) int64 {
	// arm64:-"LDP" "MOVD"
	m := n.Next
	return m.V
}

// Zero stores fuse into a paired store of the zero register.
func storeZero(p *pairLoad08) {
	// arm64:"STP \\(ZR, ZR\\), \\(R"
	p.A = 0
	p.B = 0
}

// A call between two candidate loads keeps two loads.
var pairSink func(int64)

func loadCallBarrier(p *pairLoad08) (int64, int64) {
	// arm64:-"LDP" "MOVD"
	a := p.A
	pairSink(a)
	b := p.B
	return a, b
}

// Candidates split across a loop join keep two loads.
func loadLoopBarrier(p *pairLoad08, i int, x int64) (int64, int64) {
	// arm64:-"LDP" "MOVD"
	a := p.A
	for i > 0 {
		x += int64(i)
		i--
	}
	b := p.B
	return a + x, b
}

var atomicCell atomic.Int64

// An atomic between two candidate stores keeps two stores.
func storeAtomicBarrier(p *pairLoad08, x, y int64) {
	// arm64:-"STP" "MOVD"
	// arm64/v8.1:-"STP" "MOVD"
	p.A = x
	// arm64/v8.1:"LDADDAL"
	atomicCell.Add(1)
	p.B = y
}

type pairVec016 struct {
	A [16]byte
	B [16]byte
}

// Adjacent 16-byte vector loads fuse into one FLDPQ and the adjacent
// stores into one FSTPQ, with no FMOVQ left behind.
func vecPair016(src, dst *pairVec016) {
	// arm64:"FLDPQ \\(R[0-9]+\\), \\(F[0-9]+, F[0-9]+\\)" -"FMOVQ"
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FLDPQ" -"FMOVQ"
	b := archsimd.LoadUint8x16Array(&src.B)
	// arm64:"FSTPQ \\(F[0-9]+, F[0-9]+\\), \\(R[0-9]+\\)" -"FMOVQ"
	a.StoreArray(&dst.A)
	// arm64:-"FSTPQ" -"FMOVQ"
	b.StoreArray(&dst.B)
}

type pairVec992 struct {
	Pad [992]byte
	A   [16]byte
	B   [16]byte
}

// A pair below the top of the 128-bit range keeps its lower offset.
func vecPair992(src, dst *pairVec992) {
	// arm64:"FLDPQ 992\\(R[0-9]+\\), \\(F[0-9]+, F[0-9]+\\)" -"FMOVQ"
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FLDPQ" -"FMOVQ"
	b := archsimd.LoadUint8x16Array(&src.B)
	// arm64:"FSTPQ \\(F[0-9]+, F[0-9]+\\), 992\\(R[0-9]+\\)" -"FMOVQ"
	a.StoreArray(&dst.A)
	// arm64:-"FSTPQ" -"FMOVQ"
	b.StoreArray(&dst.B)
}

type pairVec1008 struct {
	Pad [1008]byte
	A   [16]byte
	B   [16]byte
}

// The pair at the top of the 128-bit range uses the lower offset; its
// second member sits at 1024, which only ever follows a base of 1008.
func vecPair1008(src, dst *pairVec1008) {
	// arm64:"FLDPQ 1008\\(R[0-9]+\\), \\(F[0-9]+, F[0-9]+\\)" -"FMOVQ"
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FLDPQ" -"FMOVQ"
	b := archsimd.LoadUint8x16Array(&src.B)
	// arm64:"FSTPQ \\(F[0-9]+, F[0-9]+\\), 1008\\(R[0-9]+\\)" -"FMOVQ"
	a.StoreArray(&dst.A)
	// arm64:-"FSTPQ" -"FMOVQ"
	b.StoreArray(&dst.B)
}

type pairVec1024 struct {
	Pad [1024]byte
	A   [16]byte
	B   [16]byte
}

// A pair whose base is past the top of the 128-bit range stays two
// accesses, because the assembler cannot encode FLDPQ or FSTPQ there.
func vecSplit1024(src, dst *pairVec1024) {
	// arm64:-"FLDPQ" "FMOVQ 1024\\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FLDPQ" "FMOVQ 1040\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array(&src.B)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 1024\\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 1040\\(R[0-9]+\\)"
	b.StoreArray(&dst.B)
}

type pairVecRaw struct {
	Buf [32]byte
}

// Vector accesses 8 apart overlap and are 8-aligned but not
// 16-aligned, so neither the load pair nor the store pair forms.
func vecMisaligned(src, dst *pairVecRaw) {
	// arm64:-"FLDPQ" "FMOVQ \\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Pointer(&src.Buf[0])))
	// arm64:-"FLDPQ" "FMOVQ 8\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Pointer(&src.Buf[8])))
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray((*[16]byte)(unsafe.Pointer(&dst.Buf[0])))
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 8\\(R[0-9]+\\)"
	b.StoreArray((*[16]byte)(unsafe.Pointer(&dst.Buf[8])))
}

type pairVecPad8 struct {
	Pad [8]byte
	A   [16]byte
	B   [16]byte
}

// Adjacent vector accesses at 8 and 24 are 8-aligned but not
// 16-aligned, so no FLDPQ or FSTPQ offset can encode the base.
func vecMisaligned24(src, dst *pairVecPad8) {
	// arm64:-"FLDPQ" "FMOVQ 8\\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FLDPQ" "FMOVQ 24\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array(&src.B)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 8\\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 24\\(R[0-9]+\\)"
	b.StoreArray(&dst.B)
}

// A call between two vector loads keeps two loads: the call threads a
// new memory value, so the loads no longer share one.
func vecLoadCallBarrier(src *pairVec016, dst *[16]byte) {
	// arm64:-"FLDPQ" "FMOVQ \\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array(&src.A)
	pairSink(0)
	// arm64:-"FLDPQ" "FMOVQ 16\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array(&src.B)
	a.Add(b).StoreArray(dst)
}

// A call between two vector stores keeps two stores.
func vecStoreCallBarrier(src, dst *pairVec016) {
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	pairSink(0)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 16\\(R[0-9]+\\)"
	a.StoreArray(&dst.B)
}

// Vector loads split across a loop join keep two loads. Nothing in the
// loop touches memory, so only the block boundary keeps them apart.
func vecLoadJoinBarrier(src *pairVec016, dst *[16]byte, i, x int) int {
	// arm64:-"FLDPQ" "FMOVQ \\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array(&src.A)
	for i > 0 {
		x += i
		i--
	}
	// arm64:-"FLDPQ" "FMOVQ 16\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array(&src.B)
	a.Add(b).StoreArray(dst)
	return x
}

// An LSE atomic add between two vector stores keeps two stores. Below
// GOARM64=v8.1 the atomic sits behind a CPU feature branch, which is a
// block boundary of its own; v8.1 keeps it straight-line, so the atomic
// is the only barrier and the checks pin that spelling.
func vecStoreLSEAddBarrier(src, dst *pairVec016) {
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64/v8.1:"LDADDAL"
	atomicCell.Add(1)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, 16\\(R[0-9]+\\)"
	a.StoreArray(&dst.B)
}

// An LSE compare-and-swap between two vector stores keeps two stores.
func vecStoreLSECasBarrier(src, dst *pairVec016) {
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64/v8.1:"CASAL"
	atomicCell.CompareAndSwap(0, 1)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, 16\\(R[0-9]+\\)"
	a.StoreArray(&dst.B)
}
