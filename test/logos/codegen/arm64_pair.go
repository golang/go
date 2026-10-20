// asmcheck

package codegen

// Pairing checks for the 64-bit ARM target (SPEC 15.1, row 111).
// The 8-byte cases below fuse in the SSA pair pass and pin upstream
// behavior, including the call, loop-join, and atomic barriers and the
// chase that must stay separate. The 16-byte cases pin the observable
// end of the same pass: the loads exist at the listed offsets but stay
// separate, because every 16-byte temporary the ABI creates splits the
// shared memory value fusion needs. Fusion itself, with the 16-byte
// table entries and the encodable-range checks, is pinned by the SSA
// unit tests beside the pass.

import (
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

var atomicCell int64

// An atomic between two candidate stores keeps two stores.
func storeAtomicBarrier(p *pairLoad08, x, y int64) {
	// arm64:-"STP" "MOVD"
	p.A = x
	atomic.AddInt64(&atomicCell, 1)
	p.B = y
}

type pairVec016 struct {
	A [16]byte
	B [16]byte
}

// Adjacent 16-byte loads exist but stay separate end to end.
func loadVec016(p *pairVec016) ([16]byte, [16]byte) {
	// arm64:-"FLDPQ" -"FSTPQ" "FMOVQ"
	return p.A, p.B
}

type pairVec992 struct {
	Pad [992]byte
	A   [16]byte
	B   [16]byte
}

// Vector loads below the top of the range keep their offsets.
func loadVec992(p *pairVec992) ([16]byte, [16]byte) {
	// arm64:-"FLDPQ" -"FSTPQ" "FMOVQ 992\\(" "FMOVQ 1008\\("
	return p.A, p.B
}

type pairVec1008 struct {
	Pad [1008]byte
	A   [16]byte
	B   [16]byte
}

// Vector loads at the top of the range keep their offsets.
func loadVec1008(p *pairVec1008) ([16]byte, [16]byte) {
	// arm64:-"FLDPQ" -"FSTPQ" "FMOVQ 1008\\(" "FMOVQ 1024\\("
	return p.A, p.B
}

type pairVec1024 struct {
	Pad [1024]byte
	A   [16]byte
	B   [16]byte
}

// Vector loads past the top of the range keep their offsets.
func loadVec1024(p *pairVec1024) ([16]byte, [16]byte) {
	// arm64:-"FLDPQ" -"FSTPQ" "FMOVQ 1024\\(" "FMOVQ 1040\\("
	return p.A, p.B
}

type pairVecRaw struct {
	Buf [32]byte
}

// Vector loads 8 apart are not 16-byte adjacent.
func loadVecMisaligned(p *pairVecRaw) ([16]byte, [16]byte) {
	// arm64:-"FLDPQ" -"FSTPQ" "FMOVQ"
	a := *(*[16]byte)(unsafe.Pointer(&p.Buf[0]))
	b := *(*[16]byte)(unsafe.Pointer(&p.Buf[8]))
	return a, b
}

// Adjacent 16-byte stores exist but stay separate end to end.
func storeVec016(dst *pairVec016, a, b [16]byte) {
	// arm64:-"FLDPQ" -"FSTPQ" "FMOVQ"
	dst.A = a
	dst.B = b
}
