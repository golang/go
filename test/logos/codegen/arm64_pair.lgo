// asmcheck -goexperiment simd

package codegen

import (
	"simd/archsimd"
	"sync/atomic"
	"unsafe"
)

type pairLoad08 struct {
	A int64
	B int64
}

func loadPair08(p *pairLoad08) (int64, int64) {
	// arm64:"LDP \\(R[0-9]+\\), \\(R0, R1\\)"
	return p.A, p.B
}

type pairLoad504 struct {
	Pad [504]byte
	A   int64
	B   int64
}

func loadPair504(p *pairLoad504) (int64, int64) {
	// arm64:"LDP 504\\(" -"LDP 512\\("
	return p.A, p.B
}

type pairLoad512 struct {
	Pad [512]byte
	A   int64
	B   int64
}

func loadPair512(p *pairLoad512) (int64, int64) {
	// arm64:"LDP 512\\("
	return p.A, p.B
}

type pairChase struct {
	Next *pairChase
	V    int64
}

func loadChase(n *pairChase) int64 {
	// arm64:-"LDP" "MOVD"
	m := n.Next
	return m.V
}

func storeZero(p *pairLoad08) {
	// arm64:"STP \\(ZR, ZR\\), \\(R"
	p.A = 0
	p.B = 0
}

var pairSink func(int64)

func loadCallBarrier(p *pairLoad08) (int64, int64) {
	// arm64:-"LDP" "MOVD"
	a := p.A
	pairSink(a)
	b := p.B
	return a, b
}

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

func storeAtomicBarrier(p *pairLoad08, x, y int64) {
	// arm64:-"STP" "MOVD"
	// arm64/v8.1:-"STP" "MOVD"
	p.A = x
	// arm64/v8.1:"LDADDAL"
	atomicCell.Add(1)
	p.B = y
}

func loadAtomicBarrier(p *pairLoad08) (int64, int64) {
	// arm64:-"LDP" "MOVD"
	// arm64/v8.1:-"LDP" "MOVD"
	a := p.A
	atomicCell.Add(1)
	b := p.B
	return a, b
}

type pairVec016 struct {
	A [16]byte
	B [16]byte
}

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

type pairVecMis1000 struct {
	Pad [1000]byte
	A   [16]byte
	B   [16]byte
}

func vecMisaligned1008(src, dst *pairVecMis1000) {
	// arm64:-"FLDPQ" "FMOVQ 1000\\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FLDPQ" "FMOVQ 1016\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array(&src.B)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 1000\\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 1016\\(R[0-9]+\\)"
	b.StoreArray(&dst.B)
}

func vecLoadCallBarrier(src *pairVec016, dst *[16]byte) {
	// arm64:-"FLDPQ" "FMOVQ \\(R[0-9]+\\), F[0-9]+"
	a := archsimd.LoadUint8x16Array(&src.A)
	pairSink(0)
	// arm64:-"FLDPQ" "FMOVQ 16\\(R[0-9]+\\), F[0-9]+"
	b := archsimd.LoadUint8x16Array(&src.B)
	a.Add(b).StoreArray(dst)
}

func vecStoreCallBarrier(src, dst *pairVec016) {
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	pairSink(0)
	// arm64:-"FSTPQ" "FMOVQ F[0-9]+, 16\\(R[0-9]+\\)"
	a.StoreArray(&dst.B)
}

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

func vecStoreLSEAddBarrier(src, dst *pairVec016) {
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64/v8.1:"LDADDAL"
	atomicCell.Add(1)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, 16\\(R[0-9]+\\)"
	a.StoreArray(&dst.B)
}

func vecStoreLSECasBarrier(src, dst *pairVec016) {
	a := archsimd.LoadUint8x16Array(&src.A)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, \\(R[0-9]+\\)"
	a.StoreArray(&dst.A)
	// arm64/v8.1:"CASAL"
	atomicCell.CompareAndSwap(0, 1)
	// arm64/v8.1:-"FSTPQ" "FMOVQ F[0-9]+, 16\\(R[0-9]+\\)"
	a.StoreArray(&dst.B)
}
