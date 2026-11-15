// run -goexperiment simd -godebug cpu.avx=off,cpu.avx2=off,cpu.avx512f=off -goamd64=v1

package main

import "simd/archsimd"

const seamCause = " (needs GOAMD64 v1 or v2, and the GODEBUG directive in this process)"

func main() {
	if archsimd.X86.AVX() {
		panic("cpu.avx=off left AVX reported" + seamCause)
	}
	if archsimd.X86.AVX2() {
		panic("cpu.avx2=off left AVX2 reported" + seamCause)
	}
	if archsimd.X86.AVX512() {
		panic("cpu.avx512f=off left AVX512 reported" + seamCause)
	}
}
