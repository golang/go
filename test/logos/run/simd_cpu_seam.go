// run -goexperiment simd -godebug cpu.avx=off,cpu.avx2=off,cpu.avx512f=off

package main

import "simd/archsimd"

// M14-S2 seam check for simd_cpu.lgo: under the GODEBUG directive above
// the runtime must report every archsimd CPU feature the lowering
// guards on as absent. The guarded scalar branch of simd_cpu.lgo only
// runs when this holds.
//
// The cpu.avx and cpu.avx2 knobs exist only when GOAMD64 is below v3,
// and cpu.avx512f only below v4 (internal/cpu cpu_x86.go), so this seam
// needs a v1 or v2 build. A failure here is the knob missing or the
// directive not reaching the process, not a compiler defect.

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
