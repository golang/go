// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

/*

Go Assembly for ARM64 Reference Manual

1. Alphabetical list of basic instructions
    // TODO

2. Alphabetical list of float-point instructions
    // TODO

3. Alphabetical list of SIMD instructions
    VADD: Add (scalar)
      VADD	<Vm>, <Vn>, <Vd>
        Add corresponding low 64-bit elements in <Vm> and <Vn>,
        place the result into low 64-bit element of <Vd>.

    VADD: Add (vector).
      VADD	<Vm>.T, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        8B, 16B, H4, H8, S2, S4, D2

    VADDP: Add Pairwise (vector)
      VADDP	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8, S2, S4, D2

    VADDV: Add across Vector.
      VADDV	<Vn>.<T>, Vd
        <T> Is an arrangement specifier and can have the following values:
        8B, 16B, H4, H8, S4

    VAND: Bitwise AND (vector)
      VAND	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16

    VCMEQ: Compare bitwise Equal (vector)
      VCMEQ	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8, S2, S4, D2

    VDUP: Duplicate vector element to vector or scalar.
      VDUP	<Vn>.<Ts>[index], <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        8B, 16B, H4, H8, S2, S4, D2
        <Ts> Is an element size specifier and can have the following values:
        B, H, S, D

    VEOR: Bitwise exclusive OR (vector, register)
      VEOR	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16

    VLD1: Load multiple single-element structures
      VLD1	(Rn), [<Vt>.<T>, <Vt2>.<T> ...]     // no offset
      VLD1.P	imm(Rn), [<Vt>.<T>, <Vt2>.<T> ...]  // immediate offset variant
      VLD1.P	(Rn)(Rm), [<Vt>.<T>, <Vt2>.<T> ...] // register offset variant
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8, S2, S4, D1, D2

    VMOV: move
      VMOV	<Vn>.<T>[index], Rd // Move vector element to general-purpose register.
        <T> Is a source width specifier and can have the following values:
        B, H, S (Wd)
        D (Xd)

      VMOV	Rn, <Vd>.<T> // Duplicate general-purpose register to vector.
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8, S2, S4 (Wn)
        D2 (Xn)

      VMOV	<Vn>.<T>, <Vd>.<T> // Move vector.
        <T> Is an arrangement specifier and can have the following values:
        B8, B16

      VMOV	Rn, <Vd>.<T>[index] // Move general-purpose register to a vector element.
        <T> Is a source width specifier and can have the following values:
        B, H, S (Wd)
        D (Xd)

      VMOV	<Vn>.<T>[index], Vn  // Move vector element to scalar.
        <T> Is an element size specifier and can have the following values:
        B, H, S, D

    VMOVI: Move Immediate (vector).
      VMOVI	$imm8, <Vd>.<T>
        <T> is an arrangement specifier and can have the following values:
        8B, 16B

    VMOVS: Load SIMD&FP Register (immediate offset). ARMv8: LDR (immediate, SIMD&FP)
      Store SIMD&FP register (immediate offset). ARMv8: STR (immediate, SIMD&FP)
      VMOVS	(Rn), Vn
      VMOVS.W	imm(Rn), Vn
      VMOVS.P	imm(Rn), Vn
      VMOVS	Vn, (Rn)
      VMOVS.W	Vn, imm(Rn)
      VMOVS.P	Vn, imm(Rn)

    VORR: Bitwise inclusive OR (vector, register)
      VORR	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16

    VREV32: Reverse elements in 32-bit words (vector).
      REV32 <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8

    VST1: Store multiple single-element structures
      VST1	[<Vt>.<T>, <Vt2>.<T> ...], (Rn)         // no offset
      VST1.P	[<Vt>.<T>, <Vt2>.<T> ...], imm(Rn)      // immediate offset variant
      VST1.P	[<Vt>.<T>, <Vt2>.<T> ...], (Rn)(Rm)     // register offset variant
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8, S2, S4, D1, D2

    VSUB: Sub (scalar)
      VSUB	<Vm>, <Vn>, <Vd>
        Subtract low 64-bit element in <Vm> from the correponding element in <Vn>,
        place the result into low 64-bit element of <Vd>.

    VUADDLV: Unsigned sum Long across Vector.
      VUADDLV	<Vn>.<T>, Vd
        <T> Is an arrangement specifier and can have the following values:
        8B, 16B, H4, H8, S4

4. Alphabetical list of cryptographic extension instructions

    SHA1C, SHA1M, SHA1P: SHA1 hash update.
      SHA1C	<Vm>.S4, Vn, Vd
      SHA1M	<Vm>.S4, Vn, Vd
      SHA1P	<Vm>.S4, Vn, Vd

    SHA1H: SHA1 fixed rotate.
      SHA1H	Vn, Vd

    SHA1SU0:   SHA1 schedule update 0.
    SHA256SU1: SHA256 schedule update 1.
      SHA1SU0	<Vm>.S4, <Vn>.S4, <Vd>.S4
      SHA256SU1	<Vm>.S4, <Vn>.S4, <Vd>.S4

    SHA1SU1:   SHA1 schedule update 1.
    SHA256SU0: SHA256 schedule update 0.
      SHA1SU1	<Vn>.S4, <Vd>.S4
      SHA256SU0	<Vn>.S4, <Vd>.S4

    SHA256H, SHA256H2: SHA256 hash update.
      SHA256H	<Vm>.S4, Vn, Vd
      SHA256H2	<Vm>.S4, Vn, Vd


*/
