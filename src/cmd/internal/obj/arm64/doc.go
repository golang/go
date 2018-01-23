// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

/*

Go Assembly for ARM64 Reference Manual

1. Alphabetical list of basic instructions
    // TODO
    PRFM: Prefetch Memory (immediate)
     PRFM	imm(Rn), <prfop>
      prfop is the prefetch operation and can have the following values:
      PLDL1KEEP, PLDL1STRM, PLDL2KEEP, PLDL2STRM, PLDL3KEEP, PLDL3STRM,
      PLIL1KEEP, PLIL1STRM, PLIL2KEEP, PLIL2STRM, PLIL3KEEP, PLIL3STRM,
      PSTL1KEEP, PSTL1STRM, PSTL2KEEP, PSTL2STRM, PSTL3KEEP, PSTL3STRM.
     PRFM	imm(Rn), $imm
      $imm prefetch operation is encoded as an immediate.

    LDARB: Load-Acquire Register Byte
      LDARB	(<Rn>), <Rd>
        Loads a byte from memory, zero-extends it and writes it to Rd.

    LDARH: Load-Acquire Register Halfword
      LDARH	(<Rn>), <Rd>
        Loads a halfword from memory, zero-extends it and writes it to Rd.

    LDAXP: Load-Acquire Exclusive Pair of Registers
      LDAXP	(<Rn>), (<Rt1>, <Rt2>)
        Loads two 64-bit doublewords from memory, and writes them to Rt1 and Rt2.

    LDAXPW: Load-Acquire Exclusive Pair of Registers
      LDAXPW	(<Rn>), (<Rt1>, <Rt2>)
        Loads two 32-bit words from memory, and writes them to Rt1 and Rt2.

    LDXP: 64-bit Load Exclusive Pair of Registers
      LDXP	(<Rn>), (<Rt1>, <Rt2>)
        Loads two 64-bit doublewords from memory, and writes them to Rt1 and Rt2.

    LDXPW: 32-bit Load Exclusive Pair of Registers
      LDXPW	(<Rn>), (<Rt1>, <Rt2>)
        Loads two 32-bit words from memory, and writes them to Rt1 and Rt2.

    STLRB: Store-Release Register Byte
      STLRB	<Rd>, (<Rn>)
        Stores a byte from Rd to a memory location from Rn.

    STLRH: Store-Release Register Halfword
      STLRH	<Rd>, (<Rn>)
        Stores a halfword from Rd to a memory location from Rn.

    STLXP: 64-bit Store-Release Exclusive Pair of registers
      STLXP	(<Rt1>, <Rt2>), (<Rn>), <Rs>
        Stores two 64-bit doublewords from Rt1 and Rt2 to a memory location from Rn,
        and returns in Rs a status value of 0 if the store was successful, or of 1 if
        no store was performed.

    STLXPW: 32-bit Store-Release Exclusive Pair of registers
      STLXPW	(<Rt1>, <Rt2>), (<Rn>), <Rs>
        Stores two 32-bit words from Rt1 and Rt2 to a memory location from Rn, and
        returns in Rs a status value of 0 if the store was successful, or of 1 if no
        store was performed.

    STXP: 64-bit Store Exclusive Pair of registers
      STXP	(<Rt1>, <Rt2>), (<Rn>), <Rs>
        Stores two 64-bit doublewords from Rt1 and Rt2 to a memory location from Rn,
        and returns in Rs a status value of 0 if the store was successful, or of 1 if
        no store was performed.

    STXPW: 32-bit Store Exclusive Pair of registers
      STXPW	(<Rt1>, <Rt2>), (<Rn>), <Rs>
        Stores two 32-bit words from Rt1 and Rt2 to a memory location from Rn, and returns in
        a Rs a status value of 0 if the store was successful, or of 1 if no store was performed.

2. Alphabetical list of float-point instructions
    // TODO

    FMADDD: 64-bit floating-point fused Multiply-Add
      FMADDD	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>,
        adds the product to <Fa>, and writes the result to <Fd>.

    FMADDS: 32-bit floating-point fused Multiply-Add
      FMADDS	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>,
        adds the product to <Fa>, and writes the result to <Fd>.

    FMSUBD: 64-bit floating-point fused Multiply-Subtract
      FMSUBD	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>, negates the product,
        adds the product to <Fa>, and writes the result to <Fd>.

    FMSUBS: 32-bit floating-point fused Multiply-Subtract
      FMSUBS	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>, negates the product,
        adds the product to <Fa>, and writes the result to <Fd>.

    FNMADDD: 64-bit floating-point negated fused Multiply-Add
      FNMADDD	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>, negates the product,
        subtracts the value of <Fa>, and writes the result to <Fd>.

    FNMADDS: 32-bit floating-point negated fused Multiply-Add
      FNMADDS	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>, negates the product,
        subtracts the value of <Fa>, and writes the result to <Fd>.

    FNMSUBD: 64-bit floating-point negated fused Multiply-Subtract
      FNMSUBD	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>,
        subtracts the value of <Fa>, and writes the result to <Fd>.

    FNMSUBS: 32-bit floating-point negated fused Multiply-Subtract
      FNMSUBS	<Fm>, <Fa>, <Fn>, <Fd>
        Multiplies the values of <Fm> and <Fn>,
        subtracts the value of <Fa>, and writes the result to <Fd>.

3. Alphabetical list of SIMD instructions
    VADD: Add (scalar)
      VADD	<Vm>, <Vn>, <Vd>
        Add corresponding low 64-bit elements in <Vm> and <Vn>,
        place the result into low 64-bit element of <Vd>.

    VADD: Add (vector).
      VADD	<Vm>.T, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        B8, B16, H4, H8, S2, S4, D2

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

    VFMLA: Floating-point fused Multiply-Add to accumulator (vector)
      VFMLA	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        S2, S4, D2

    VFMLS: Floating-point fused Multiply-Subtract from accumulator (vector)
      VFMLS	<Vm>.<T>, <Vn>.<T>, <Vd>.<T>
        <T> Is an arrangement specifier and can have the following values:
        S2, S4, D2

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

      VMOV	<Vn>.<T>[index], <Vd>.<T>[index] // Move vector element to another vector element.
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
        Subtract low 64-bit element in <Vm> from the corresponding element in <Vn>,
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
