// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"cmp"
	"fmt"
	"maps"
	"simd/archsimd/_gen/sgutil"
	"slices"
	"strings"
)

// writeSIMDEnlargeTables writes the tables the compiler's "simd enlarge" pass
// uses to bypass Enlarge for operations that read only the low part of an
// input. It writes nothing if the architecture has no Enlarge.
//
// x.Enlarge() puts x in the low half of a vector twice as wide (zeroing the
// upper half). An operation with a narrow input (see [Operand.NarrowReg])
// reads only the low part of that input, so if that part lies within x, the
// operation can read x directly. For example, Int8x32.ExtendToInt32 reads only
// the low 128 bits of its input, so for an Int8x16 x, the pass rewrites
// x.Enlarge().ExtendToInt32() to take x as its input, which then lowers to a
// single VPMOVSXBD from x's register.
//
// The pass works on generic ops, before lowering, so the tables describe
// generic ops. This means the masked and merging forms of each machine op need
// nothing extra: they're formed by lowering rules from the unmasked op, which
// already has the bypassed input.
//
// After the pass, the op's input is narrower than its Go type. This is fine:
// lowering passes the input through to the machine op, and codegen names the
// input's register by the size the instruction reads (see simdNarrowArg).
// A load of a 128-bit x is still combined with the instruction, because the
// widening ExtendLo* conversions share these machine ops and give them rules
// that combine them with 128-bit loads.
func writeSIMDEnlargeTables(ops []Operation) *bytes.Buffer {
	// narrowArg identifies an op's narrow input: the index of the argument
	// and how many bits of it the op reads.
	type narrowArg struct{ arg, bits int }
	var enlarge []string
	narrow := make(map[narrowArg][]string) // -> generic op names
	seen := make(map[string]bool)
	for _, opr := range ops {
		if opr.NoGenericOps != nil && *opr.NoGenericOps == "true" {
			continue
		}
		if opr.SkipMaskedMethod() {
			continue
		}
		_, _, maskType, immType, gOp, _ := opr.shape()
		name := gOp.GenericName()
		if seen[name] {
			continue
		}
		seen[name] = true

		if gOp.Go == "Enlarge" {
			if maskType != NoMask || len(gOp.In) != 1 || len(gOp.Out) != 1 || !gOp.Out[0].NarrowReg() {
				panic(fmt.Errorf("simdgen: unexpected shape of Enlarge: %v", gOp))
			}
			enlarge = append(enlarge, name)
			continue
		}

		// Find if gOp has any narrow inputs. The generic op's
		// arguments are gOp.In, in order, with no immediates.
		arg := -1
		for i, in := range gOp.In {
			if in.Class != "vreg" || !in.NarrowReg() {
				continue
			}
			if arg >= 0 {
				// TODO: We could support this, but it's not
				// needed right now.
				panic(fmt.Errorf("simdgen: Enlarge bypass supports one narrow input, found more: %v", gOp))
			}
			arg = i
		}
		if arg < 0 {
			continue
		}
		if gOp.OperandOrder != nil || opr.implicitPredCount() > 0 {
			panic(fmt.Errorf("simdgen: Enlarge bypass needs generic op arguments in operand order: %v", gOp))
		}
		if immType != NoImm {
			// No operation with a narrow input takes an
			// immediate today. Supporting one requires
			// accounting for the immediate when numbering the
			// generic op's arguments.
			panic(fmt.Errorf("simdgen: Enlarge bypass does not support immediates: %v", gOp))
		}
		key := narrowArg{arg, gOp.In[arg].Bits.N()}
		narrow[key] = append(narrow[key], name)
	}
	if len(enlarge) == 0 {
		return nil
	}

	archInfo := CurrentArch()
	arch := archInfo.SIMDTag
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s\n\n", archInfo.GeneratedHeader)
	fmt.Fprintf(&buf, "package ssacompile\n\n")
	fmt.Fprintf(&buf, "import \"cmd/compile/internal/ssa/ssaop\"\n\n")

	caseList := func(names []string) string {
		slices.SortFunc(names, sgutil.CompareNatural)
		var b strings.Builder
		for i, n := range names {
			if i > 0 {
				b.WriteString(",\n\t\t")
			}
			b.WriteString("ssaop.Op" + n)
		}
		return b.String()
	}

	fmt.Fprintf(&buf, "// simd%sIsEnlarge reports whether op is an Enlarge generic op.\n", arch)
	fmt.Fprintf(&buf, "func simd%sIsEnlarge(op ssaop.Op) bool {\n", arch)
	fmt.Fprintf(&buf, "\tswitch op {\n\tcase %s:\n\t\treturn true\n\t}\n\treturn false\n}\n\n", caseList(enlarge))

	fmt.Fprintf(&buf, "// simd%sNarrowArg reports whether a generic op reads only the low part of\n", arch)
	fmt.Fprintf(&buf, "// one of its arguments. If so, it returns the index of that argument and how\n")
	fmt.Fprintf(&buf, "// many of its low bits the op reads. Otherwise, it returns -1, 0.\n")
	fmt.Fprintf(&buf, "func simd%sNarrowArg(op ssaop.Op) (arg int, bits int64) {\n", arch)
	fmt.Fprintf(&buf, "\tswitch op {\n")
	keys := slices.SortedFunc(maps.Keys(narrow), func(a, b narrowArg) int {
		return cmp.Or(cmp.Compare(a.arg, b.arg), cmp.Compare(a.bits, b.bits))
	})
	for _, k := range keys {
		fmt.Fprintf(&buf, "\tcase %s:\n\t\treturn %d, %d\n", caseList(narrow[k]), k.arg, k.bits)
	}
	fmt.Fprintf(&buf, "\t}\n\treturn -1, 0\n}\n")
	return &buf
}
