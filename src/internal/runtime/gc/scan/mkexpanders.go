// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a fork of mkasm.go, instead of generating
// assembly code, this file generates Go code that uses
// the simd package.

//go:build ignore

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"log"
	"os"
	"slices"
	"strconv"
	"strings"
	"text/template"
	"unsafe"

	"internal/runtime/gc"
)

var simdTemplate = template.Must(template.New("template").Parse(`
{{- define "header"}}
// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package scan

import (
	"simd"
	"unsafe"
)
{{- end}}
{{- define "expandersList"}}
var gcExpandersAVX512 = [{{- len .}}]func(unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8){
{{- range .}}
	{{.}},
{{- end}}
}
{{- end}}

{{- define "expanderData"}}
var {{.Name}} = [8]uint64{
{{.Vals}}
}
{{- end}}

{{- define "expander"}}
func {{.Name}}(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	{{- .BodyLoadString }}
	{{- .BodyString }}
}
{{- end}}
`))

// expanderData is global data used by the expanders.
// They will be generated as global arrays.
type expanderData struct {
	Name string // Name of the global array
	Vals string // The values of the arrays, should already be formatted.
}

// expander is the expander function, it only operates on 3 kinds of values:
//
//	uint8x64, mask8x64, uint64.
//
// And a limited set of operations.
type expander struct {
	Name        string // The name of the expander function
	BodyLoad    strings.Builder
	Body        strings.Builder // The actual expand computations, after loads.
	data        []expanderData
	dataByVals  map[string]string
	uint8x64Cnt int
	mask8x64Cnt int
	uint64Cnt   int
}

// Used by text/template.
// This is needed because tex/template cannot call pointer receiver methods.
func (e expander) BodyLoadString() string {
	return e.BodyLoad.String()
}

func (e expander) BodyString() string {
	return e.Body.String()
}

// mat8x8 is an 8x8 bit matrix.
type mat8x8 struct {
	mat [8]uint8
}

func matGroupToVec(mats *[8]mat8x8) [8]uint64 {
	var out [8]uint64
	for i, mat := range mats {
		for j, row := range mat.mat {
			// For some reason, Intel flips the rows.
			out[i] |= uint64(row) << ((7 - j) * 8)
		}
	}
	return out
}

func (fn *expander) newVec() string {
	v := fmt.Sprintf("v%d", fn.uint8x64Cnt)
	fn.uint8x64Cnt++
	return v
}

func (fn *expander) newMask() string {
	v := fmt.Sprintf("m%d", fn.mask8x64Cnt)
	fn.mask8x64Cnt++
	return v
}

func (fn *expander) newU() string {
	v := fmt.Sprintf("u%d", fn.uint64Cnt)
	fn.uint64Cnt++
	return v
}

// expandIdentity implements 1x expansion (that is, no expansion).
func (fn *expander) expandIdentity() {
	fn.Body.WriteString(`
	x := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	y := simd.LoadUint64x8((*[8]uint64)(unsafe.Pointer(uintptr(src)+64))).AsUint8x64()
	return x.AsUint64x8(), y.AsUint64x8()`)
}

func (fn *expander) loadSrcAsUint8x64() string {
	v := fn.newVec()
	fn.BodyLoad.WriteString(fmt.Sprintf("%s := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()\n", v))
	return v
}

func (fn *expander) loadGlobalArrAsUint8x64(arrName string) string {
	v := fn.newVec()
	fn.BodyLoad.WriteString(fmt.Sprintf("%s := simd.LoadUint64x8(&%s).AsUint8x64()\n", v, arrName))
	return v
}

func (fn *expander) permuteUint8x64(data, indices string) string {
	v := fn.newVec()
	fn.Body.WriteString(fmt.Sprintf("%s := %s.Permute(%s)\n", v, data, indices))
	return v
}

func (fn *expander) permute2Uint8x64(x, y, indices string) string {
	v := fn.newVec()
	fn.Body.WriteString(fmt.Sprintf("%s := %s.ConcatPermute(%s, %s)\n", v, x, y, indices))
	return v
}

func (fn *expander) permuteMaskedUint8x64(data, indices, mask string) string {
	v := fn.newVec()
	fn.Body.WriteString(fmt.Sprintf("%s := %s.Permute(%s).Masked(%s)\n", v, data, indices, mask))
	return v
}

func (fn *expander) permute2MaskedUint8x64(x, y, indices, mask string) string {
	v := fn.newVec()
	fn.Body.WriteString(fmt.Sprintf("%s := %s.ConcatPermute(%s, %s).Masked(%s)\n", v, x, y, indices, mask))
	return v
}

func (fn *expander) galoisFieldAffineTransformUint8x64(data, matrix string) string {
	v := fn.newVec()
	fn.Body.WriteString(fmt.Sprintf("%s := %s.GaloisFieldAffineTransform(%s.AsUint64x8(), 0)\n", v, data, matrix))
	return v
}

func (fn *expander) returns(x, y string) {
	fn.Body.WriteString(fmt.Sprintf("return %s.AsUint64x8(), %s.AsUint64x8()", x, y))
}

func uint8x64Data(data [64]uint8) string {
	res := ""
	for i := range 8 {
		ptr64 := (*uint64)(unsafe.Pointer(&data[i*8]))
		res += fmt.Sprintf("%#016x,", *ptr64)
		if i == 3 {
			res += "\n"
		}
	}
	return res
}

func uint64x8Data(data [8]uint64) string {
	res := ""
	for i := range 8 {
		res += fmt.Sprintf("%#016x,", data[i])
		if i == 3 {
			res += "\n"
		}
	}
	return res
}

func (fn *expander) loadGlobalUint8x64(name string, data [64]uint8) string {
	val := uint8x64Data(data)
	if n, ok := fn.dataByVals[val]; !ok {
		fullName := fmt.Sprintf("%s_%s", fn.Name, name)
		fn.data = append(fn.data, expanderData{fullName, val})
		v := fn.loadGlobalArrAsUint8x64(fullName)
		fn.dataByVals[val] = v
		return v
	} else {
		return n
	}
}

func (fn *expander) loadGlobalUint64x8(name string, data [8]uint64) string {
	val := uint64x8Data(data)
	if n, ok := fn.dataByVals[val]; !ok {
		fullName := fmt.Sprintf("%s_%s", fn.Name, name)
		fn.data = append(fn.data, expanderData{fullName, val})
		v := fn.loadGlobalArrAsUint8x64(fullName)
		fn.dataByVals[val] = v
		return v
	} else {
		return n
	}
}

func (fn *expander) mask8x64FromBits(data uint64) string {
	v1 := fn.newU()
	v2 := fn.newMask()
	fn.Body.WriteString(fmt.Sprintf("%s := uint64(%#x)\n%s := simd.Mask8x64FromBits(%s)\n",
		v1, data, v2, v1))
	return v2
}

func (fn *expander) orUint8x64(x, y string) string {
	v := fn.newVec()
	fn.Body.WriteString(fmt.Sprintf("%s := %s.Or(%s)\n", v, x, y))
	return v
}

func main() {
	generate("expanders_amd64.go", genExpanders)
}

func generate(fileName string, genFunc func(*bytes.Buffer)) {
	var buf bytes.Buffer
	genFunc(&buf)
	f, err := os.Create(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	b, err := format.Source(buf.Bytes())
	if err != nil {
		log.Printf(string(buf.Bytes()))
		log.Fatal(err)
	}
	_, err = f.Write(b)
	if err != nil {
		log.Fatal(err)
	}
}

func genExpanders(buffer *bytes.Buffer) {
	if err := simdTemplate.ExecuteTemplate(buffer, "header", nil); err != nil {
		panic(fmt.Errorf("failed to execute header template: %w", err))
	}
	gcExpandersAVX512 := make([]expander, len(gc.SizeClassToSize))
	for sc, ob := range gc.SizeClassToSize {
		if gc.SizeClassToNPages[sc] != 1 {
			// These functions all produce a bitmap that covers exactly one
			// page.
			continue
		}
		if ob > gc.MinSizeForMallocHeader {
			// This size class is too big to have a packed pointer/scalar bitmap.
			break
		}

		xf := int(ob) / 8
		log.Printf("size class %d bytes, expansion %dx", ob, xf)

		fn := expander{Name: fmt.Sprintf("expandAVX512_%d", xf), dataByVals: make(map[string]string)}

		if xf == 1 {
			fn.expandIdentity()
		} else {
			ok := gfExpander(xf, &fn)
			if !ok {
				log.Printf("failed to generate expander for size class %d", sc)
			}
		}
		gcExpandersAVX512[sc] = fn
	}
	// Fill in the expanders data first
	eld := make([]string, len(gcExpandersAVX512))
	for i, gce := range gcExpandersAVX512 {
		if gce.Name == "" {
			eld[i] = "nil"
		} else {
			eld[i] = gce.Name
		}
	}
	if err := simdTemplate.ExecuteTemplate(buffer, "expandersList", eld); err != nil {
		panic(fmt.Errorf("failed to execute expandersList template: %w", err))
	}
	// List out the expander functions and their data
	for _, gce := range gcExpandersAVX512 {
		if gce.Name == "" {
			continue
		}
		for _, data := range gce.data {
			if err := simdTemplate.ExecuteTemplate(buffer, "expanderData", data); err != nil {
				panic(fmt.Errorf("failed to execute expanderData template: %w", err))
			}
		}
		if err := simdTemplate.ExecuteTemplate(buffer, "expander", gce); err != nil {
			panic(fmt.Errorf("failed to execute expander template: %w", err))
		}
	}
}

// gfExpander produces a function that expands each bit in an input bitmap into
// f consecutive bits in an output bitmap.
//
// The input is
//
//	*[8]uint64 = A pointer to floor(1024/f) bits (f >= 2, so at most 512 bits)
//
// The output is
//
//	[64]uint8  = The bottom 512 bits of the expanded bitmap
//	[64]uint8  = The top 512 bits of the expanded bitmap
func gfExpander(f int, fn *expander) bool {
	// TODO(austin): For powers of 2 >= 8, we can use mask expansion ops to make this much simpler.

	// TODO(austin): For f >= 8, I suspect there are better ways to do this.
	//
	// For example, we could use a mask expansion to get a full byte for each
	// input bit, and separately create the bytes that blend adjacent bits, then
	// shuffle those bytes together. Certainly for f >= 16 this makes sense
	// because each of those bytes will be used, possibly more than once.

	objBits := fn.loadSrcAsUint8x64()

	type term struct {
		iByte, oByte int
		mat          mat8x8
	}
	var terms []term

	// Iterate over all output bytes and construct the 8x8 GF2 matrix to compute
	// the output byte from the appropriate input byte. Gather all of these into
	// "terms".
	for oByte := 0; oByte < 1024/8; oByte++ {
		var byteMat mat8x8
		iByte := -1
		for oBit := oByte * 8; oBit < oByte*8+8; oBit++ {
			iBit := oBit / f
			if iByte == -1 {
				iByte = iBit / 8
			} else if iByte != iBit/8 {
				log.Printf("output byte %d straddles input bytes %d and %d", oByte, iByte, iBit/8)
				return false
			}
			// One way to view this is that the i'th row of the matrix will be
			// ANDed with the input byte, and the parity of the result will set
			// the i'th bit in the output. We use a simple 1 bit mask, so the
			// parity is irrelevant beyond selecting out that one bit.
			byteMat.mat[oBit%8] = 1 << (iBit % 8)
		}
		terms = append(terms, term{iByte, oByte, byteMat})
	}

	if false {
		// Print input byte -> output byte as a matrix
		maxIByte, maxOByte := 0, 0
		for _, term := range terms {
			maxIByte = max(maxIByte, term.iByte)
			maxOByte = max(maxOByte, term.oByte)
		}
		iToO := make([][]rune, maxIByte+1)
		for i := range iToO {
			iToO[i] = make([]rune, maxOByte+1)
		}
		matMap := make(map[mat8x8]int)
		for _, term := range terms {
			i, ok := matMap[term.mat]
			if !ok {
				i = len(matMap)
				matMap[term.mat] = i
			}
			iToO[term.iByte][term.oByte] = 'A' + rune(i)
		}
		for o := range maxOByte + 1 {
			fmt.Printf("%d", o)
			for i := range maxIByte + 1 {
				fmt.Printf(",")
				if mat := iToO[i][o]; mat != 0 {
					fmt.Printf("%c", mat)
				}
			}
			fmt.Println()
		}
	}

	// In hardware, each (8 byte) matrix applies to 8 bytes of data in parallel,
	// and we get to operate on up to 8 matrixes in parallel (or 64 values). That is:
	//
	//  abcdefgh ijklmnop qrstuvwx yzABCDEF GHIJKLMN OPQRSTUV WXYZ0123 456789_+
	//    mat0     mat1     mat2     mat3     mat4     mat5     mat6     mat7

	// Group the terms by matrix, but limit each group to 8 terms.
	const termsPerGroup = 8       // Number of terms we can multiply by the same matrix.
	const groupsPerSuperGroup = 8 // Number of matrixes we can fit in a vector.

	matMap := make(map[mat8x8]int)
	allMats := make(map[mat8x8]bool)
	var termGroups [][]term
	for _, term := range terms {
		allMats[term.mat] = true

		i, ok := matMap[term.mat]
		if ok && f > groupsPerSuperGroup {
			// The output is ultimately produced in two [64]uint8 registers.
			// Getting every byte in the right place of each of these requires a
			// final permutation that often requires more than one source.
			//
			// Up to 8x expansion, we can get a really nice grouping so we can use
			// the same 8 matrix vector several times, without producing
			// permutations that require more than two sources.
			//
			// Above 8x, however, we can't get nice matrixes anyway, so we
			// instead prefer reducing the complexity of the permutations we
			// need to produce the final outputs. To do this, avoid grouping
			// together terms that are split across the two registers.
			outRegister := termGroups[i][0].oByte / 64
			if term.oByte/64 != outRegister {
				ok = false
			}
		}
		if !ok {
			// Start a new term group.
			i = len(termGroups)
			matMap[term.mat] = i
			termGroups = append(termGroups, nil)
		}

		termGroups[i] = append(termGroups[i], term)

		if len(termGroups[i]) == termsPerGroup {
			// This term group is full.
			delete(matMap, term.mat)
		}
	}

	for i, termGroup := range termGroups {
		log.Printf("term group %d:", i)
		for _, term := range termGroup {
			log.Printf("  %+v", term)
		}
	}

	// We can do 8 matrix multiplies in parallel, which is 8 term groups. Pack
	// as many term groups as we can into each super-group to minimize the
	// number of matrix multiplies.
	//
	// Ideally, we use the same matrix in each super-group, which might mean
	// doing fewer than 8 multiplies at a time. That's fine because it never
	// increases the total number of matrix multiplies.
	//
	// TODO: Packing the matrixes less densely may let us use more broadcast
	// loads instead of general permutations, though. That replaces a load of
	// the permutation with a load of the matrix, but is probably still slightly
	// better.
	var sgSize, nSuperGroups int
	oneMatVec := f <= groupsPerSuperGroup
	if oneMatVec {
		// We can use the same matrix in each multiply by doing sgSize
		// multiplies at a time.
		sgSize = groupsPerSuperGroup / len(allMats) * len(allMats)
		nSuperGroups = (len(termGroups) + sgSize - 1) / sgSize
	} else {
		// We can't use the same matrix for each multiply. Just do as many at a
		// time as we can.
		//
		// TODO: This is going to produce several distinct matrixes, when we
		// probably only need two. Be smarter about how we create super-groups
		// in this case. Maybe we build up an array of super-groups and then the
		// loop below just turns them into ops?
		sgSize = 8
		nSuperGroups = (len(termGroups) + groupsPerSuperGroup - 1) / groupsPerSuperGroup
	}

	// Construct each super-group.
	var matGroup [8]mat8x8
	var matMuls []string
	var perm [128]int
	for sgi := range nSuperGroups {
		var iperm [64]uint8
		for i := range iperm {
			iperm[i] = 0xff // "Don't care"
		}
		// Pick off sgSize term groups.
		superGroup := termGroups[:min(len(termGroups), sgSize)]
		termGroups = termGroups[len(superGroup):]
		// Build the matrix and permutations for this super-group.
		var thisMatGroup [8]mat8x8
		for i, termGroup := range superGroup {
			// All terms in this group have the same matrix. Pick one.
			thisMatGroup[i] = termGroup[0].mat
			for j, term := range termGroup {
				// Build the input permutation.
				iperm[i*termsPerGroup+j] = uint8(term.iByte)
				// Build the output permutation.
				perm[term.oByte] = sgi*groupsPerSuperGroup*termsPerGroup + i*termsPerGroup + j
			}
		}
		log.Printf("input permutation %d: %v", sgi, iperm)

		// Check that we're not making more distinct matrixes than expected.
		if oneMatVec {
			if sgi == 0 {
				matGroup = thisMatGroup
			} else if matGroup != thisMatGroup {
				log.Printf("super-groups have different matrixes:\n%+v\n%+v", matGroup, thisMatGroup)
				return false
			}
		}

		// Emit matrix op.
		matConst :=
			fn.loadGlobalUint64x8(fmt.Sprintf("mat%d", sgi),
				matGroupToVec(&thisMatGroup))
		inShufConst :=
			fn.loadGlobalUint8x64(fmt.Sprintf("inShuf%d", sgi),
				iperm)
		inOp := fn.permuteUint8x64(objBits, inShufConst)
		matMul := fn.galoisFieldAffineTransformUint8x64(inOp, matConst)
		matMuls = append(matMuls, matMul)
	}

	log.Printf("output permutation: %v", perm)

	outLo, ok := genShuffle(fn, "outShufLo", (*[64]int)(perm[:64]), matMuls...)
	if !ok {
		log.Printf("bad number of inputs to final shuffle: %d != 1, 2, or 4", len(matMuls))
		return false
	}
	outHi, ok := genShuffle(fn, "outShufHi", (*[64]int)(perm[64:]), matMuls...)
	if !ok {
		log.Printf("bad number of inputs to final shuffle: %d != 1, 2, or 4", len(matMuls))
		return false
	}
	fn.returns(outLo, outHi)

	return true
}

func genShuffle(fn *expander, name string, perm *[64]int, args ...string) (string, bool) {
	// Construct flattened permutation.
	var vperm [64]byte

	// Get the inputs used by this permutation.
	var inputs []int
	for i, src := range perm {
		inputIdx := slices.Index(inputs, src/64)
		if inputIdx == -1 {
			inputIdx = len(inputs)
			inputs = append(inputs, src/64)
		}
		vperm[i] = byte(src%64 | (inputIdx << 6))
	}

	// Emit instructions for easy cases.
	switch len(inputs) {
	case 1:
		constOp := fn.loadGlobalUint8x64(name, vperm)
		return fn.permuteUint8x64(args[inputs[0]], constOp), true
	case 2:
		constOp := fn.loadGlobalUint8x64(name, vperm)
		return fn.permute2Uint8x64(args[inputs[0]], args[inputs[1]], constOp), true
	}

	// Harder case, we need to shuffle in from up to 2 more tables.
	//
	// Perform two shuffles. One shuffle will get its data from the first
	// two inputs, the other shuffle will get its data from the other one
	// or two inputs. All values they don't care each don't care about will
	// be zeroed.
	var vperms [2][64]byte
	var masks [2]uint64
	for j, idx := range vperm {
		for i := range vperms {
			vperms[i][j] = 0xff // "Don't care"
		}
		if idx == 0xff {
			continue
		}
		vperms[idx/128][j] = idx % 128
		masks[idx/128] |= uint64(1) << j
	}

	// Validate that the masks are fully disjoint.
	if masks[0]^masks[1] != ^uint64(0) {
		panic("bad shuffle!")
	}

	// Generate constants.
	constOps := make([]string, len(vperms))
	for i, v := range vperms {
		constOps[i] = fn.loadGlobalUint8x64(name+strconv.Itoa(i), v)
	}

	// Generate shuffles.
	switch len(inputs) {
	case 3:
		r0 := fn.permute2MaskedUint8x64(args[inputs[0]], args[inputs[1]], constOps[0], fn.mask8x64FromBits(masks[0]))
		r1 := fn.permuteMaskedUint8x64(args[inputs[2]], constOps[1], fn.mask8x64FromBits(masks[1]))
		return fn.orUint8x64(r0, r1), true
	case 4:
		r0 := fn.permute2MaskedUint8x64(args[inputs[0]], args[inputs[1]], constOps[0], fn.mask8x64FromBits(masks[0]))
		r1 := fn.permute2MaskedUint8x64(args[inputs[2]], args[inputs[3]], constOps[1], fn.mask8x64FromBits(masks[1]))
		return fn.orUint8x64(r0, r1), true
	}

	// Too many inputs. To support more, we'd need to separate tables much earlier.
	// Right now all the indices fit in a byte, but with >4 inputs they might not (>256 bytes).
	return args[0], false
}
