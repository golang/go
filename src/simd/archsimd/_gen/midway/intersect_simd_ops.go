// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"simd/archsimd/_gen/sgutil"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/specdoc"
	"simd/archsimd/_gen/specgen"
)

type MethodSet map[string]*ast.FuncDecl
type TypeMethods map[string]MethodSet

// clmulNote is the implementation note for the CarrylessMul methods.
const clmulNote = `Performance: If the CPU has no carryless multiply instruction at the vector
size in use, this is implemented with narrower instructions, or with
software if it has none at all. That is slower than the instruction, and
can be the case even when [Emulated] returns false.`

// implementationNotes maps from a method name to an implementation note to add
// to the doc comment of that method of the portable types. The spec provides
// the rest of the doc comment, so each of these must start with an
// implementation note prefix, such as "Performance:".
var implementationNotes = map[string]string{
	"CarrylessMulEven":    clmulNote,
	"CarrylessMulOdd":     clmulNote,
	"CarrylessMulEvenOdd": clmulNote,
	"CarrylessMulOddEven": clmulNote,
}

var genFlags = gentools.RegisterFlags(nil)
var verbose = flag.Bool("v", false, "Be much chattier about processing")

type ArchAndFiles struct {
	arch  string
	files []string
}

type TypeMethod struct {
	t, m string
}

type whyMissing struct {
	wasm128, arm128, amd128, amd256, amd512 bool
}

func (w whyMissing) String() string {
	why := ""
	if w.wasm128 {
		why += " wasm"
	}
	if w.arm128 {
		why += " neon"
	}
	if w.amd128 {
		why += " avx"
	}
	if w.amd256 {
		why += " avx2"
	}
	if w.amd512 {
		why += " avx512"
	}
	return why[1:]
}

func combine(arch, typ string) string {
	return arch + "-" + typ
}

// requiredCPUFeature returns the CPU feature that doc's implementation notes
// say is required to call the documented method, or "" if there is none. This
// is the "CPU Feature:" clause of a note like
//
//	Asm: VPADDD, CPU Feature: AVX
//
// A method that uses an instruction only if some further CPU feature is
// available, and is emulated otherwise, names that feature in its "Asm:" clause
// instead, like
//
//	Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
//
// Such a method is safe to call without that further feature, so it doesn't
// count as required.
func requiredCPUFeature(doc string) string {
	_, feature, ok := strings.Cut(doc, "CPU Feature:")
	if !ok {
		return ""
	}
	feature, _, _ = strings.Cut(feature, "\n")
	return strings.TrimSpace(feature)
}

var files gentools.Files

func main() {
	minorProblem := false

	flag.Parse()

	specDir := specgen.MustFindSpecDir(genFlags.GOROOT)
	specIdx, err := specgen.LoadIndex(specDir, nil)
	if err != nil {
		log.Fatalf("loading spec: %v", err)
	}
	files.AddPostProcessor(specdoc.Filler(specIdx, specdoc.Options{
		AllowDocRewrite: true,
	}))

	defer files.FlushOrExit()

	pv := func(f string, s ...any) {
		if *verbose {
			fmt.Fprintf(os.Stderr, f, s...)
		}
	}
	pw := func(f string, s ...any) {
		minorProblem = true
		fmt.Fprintf(os.Stderr, f, s...)
	}

	archSimdPath := filepath.Join(genFlags.GOROOT, "src", "simd", "archsimd")

	// Hardcoded list of files. These are the files with generally
	// available API, since the simd package is built from that alone:
	// deprecated and other simd-only files are left out, and checkSpec
	// below rejects any experimental methods in these files.
	amd64Files := []string{"ops_amd64.go", "compare_gen_amd64.go", "types_amd64.go",
		"other_gen_amd64.go", "maskmerge_gen_amd64.go",
		"clmul_amd64.go", "slice_gen_amd64.go", "slicepart_amd64.go",
		"slicepart_128.go", "ops_emulated_amd64.go", "maskrange_amd64.go",
		"converts_emulated_amd64.go",
		"converts_gen_amd64.go"}
	wasmFiles := []string{"ops_wasm.go", "types_wasm.go", "slicepart_wasm.go",
		"slicepart_128.go", "ops_emulated_wasm.go", "maskrange_128.go", "other_gen_wasm.go",
		"converts_emulated_wasm.go"}
	neonFiles := []string{"clmul_arm64.go", "compare_gen_arm64.go",
		"maskmerge_gen_arm64.go", "ops_arm64.go", "slicepart_128.go",
		"ops_internal_arm64.go", "other_gen_arm64.go", "slice_gen_arm64.go",
		"slicepart_arm64.go", "types_arm64.go", "ops_emulated_arm64.go", "maskrange_128.go"}

	emulatedFile := genFlags.InputPath("simd/simd_emulated.go")

	archAndFiles := []ArchAndFiles{
		ArchAndFiles{"wasm", wasmFiles},
		ArchAndFiles{"amd64", amd64Files},
		ArchAndFiles{"arm64", neonFiles},
	}

	// Categories based on bit size
	// 128-bit map: ElementType -> TypeName
	map128 := map[string]string{
		"Int8":    "Int8x16",
		"Int16":   "Int16x8",
		"Int32":   "Int32x4",
		"Int64":   "Int64x2",
		"Uint8":   "Uint8x16",
		"Uint16":  "Uint16x8",
		"Uint32":  "Uint32x4",
		"Uint64":  "Uint64x2",
		"Float32": "Float32x4",
		"Float64": "Float64x2",
		"Mask8":   "Mask8x16",
		"Mask16":  "Mask16x8",
		"Mask32":  "Mask32x4",
		"Mask64":  "Mask64x2",
	}

	// 256-bit map: ElementType -> TypeName
	map256 := map[string]string{
		"Int8":    "Int8x32",
		"Int16":   "Int16x16",
		"Int32":   "Int32x8",
		"Int64":   "Int64x4",
		"Uint8":   "Uint8x32",
		"Uint16":  "Uint16x16",
		"Uint32":  "Uint32x8",
		"Uint64":  "Uint64x4",
		"Float32": "Float32x8",
		"Float64": "Float64x4",
		"Mask8":   "Mask8x32",
		"Mask16":  "Mask16x16",
		"Mask32":  "Mask32x8",
		"Mask64":  "Mask64x4",
	}

	map512 := map[string]string{
		"Int8":    "Int8x64",
		"Int16":   "Int16x32",
		"Int32":   "Int32x16",
		"Int64":   "Int64x8",
		"Uint8":   "Uint8x64",
		"Uint16":  "Uint16x32",
		"Uint32":  "Uint32x16",
		"Uint64":  "Uint64x8",
		"Float32": "Float32x16",
		"Float64": "Float64x8",
		"Mask8":   "Mask8x64",
		"Mask16":  "Mask16x32",
		"Mask32":  "Mask32x16",
		"Mask64":  "Mask64x8",
	}

	// sizeForType translates types of the form "Int32x16" to their size in bits.
	// i.e. it does arithmetic.
	sizeForType := make(map[string]int)

	methodsByType := make(TypeMethods)

	allMethodNames := make(map[string]bool)

	missing := make(map[string]whyMissing)

	fset := token.NewFileSet()

	// knownReceivers translates types of the form "Int32x16" to "Int32s"
	knownReceivers := make(map[string]string)
	for k, v := range map128 {
		knownReceivers[v] = k + "s"
		sizeForType[v] = 128
	}
	for k, v := range map256 {
		knownReceivers[v] = k + "s"
		sizeForType[v] = 256
	}
	for k, v := range map512 {
		knownReceivers[v] = k + "s"
		sizeForType[v] = 512
	}

	receiver := func(funcDecl *ast.FuncDecl) string {
		if funcDecl.Recv == nil {
			return ""
		}
		recvType := ""
		for _, field := range funcDecl.Recv.List {
			// We assume single receiver
			if ident, ok := field.Type.(*ast.Ident); ok {
				recvType = ident.Name
			} else if star, ok := field.Type.(*ast.StarExpr); ok {
				if ident, ok := star.X.(*ast.Ident); ok {
					recvType = ident.Name
				}
			}
		}
		return recvType
	}

	// emulated is used for error checking; every simd method should have an emulation,
	// and there should not be any emulations that are not simd methods.
	emulated := make(map[TypeMethod]bool)
	f, err := parser.ParseFile(fset, emulatedFile, nil, parser.ParseComments)
	if err != nil {
		log.Fatalf("Failed to parse %s: %v", emulatedFile, err)
	}

	for _, decl := range f.Decls {
		if funcDecl, ok := decl.(*ast.FuncDecl); ok {
			if receiver := receiver(funcDecl); receiver != "" {
				method := funcDecl.Name.Name
				// Exported methods only (must begin with uppercase)
				if m, _ := utf8.DecodeRuneInString(method); unicode.IsUpper(m) {
					emulated[TypeMethod{receiver, method}] = true
				}
			}
		}
	}

	// Collect all the eligible methods from all the architectures and sizes.
	for _, aaf := range archAndFiles {
		for _, fname := range aaf.files {
			path := filepath.Join(archSimdPath, fname)
			f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
			if err != nil {
				log.Fatalf("Failed to parse %s: %v", path, err)
			}

			lci := 0
			fComments := f.Comments

			for _, decl := range f.Decls {
				if funcDecl, ok := decl.(*ast.FuncDecl); ok {

					lastComment := ""
					for ; lci < len(fComments) && fComments[lci].Pos() > funcDecl.Pos(); lci++ {
						lastComment = fComments[lci].Text()
					}

					recvType := receiver(funcDecl)

					if recvType == "" || knownReceivers[recvType] == "" {
						continue
					}

					methodName := funcDecl.Name.Name

					if strings.Contains(funcDecl.Doc.Text(), "Deprecated:") {
						pv("Skipping deprecated %s.%s\n", recvType, methodName)
						continue
					}

					if strings.Contains(lastComment, "Deprecated:") {
						pv("Skipping MAYBE deprecated %s.%s (check comment)\n", recvType, methodName)
						continue
					}

					// Skip methods that require a CPU feature beyond
					// what the simd package requires to use vectors
					// of this size.
					feature := requiredCPUFeature(funcDecl.Doc.Text())
					if sizeForType[recvType] == 128 {
						if strings.Contains(feature, "AVX512") || strings.Contains(feature, "AVX2") {
							pv("Skipping 128-bit %s.%s because AVX2/AVX512\n", recvType, methodName)
							continue
						}
					}
					if sizeForType[recvType] == 256 {
						if strings.Contains(feature, "AVX512") {
							pv("Skipping 256-bit %s.%s because AVX512\n", recvType, methodName)
							continue
						}
					}

					eltType := recvType[:strings.Index(recvType, "x")]

					// Allow reinterpret vectors.
					if xAt := strings.Index(methodName, "x"); xAt != -1 && (strings.HasPrefix(methodName, "As") || strings.HasPrefix(methodName, "ToInt") && strings.HasPrefix(eltType, "Mask")) {
						// We think this is fine, even if it changes the number of elements in the vector.
						// Tweak the method name so that they will line up properly.
						methodName = methodName[:xAt] + "s"
					} else if strings.HasPrefix(methodName, "Broadcast") {
						// Broadcast is okay
					} else {
						// Exclude "grouped", "Store" (not slice), and vector-size-changing methods.
						// TODO there may be "Grouped" operations that work.
						if strings.Contains(methodName, "Group") {
							pv("Skipping grouped method %s.%s\n", recvType, methodName)
							continue
						}
						if methodName == "StoreArray" || methodName == "StoreMasked" {
							pv("Skipping fixed-size Store method method %s.%s\n", recvType, methodName)
							continue
						}
						if methodName == "ToBits" && recvType[0] == 'M' {
							pv("Skipping Mask ToBits method (has varying return type) %s.%s\n", recvType, methodName)
							continue
						}
						if r := funcDecl.Type.Results; r != nil && len(r.List) == 1 {
							if id, ok := r.List[0].Type.(*ast.Ident); ok {
								if rs, ok := sizeForType[id.Name]; ok && rs != sizeForType[recvType] {
									pv("Skipping vector-size-changing method %s.%s\n", recvType, methodName)
									continue
								}
							}
						}
					}

					archReceiver := combine(aaf.arch, recvType)

					if methodsByType[archReceiver] == nil {
						methodsByType[archReceiver] = make(MethodSet)
					}
					methodsByType[archReceiver][methodName] = funcDecl
					allMethodNames[methodName] = true
				}
			}
		}
	}

	type ElemMethod struct {
		e, m string
	}

	intersectionByElem := make(map[string][]string)
	signatureByElemMethod := make(map[ElemMethod]*ast.FuncDecl)

	// elems is a slice of stems of vector types.
	elems := []string{"Int8", "Int16", "Int32", "Int64", "Uint8", "Uint16", "Uint32", "Uint64", "Float32", "Float64", "Mask8", "Mask16", "Mask32", "Mask64"}

	for _, elem := range elems {
		type128 := map128[elem]
		type256 := map256[elem]
		type512 := map512[elem]

		methods128w := methodsByType[combine("wasm", type128)]
		methods128n := methodsByType[combine("arm64", type128)]
		methods128 := methodsByType[combine("amd64", type128)]
		methods256 := methodsByType[combine("amd64", type256)]
		methods512 := methodsByType[combine("amd64", type512)]

		var intersection []string
		var missingNames []string
		for m := range allMethodNames {
			if wasm128, arm128, amd128, amd256, amd512 :=
				methods128w[m] == nil, methods128n[m] == nil, methods128[m] == nil, methods256[m] == nil, methods512[m] == nil; !wasm128 && !arm128 && !amd128 && !amd256 && !amd512 {
				intersection = append(intersection, m)
				signatureByElemMethod[ElemMethod{elem, m}] = methods512[m] // Use 512-bit signature (arbitrary choice, they should match)
			} else if !(wasm128 && arm128 && amd128 && amd256 && amd512) {
				missing[m] = whyMissing{wasm128, arm128, amd128, amd256, amd512}
				missingNames = append(missingNames, m)
			}
		}
		sort.Strings(missingNames)

		for _, m := range missingNames {
			pv("Missing implementation for %ss.%s on %s\n", elem, m, missing[m].String())
		}

		sort.Strings(intersection)

		intersectionByElem[elem] = intersection
	}

	// xlateType translates a type by replacing instances of types with keys in knownReceivers with their values,
	// and generates the string representation of the resulting type.  E.g., []Int8x32 -> []Int8s
	// (because Int8x32 -> Int8s in knownReceivers).
	var xlateType func(ast.Expr) string
	xlateType = func(e ast.Expr) string {
		switch t := e.(type) {
		case *ast.Ident:
			if mapped, ok := knownReceivers[t.Name]; ok {
				return mapped
			}
			return t.Name
		case *ast.StarExpr:
			return "*" + xlateType(t.X)
		case *ast.ArrayType:
			lenStr := ""
			if t.Len != nil {
				var buf strings.Builder
				format.Node(&buf, token.NewFileSet(), t.Len)
				lenStr = buf.String()
			}
			return "[" + lenStr + "]" + xlateType(t.Elt)
		case *ast.SelectorExpr:
			return xlateType(t.X) + "." + t.Sel.Name
		case *ast.Ellipsis:
			return "..." + xlateType(t.Elt)
		default:
			var buf strings.Builder
			format.Node(&buf, token.NewFileSet(), t)
			return buf.String()
		}
	}

	toScalar := func(s string) string {
		if strings.HasPrefix(s, "Mask") {
			return "int" + s[4:]
		}
		return strings.ToLower(s)
	}

	doTypes := func(w io.Writer) {

		pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }

		fmt.Fprint(w,
			`// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.

//go:build goexperiment.simdstable

// Scalable vector types for rewriting and emulation

package simd

import "simd/internal/bridge"

// internal SIMD marker, and hard dependence on simd/internal/bridge
type _simd bridge.ZeroSized
`)

		for _, elem := range elems {
			pf("type %ss struct {\n\t_ _simd\n\t%s // the actual vector size may be larger.\n}\n", elem, emulatedLanes(elem))
		}
	}

	doMethods := func(w io.Writer) {

		p := func(s ...any) { fmt.Fprint(w, s...) }
		pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }
		nl := func() { fmt.Fprintln(w) }

		fmt.Fprint(w,
			`// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.

//go:build goexperiment.simdstable && (amd64 || wasm || arm64)

// Computed intersection of methods for supported SIMD architectures and vector widths

package simd

`)

		// checkSpec checks that recv.name is in the generally available
		// spec API. Everything in the simd package must be, since the
		// simd package is the intersection of the generally available
		// archsimd APIs.
		//
		// TODO: Support emitting experimental methods in simd if they're
		// experimental in all inputs.
		checkSpec := func(recv string, name string) {
			fullName := name
			if recv != "" {
				fullName = recv + "." + name
			}
			if f := specIdx.Lookup(recv, name); f == nil {
				log.Fatalf("%s is missing from spec", fullName)
			} else if f.Experiment {
				log.Fatalf("%s is experimental in spec", fullName)
			}
		}

		for _, elem := range elems {
			intersection := intersectionByElem[elem]

			if elem[0] != 'M' {
				// cannot load masks

				checkSpec("", "Load"+elem+"s")
				pf("func Load%ss([]%s) %ss\n", elem, toScalar(elem), elem)

				checkSpec("", "Load"+elem+"sPart")
				pf("func Load%ssPart([]%s) (%ss, int)\n", elem, toScalar(elem), elem)

				checkSpec("", "Broadcast"+elem+"s")
				pf("func Broadcast%ss(%s) %ss\n", elem, toScalar(elem), elem)
			}

			for _, m := range intersection {
				fd := signatureByElemMethod[ElemMethod{elem, m}]
				elems := elem + "s"
				checkSpec(elems, m)
				// The spec provides the doc comment. Add any
				// implementation notes, which follow it.
				if note := implementationNotes[m]; note != "" {
					for line := range strings.SplitSeq(note, "\n") {
						pf("// %s\n", line)
					}
				}
				pf("func (x %s) %s(", elems, m)

				if !emulated[TypeMethod{elems, m}] {
					pw("Missing emulated method for %s.%s\n", elems, m)
				} else {
					delete(emulated, TypeMethod{elems, m})
				}

				if fd.Type.Params != nil {
					for i, field := range fd.Type.Params.List {
						if i > 0 {
							p(", ")
						}
						if len(field.Names) > 0 {
							for j, name := range field.Names {
								if j > 0 {
									p(", ")
								}
								p(name.Name)
							}
							p(" ")
						}
						p(xlateType(field.Type))
					}
				}
				p(")")

				if fd.Type.Results != nil && len(fd.Type.Results.List) > 0 {
					p(" ")
					needsParens := len(fd.Type.Results.List) > 1 || (len(fd.Type.Results.List) == 1 && len(fd.Type.Results.List[0].Names) > 0)
					if needsParens {
						p("(")
					}
					for i, field := range fd.Type.Results.List {
						if i > 0 {
							p(", ")
						}
						if len(field.Names) > 0 {
							for j, name := range field.Names {
								if j > 0 {
									p(", ")
								}
								p(name.Name)
							}
							p(" ")
						}
						p(xlateType(field.Type))
					}
					if needsParens {
						p(")")
					}
				}
				nl()
			}
		}
	}

	doTypes(files.NewGoFile("simd/simd_types.go"))
	doMethods(files.NewGoFile("simd/simd_stubs.go"))

	// Report any emulated functions/methods that do not appear in simd.go.
	var extraMocks []TypeMethod
	for x := range emulated {
		extraMocks = append(extraMocks, x)
	}
	slices.SortFunc(extraMocks, func(a, b TypeMethod) int {
		if c := strings.Compare(a.t, b.t); c != 0 {
			return c
		}
		return strings.Compare(a.m, b.m)
	})

	for _, x := range extraMocks {
		pw("%s contains %s.%s missing from intersected methods\n", emulatedFile, x.t, x.m)
	}

	// archTypes returns the vector and mask types of arch, in order.
	archTypes := func(arch string) []string {
		var typesForArch []string
		for t := range knownReceivers {
			if methodsByType[combine(arch, t)] != nil {
				typesForArch = append(typesForArch, t)
			}
		}
		slices.SortFunc(typesForArch, sgutil.CompareNatural)
		return typesForArch
	}

	for _, aaf := range archAndFiles {
		arch := aaf.arch
		// This is what writes the bridge.  The rewriter in the compiler will target these types.
		doArchWrites := func(w io.Writer) {
			p := func(s ...any) { fmt.Fprint(w, s...) }
			pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }
			nl := func() { fmt.Fprintln(w) }

			pf("// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.\n\n")
			pf("//go:build goexperiment.simdstable && %s\n\n", arch)
			pf("package bridge\n\n")
			pf("import \"simd/archsimd\"\n\n")
			pf("\n")
			pf("// These types/methods/functions forward calls to their counterparts in simd/archsimd.\n")
			pf("// Interposing this package allows a clean separation of \"simd\" from \"archsimd\" and\n")
			pf("// also allows additional useful exported declarations that would weirdly pollute archsimd.\n")
			pf("\n")

			typesForArch := archTypes(arch)

			for _, t := range typesForArch {
				pf("type %s archsimd.%s\n", t, t)
				if xAt := strings.Index(t, "x"); xAt != -1 && !strings.HasPrefix(t, "Mask") {
					elem := t[:xAt]
					scalar := toScalar(elem)
					pf("func Load%s(s []%s) %s {\n\treturn %s(archsimd.Load%s(s))\n}\n", t, scalar, t, t, t)
					pf("func Load%sPart(s []%s) (%s, int) {\n\tv, n := archsimd.Load%sPart(s)\n\treturn %s(v), n\n}\n",
						t, scalar, t, t, t)
					pf("func Broadcast%s(x %s) %s {\n\treturn %s(archsimd.Broadcast%s(x))\n}\n", t, scalar, t, t, t)
				}
			}
			nl()

			typeStr := func(e ast.Expr) string {
				var buf strings.Builder
				format.Node(&buf, token.NewFileSet(), e)
				return buf.String()
			}

			// convertArg emits a type conversion for name to the
			// type in e (which should be Ident or StarExpr)
			convertArg := func(name string, e ast.Expr) string {
				switch t := e.(type) {
				case *ast.Ident:
					if _, ok := knownReceivers[t.Name]; ok {
						return fmt.Sprintf("archsimd.%s(%s)", t.Name, name)
					}
				case *ast.StarExpr:
					if ident, ok := t.X.(*ast.Ident); ok {
						if _, ok := knownReceivers[ident.Name]; ok {
							return fmt.Sprintf("(*archsimd.%s)(%s)", ident.Name, name)
						}
					}
				}
				return name
			}

			wrapResult := func(call string, e ast.Expr) string {
				switch t := e.(type) {
				case *ast.Ident:
					if _, ok := knownReceivers[t.Name]; ok {
						return fmt.Sprintf("%s(%s)", t.Name, call)
					}
				case *ast.StarExpr:
					if ident, ok := t.X.(*ast.Ident); ok {
						if _, ok := knownReceivers[ident.Name]; ok {
							return fmt.Sprintf("(*%s)(%s)", ident.Name, call)
						}
					}
				}
				return call
			}

			for _, elem := range elems {
				intersection := intersectionByElem[elem]
				for _, m := range intersection {
					for _, at := range typesForArch {
						if map128[elem] != at && map256[elem] != at && map512[elem] != at {
							continue
						}
						fd := methodsByType[combine(arch, at)][m]
						if fd == nil {
							continue
						}
						bridgeName := fd.Name.Name
						if strings.HasPrefix(at, "Mask") && strings.HasPrefix(bridgeName, "ToInt") {
							x := strings.Index(bridgeName, "x")
							bridgeName = bridgeName[:x] + "s"
						}
						pf("func (x %s) %s(", at, bridgeName)
						var args []string
						if fd.Type.Params != nil {
							paramCount := 0
							for _, field := range fd.Type.Params.List {
								if len(field.Names) > 0 {
									for _, name := range field.Names {
										if paramCount > 0 {
											p(", ")
										}
										pf("%s %s", name.Name, typeStr(field.Type))
										args = append(args, convertArg(name.Name, field.Type))
										paramCount++
									}
								} else {
									if paramCount > 0 {
										p(", ")
									}
									paramName := fmt.Sprintf("p%d", paramCount)
									pf("%s %s", paramName, typeStr(field.Type))
									args = append(args, convertArg(paramName, field.Type))
									paramCount++
								}
							}
						}
						p(")")

						var results []ast.Expr
						if fd.Type.Results != nil {
							p(" ")
							needsParens := len(fd.Type.Results.List) > 1 || (len(fd.Type.Results.List) == 1 && len(fd.Type.Results.List[0].Names) > 0)
							if needsParens {
								p("(")
							}
							for i, field := range fd.Type.Results.List {
								if i > 0 {
									p(", ")
								}
								results = append(results, field.Type)
								p(typeStr(field.Type))
							}
							if needsParens {
								p(")")
							}
						}

						p(" {\n\t")
						if len(results) > 0 {
							p("return ")
						}

						callStr := fmt.Sprintf("(archsimd.%s(x)).%s(%s)", at, fd.Name.Name, strings.Join(args, ", "))
						if len(results) == 1 {
							p(wrapResult(callStr, results[0]))
						} else {
							p(callStr)
						}
						p("\n}\n\n")
					}
				}
			}
		}
		doArchWrites(files.NewGoFile("simd/internal/bridge/decls_" + arch + ".go"))

		doToFromWrites := func(w io.Writer) {
			pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }

			pf("// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.\n\n")
			pf("//go:build goexperiment.simdstable && %s\n\n", arch)
			pf("package simd\n\n")
			pf("import (\n\t\"simd/archsimd\"\n\t\"simd/internal/bridge\"\n)\n\n")

			for _, elem := range elems {
				var archTypes []string
				if methodsByType[combine(arch, map128[elem])] != nil {
					archTypes = append(archTypes, map128[elem])
				}
				if methodsByType[combine(arch, map256[elem])] != nil {
					archTypes = append(archTypes, map256[elem])
				}
				if methodsByType[combine(arch, map512[elem])] != nil {
					archTypes = append(archTypes, map512[elem])
				}

				if len(archTypes) == 0 {
					continue
				}

				writeToArchDoc(w, elem+"s")
				pf("func (x %ss) ToArch() any\n\n", elem)

				var intfOpts []string
				for _, t := range archTypes {
					intfOpts = append(intfOpts, "archsimd."+t)
				}

				writeFromArchDoc(w, elem+"s")
				pf("func %ssFromArch[A %s](x A) %ss {\n", elem, strings.Join(intfOpts, " | "), elem)
				pf("\tswitch a := any(x).(type) {\n")
				pf("\t// The return expression is written this way because the code will be rewritten\n")
				pf("\t// with %ss replaced by one of the arch types, and without the any-assert\n", elem)
				pf("\t// hack the rewritten code would not pass type checking.\n")
				pf("\t// The backend of the compiler will eat this and turn it into no code at all,\n")
				pf("\t// assuming it inlines.\n")

				for _, t := range archTypes {
					pf("\tcase archsimd.%s:\n", t)
					pf("\t\tif z, ok := any(bridge.%s(a)).(%ss); ok {\n", t, elem)
					pf("\t\t\treturn z\n")
					pf("\t\t}\n")
					pf("\t\tpanic(fromArchError(\"simd.%ssFromArch: cannot convert from archsimd.%s\"))\n", elem, t)
				}
				pf("\t}\n\tpanic(\"unreachable\")\n}\n\n")
			}
		}
		doToFromWrites(files.NewGoFile("simd/tofrom_" + arch + ".go"))
	}

	doToFromEmulated := func(w io.Writer) {
		pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }
		pf("// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.\n\n")
		pf("//go:build goexperiment.simdstable && !(amd64 || wasm || arm64)\n\n")
		pf("package simd\n\n")
		for _, elem := range elems {
			writeToArchDoc(w, elem+"s")
			pf("func (x %ss) ToArch() any {\n\treturn nil\n}\n\n", elem)
		}
	}
	doToFromEmulated(files.NewGoFile("simd/tofrom_emulated.go"))

	generateBridgeEmulated(specIdx)

	if minorProblem {
		pw("The logged warnings did not prevent generation of the midway API files, but the API is flawed (lacks emulations, documentation, etc).\n")
	}
}

// generatedBridgeEmulated generates internal/bridge/simd_emulated.go from
// simd_emulated.go.
//
// simd_emulated.go is hand-written, and cmd/specfill fills its doc comments
// from spec. To keep this output independent of whether specfill has run since
// spec last changed, fill the doc comments here, too, before transforming the
// file.
func generateBridgeEmulated(specIdx *specgen.Index) {
	const path = "simd/simd_emulated.go"
	b, err := genFlags.ReadFile(path)
	if err != nil {
		log.Fatalf("reading %s: %v", path, err)
	}
	b, err = specdoc.Fill(b, specIdx, specdoc.Options{
		Filename:        "src/" + path,
		AllowDocRewrite: true,
	})
	if err != nil {
		log.Fatalf("filling %s: %v", path, err)
	}
	lines := strings.Split(string(b), "\n")
	var sawBuild, sawPackage bool
	for i, line := range lines {
		if strings.HasPrefix(line, "//go:build ") && strings.Contains(line, "!(") && !sawBuild {
			lines[i] = strings.Replace(line, "!(", "(", 1)
			sawBuild = true
		} else if strings.TrimSpace(line) == "package simd" && !sawPackage {
			lines[i] = "package bridge"
			sawPackage = true
		}
	}
	if !sawBuild || !sawPackage {
		log.Fatalf("transforming simd/simd_emulated.go: sawBuild=%v sawPackage=%v", sawBuild, sawPackage)
	}
	dst := files.NewGoFile("simd/internal/bridge/simd_emulated.go")
	dst.WriteString("// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.\n\n")
	dst.WriteString(strings.Join(lines, "\n"))
}

var emulatedElemRE = regexp.MustCompile(`^(Mask|Int|Uint|Float)(8|16|32|64)$`)

// emulatedLanes returns the field declaration holding the lanes of the
// 128-bit emulated form of simd type elem+"s". Vectors are an array of
// their element type, so == on them compares lanes, like a struct with one
// field per lane. Masks are a bitmask with one bit per lane, using the
// narrowest unsigned type that fits, whose unused high bits are always zero.
//
// Masks could instead be an array of lanes that are 0 or -1, like most
// native fixed-width masks (AVX-512 masks are bitmasks). That would let
// Masked and IfElse use bitwise operations instead of testing one bit per
// lane, but they are only about 4% faster that way, while every mask
// operation (And, Or, Xor, AndNot, All, Any, None, TrailingZeros, ToBits)
// would become a loop over the lanes, which is slower than the single word
// operation it is now.
func emulatedLanes(elem string) string {
	m := emulatedElemRE.FindStringSubmatch(elem)
	if m == nil {
		panic("unexpected element type " + elem)
	}
	elemBits, _ := strconv.Atoi(m[2])
	lanes := 128 / elemBits
	if m[1] == "Mask" {
		if lanes > 8 {
			return "m uint16"
		}
		return "m uint8"
	}
	return fmt.Sprintf("v [%d]%s", lanes, strings.ToLower(elem))
}

func writeToArchDoc(w io.Writer, typ string) {
	fmt.Fprintf(w, `// ToArch returns x as the [simd/archsimd] type backing %s in this process
// (for vector width [VectorBits]), or nil if [Emulated] is true.
//
// Because the dynamic type of the result depends on CPU features and
// GODEBUG=simd, and future Go releases may return other [simd/archsimd] types
// (such as scalable types on arm64), callers should use a type switch with a
// default case.
`, typ)
}

func writeFromArchDoc(w io.Writer, typ string) {
	fmt.Fprintf(w, `// %sFromArch converts x to %s.
//
// It panics if [Emulated] is true or if x is not for vector width
// [VectorBits] (that is, x.Len() != (%s{}).Len()).
`, typ, typ, typ)
}
