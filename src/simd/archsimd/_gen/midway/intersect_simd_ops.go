// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
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
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

type MethodSet map[string]*ast.FuncDecl
type TypeMethods map[string]MethodSet

type Comments struct {
	Types     map[string]string            `yaml:"types"`
	Functions map[string]string            `yaml:"functions"`
	Methods   map[string]map[string]string `yaml:"methods"`
}

var goRoot = flag.String("goroot", "../../../../..", "Go root")
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

func main() {
	minorProblem := false

	flag.Parse()

	var comments Comments
	commentsData, err := os.ReadFile("comments.yaml")
	if err != nil {
		log.Fatalf("Failed to read comments.yaml: %v", err)
	}
	if err := yaml.Unmarshal(commentsData, &comments); err != nil {
		log.Fatalf("Failed to parse comments.yaml: %v", err)
	}

	pv := func(f string, s ...any) {
		if *verbose {
			fmt.Fprintf(os.Stderr, f, s...)
		}
	}
	pw := func(f string, s ...any) {
		minorProblem = true
		fmt.Fprintf(os.Stderr, f, s...)
	}

	// Hardcoded path to archsimd
	archSimdPath := *goRoot + "/src/simd/archsimd"

	// Hardcoded list of files
	amd64Files := []string{"ops_amd64.go", "compare_gen_amd64.go", "types_amd64.go", "other_gen_amd64.go", "extra_amd64.go", "maskmerge_gen_amd64.go", "shuffles_amd64.go", "slice_gen_amd64.go", "slicepart_amd64.go", "slicepart_128.go", "string.go"}
	wasmFiles := []string{"ops_wasm.go", "types_wasm.go", "slicepart_wasm.go", "string.go", "slicepart_128.go", "ops_emulated_wasm.go"}
	neonFiles := []string{"compare_gen_arm64.go", "maskmerge_gen_arm64.go", "ops_arm64.go", "slicepart_128.go", "ops_internal_arm64.go", "other_gen_arm64.go", "slice_gen_arm64.go", "slicepart_arm64.go", "types_arm64.go"}

	emulatedFile := *goRoot + "/src/simd/simd_emulated.go"

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

	sizeForType := make(map[string]int)

	methodsByType := make(TypeMethods)

	allMethodNames := make(map[string]bool)

	missing := make(map[string]whyMissing)

	fset := token.NewFileSet()

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

					if sizeForType[recvType] == 128 {
						if s := funcDecl.Doc.Text(); strings.Contains(s, "AVX512") || strings.Contains(s, "AVX2") {
							pv("Skipping 128-bit %s.%s because AVX2/AVX512\n", recvType, methodName)
							continue
						}
					}
					if sizeForType[recvType] == 256 {
						if s := funcDecl.Doc.Text(); strings.Contains(s, "AVX512") {
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
						if lastChar := methodName[len(methodName)-1]; unicode.IsDigit(rune(lastChar)) && lastChar != eltType[len(eltType)-1] {
							pv("Skipping size-changing method %s.%s\n", recvType, methodName)
							continue
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

	intersectionByElem := make(map[string][]string)

	doWrites := func(w io.Writer) {

		p := func(s ...any) { fmt.Fprint(w, s...) }
		pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }
		nl := func() { fmt.Fprintln(w) }

		// elems is a slice of stems of vector types.
		elems := []string{"Int8", "Int16", "Int32", "Int64", "Uint8", "Uint16", "Uint32", "Uint64", "Float32", "Float64", "Mask8", "Mask16", "Mask32", "Mask64"}

		fmt.Fprintln(w,
			`// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

// Computed intersection of methods for supported SIMD architectures and vector widths

package simd`)

		if c := comments.Types["_simd"]; c != "" {
			pf("// %s\n", c)
		}
		p("type _simd struct {\n\t_ [0]func(*_simd) *_simd\n}\n")

		sigForMethod := make(map[string]*ast.FuncDecl)

		// xlateType translates a type by replacing instances of types with keys in knownReceivers with their values,
		// and generates the string representation of the resulting type.  E.g., []Int8x32 -> []Int8s
		// (because Int8x32 -> Int8s in knownReceivers
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
					sigForMethod[m] = methods512[m] // Use 512-bit signature (arbitrary choice, they should match)
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

			if c := comments.Types[elem+"s"]; c != "" {
				pf("// %s\n", c)
			}
			pf("type %ss struct {\n\t_       _simd\n\tatLeast [2]uint64 // the actual vector size may be larger.\n}\n", elem)

			if elem[0] != 'M' {
				// cannot load masks

				loadComment := comments.Functions["Load"+elem]
				if loadComment == "" && comments.Functions["default_LoadSlice"] != "" {
					loadComment = fmt.Sprintf(comments.Functions["default_LoadSlice"], elem, toScalar(elem), elem)
				}
				if loadComment != "" {
					pf("// %s\n", loadComment)
				}
				pf("func Load%ss([]%s) %ss\n", elem, toScalar(elem), elem)

				loadPartComment := comments.Functions["Load"+elem+"Part"]
				if loadPartComment == "" && comments.Functions["default_LoadPart"] != "" {
					loadPartComment = fmt.Sprintf(comments.Functions["default_LoadPart"], elem, toScalar(elem), elem)
				}
				if loadPartComment != "" {
					pf("// %s\n", loadPartComment)
				}
				pf("func Load%ssPart([]%s) (%ss, int)\n", elem, toScalar(elem), elem)

			}

			for _, m := range intersection {
				fd := sigForMethod[m]
				elems := elem + "s"
				methodComment := ""
				if typeMethods, ok := comments.Methods[elem+"s"]; ok {
					methodComment = typeMethods[m]
				}
				if methodComment != "" {
					pf("// %s\n", methodComment)
				} else {
					pw("Missing doc comment (in midway/comments.yaml) for %s.%s\n", elems, m)
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
	formatAndWrite(*goRoot+"/src/simd/simd.go", doWrites)
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

	elems := []string{"Int8", "Int16", "Int32", "Int64", "Uint8", "Uint16", "Uint32", "Uint64", "Float32", "Float64", "Mask8", "Mask16", "Mask32", "Mask64"}

	for _, aaf := range archAndFiles {
		arch := aaf.arch
		doArchWrites := func(w io.Writer) {
			p := func(s ...any) { fmt.Fprint(w, s...) }
			pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }
			nl := func() { fmt.Fprintln(w) }

			pf("// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.\n\n")
			pf("//go:build goexperiment.simd && %s\n\n", arch)
			pf("package bridge\n\n")
			pf("import \"simd/archsimd\"\n\n")
			pf("\n")
			pf("// These types/methods/functions forward calls to their counterparts in simd/archsimd.\n")
			pf("// Interposing this package allows a clean separation of \"simd\" from \"archsimd\" and\n")
			pf("// also allows additional useful exported declarations that would weirdly pollute archsimd.\n")
			pf("\n")

			var typesForArch []string
			for t := range knownReceivers {
				if methodsByType[combine(arch, t)] != nil {
					typesForArch = append(typesForArch, t)
				}
			}
			sort.Strings(typesForArch)

			toScalar := func(s string) string {
				if strings.HasPrefix(s, "Mask") {
					return "int" + s[4:]
				}
				return strings.ToLower(s)
			}

			for _, t := range typesForArch {
				pf("type %s archsimd.%s\n", t, t)
				if xAt := strings.Index(t, "x"); xAt != -1 && !strings.HasPrefix(t, "Mask") {
					elem := t[:xAt]
					scalar := toScalar(elem)
					pf("func Load%s(s []%s) %s {\n\treturn %s(archsimd.Load%s(s))\n}\n", t, scalar, t, t, t)
					pf("func Load%sPart(s []%s) (%s, int) {\n\tv, n := archsimd.Load%sPart(s)\n\treturn %s(v), n\n}\n", t, scalar, t, t, t)
				}
			}
			nl()

			typeStr := func(e ast.Expr) string {
				var buf strings.Builder
				format.Node(&buf, token.NewFileSet(), e)
				return buf.String()
			}

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
					for _, t := range typesForArch {
						if map128[elem] != t && map256[elem] != t && map512[elem] != t {
							continue
						}
						fd := methodsByType[combine(arch, t)][m]
						if fd == nil {
							continue
						}
						pf("func (x %s) %s(", t, fd.Name.Name)
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

						callStr := fmt.Sprintf("(archsimd.%s(x)).%s(%s)", t, fd.Name.Name, strings.Join(args, ", "))
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
		archDir := filepath.Join(*goRoot, "src", "simd", "internal", "bridge")
		os.MkdirAll(archDir, 0755)
		filename := filepath.Join(archDir, "decls_"+arch+".go")
		formatAndWrite(filename, doArchWrites)

		doToFromWrites := func(w io.Writer) {
			pf := func(f string, s ...any) { fmt.Fprintf(w, f, s...) }

			pf("// Code generated by 'go run -C $GOROOT/src/simd/archsimd/_gen/midway'; DO NOT EDIT.\n\n")
			pf("//go:build goexperiment.simd && %s\n\n", arch)
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

				pf("func (x %ss) ToArch() any\n\n", elem)

				var intfOpts []string
				for _, t := range archTypes {
					intfOpts = append(intfOpts, "archsimd."+t)
				}
				pf("type archSimd%ss interface {\n\t%s\n}\n\n", elem, strings.Join(intfOpts, " | "))

				pf("func %ssFromArch[T archSimd%ss](x T) %ss {\n", elem, elem, elem)
				pf("\tswitch a := any(x).(type) {\n")
				pf("\t// The return expression is written this way because the code will be rewritten\n")
				pf("\t// with %ss replaced by one of the arch types, and without the any-assert\n", elem)
				pf("\t// hack the rewritten code would not pass type checking.\n")
				pf("\t// The backend of the compiler will eat this and turn it into no code at all,\n")
				pf("\t// assuming it inlines.\n")

				for _, t := range archTypes {
					pf("\tcase archsimd.%s:\n", t)
					pf("\t\tvar t bridge.%s = bridge.%s(a)\n", t, t)
					pf("\t\treturn (any(t)).(%ss)\n", elem)
				}
				pf("\t}\n\tpanic(\"wrong type\")\n}\n\n")
			}
		}
		toFromFilename := filepath.Join(*goRoot, "src", "simd", "tofrom_"+arch+".go")
		formatAndWrite(toFromFilename, doToFromWrites)
	}

	if minorProblem {
		pw("The logged warnings did not prevent generation of the midway API files, but the API is flawed (lacks emulations, documentation, etc).\n")
	}
}

// numberLines takes a slice of bytes, and returns a string where each line
// is numbered, starting from 1.
func numberLines(data []byte) string {
	var buf bytes.Buffer
	r := bytes.NewReader(data)
	s := bufio.NewScanner(r)
	for i := 1; s.Scan(); i++ {
		fmt.Fprintf(&buf, "%d: %s\n", i, s.Text())
	}
	return buf.String()
}

func formatAndWrite(filename string, doWrites func(w io.Writer)) {
	if filename == "" {
		return
	}
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	out := new(bytes.Buffer)
	doWrites(out)

	b, err := format.Source(out.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "There was a problem formatting the generated code for %s, %v\n", filename, err)
		fmt.Fprintf(os.Stderr, "%s\n", numberLines(out.Bytes()))
		fmt.Fprintf(os.Stderr, "There was a problem formatting the generated code for %s, %v\n", filename, err)
		os.Exit(1)
	} else {
		f.Write(b)
		f.Close()
	}
}
