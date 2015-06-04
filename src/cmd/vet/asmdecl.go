// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Identify mismatches between assembly files and Go func declarations.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"regexp"
	"strconv"
	"strings"
)

// 'kind' is a kind of assembly variable.
// The kinds 1, 2, 4, 8 stand for values of that size.
type asmKind int

// These special kinds are not valid sizes.
const (
	asmString asmKind = 100 + iota
	asmSlice
	asmInterface
	asmEmptyInterface
)

// An asmArch describes assembly parameters for an architecture
type asmArch struct {
	name      string
	ptrSize   int
	intSize   int
	maxAlign  int
	bigEndian bool
	stack     string
	lr        bool
}

// An asmFunc describes the expected variables for a function on a given architecture.
type asmFunc struct {
	arch        *asmArch
	size        int // size of all arguments
	vars        map[string]*asmVar
	varByOffset map[int]*asmVar
}

// An asmVar describes a single assembly variable.
type asmVar struct {
	name  string
	kind  asmKind
	typ   string
	off   int
	size  int
	inner []*asmVar
}

var (
	asmArch386       = asmArch{"386", 4, 4, 4, false, "SP", false}
	asmArchArm       = asmArch{"arm", 4, 4, 4, false, "R13", true}
	asmArchArm64     = asmArch{"arm64", 8, 8, 8, false, "RSP", true}
	asmArchAmd64     = asmArch{"amd64", 8, 8, 8, false, "SP", false}
	asmArchAmd64p32  = asmArch{"amd64p32", 4, 4, 8, false, "SP", false}
	asmArchPower64   = asmArch{"power64", 8, 8, 8, true, "R1", true}
	asmArchPower64LE = asmArch{"power64le", 8, 8, 8, false, "R1", true}

	arches = []*asmArch{
		&asmArch386,
		&asmArchArm,
		&asmArchArm64,
		&asmArchAmd64,
		&asmArchAmd64p32,
		&asmArchPower64,
		&asmArchPower64LE,
	}
)

var (
	re           = regexp.MustCompile
	asmPlusBuild = re(`//\s+\+build\s+([^\n]+)`)
	asmTEXT      = re(`\bTEXT\b.*·([^\(]+)\(SB\)(?:\s*,\s*([0-9A-Z|+]+))?(?:\s*,\s*\$(-?[0-9]+)(?:-([0-9]+))?)?`)
	asmDATA      = re(`\b(DATA|GLOBL)\b`)
	asmNamedFP   = re(`([a-zA-Z0-9_\xFF-\x{10FFFF}]+)(?:\+([0-9]+))\(FP\)`)
	asmUnnamedFP = re(`[^+\-0-9](([0-9]+)\(FP\))`)
	asmSP        = re(`[^+\-0-9](([0-9]+)\(([A-Z0-9]+)\))`)
	asmOpcode    = re(`^\s*(?:[A-Z0-9a-z_]+:)?\s*([A-Z]+)\s*([^,]*)(?:,\s*(.*))?`)
	power64Suff  = re(`([BHWD])(ZU|Z|U|BR)?$`)
)

func asmCheck(pkg *Package) {
	if !vet("asmdecl") {
		return
	}

	// No work if no assembly files.
	if !pkg.hasFileWithSuffix(".s") {
		return
	}

	// Gather declarations. knownFunc[name][arch] is func description.
	knownFunc := make(map[string]map[string]*asmFunc)

	for _, f := range pkg.files {
		if f.file != nil {
			for _, decl := range f.file.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok && decl.Body == nil {
					knownFunc[decl.Name.Name] = f.asmParseDecl(decl)
				}
			}
		}
	}

Files:
	for _, f := range pkg.files {
		if !strings.HasSuffix(f.name, ".s") {
			continue
		}
		Println("Checking file", f.name)

		// Determine architecture from file name if possible.
		var arch string
		var archDef *asmArch
		for _, a := range arches {
			if strings.HasSuffix(f.name, "_"+a.name+".s") {
				arch = a.name
				archDef = a
				break
			}
		}

		lines := strings.SplitAfter(string(f.content), "\n")
		var (
			fn                 *asmFunc
			fnName             string
			localSize, argSize int
			wroteSP            bool
			haveRetArg         bool
			retLine            []int
		)

		flushRet := func() {
			if fn != nil && fn.vars["ret"] != nil && !haveRetArg && len(retLine) > 0 {
				v := fn.vars["ret"]
				for _, line := range retLine {
					f.Badf(token.NoPos, "%s:%d: [%s] %s: RET without writing to %d-byte ret+%d(FP)", f.name, line, arch, fnName, v.size, v.off)
				}
			}
			retLine = nil
		}
		for lineno, line := range lines {
			lineno++

			badf := func(format string, args ...interface{}) {
				f.Badf(token.NoPos, "%s:%d: [%s] %s: %s", f.name, lineno, arch, fnName, fmt.Sprintf(format, args...))
			}

			if arch == "" {
				// Determine architecture from +build line if possible.
				if m := asmPlusBuild.FindStringSubmatch(line); m != nil {
				Fields:
					for _, fld := range strings.Fields(m[1]) {
						for _, a := range arches {
							if a.name == fld {
								arch = a.name
								archDef = a
								break Fields
							}
						}
					}
				}
			}

			if m := asmTEXT.FindStringSubmatch(line); m != nil {
				flushRet()
				if arch == "" {
					f.Warnf(token.NoPos, "%s: cannot determine architecture for assembly file", f.name)
					continue Files
				}
				fnName = m[1]
				fn = knownFunc[m[1]][arch]
				if fn != nil {
					size, _ := strconv.Atoi(m[4])
					if size != fn.size && (m[2] != "7" && !strings.Contains(m[2], "NOSPLIT") || size != 0) {
						badf("wrong argument size %d; expected $...-%d", size, fn.size)
					}
				}
				localSize, _ = strconv.Atoi(m[3])
				localSize += archDef.intSize
				if archDef.lr {
					// Account for caller's saved LR
					localSize += archDef.intSize
				}
				argSize, _ = strconv.Atoi(m[4])
				if fn == nil && !strings.Contains(fnName, "<>") {
					badf("function %s missing Go declaration", fnName)
				}
				wroteSP = false
				haveRetArg = false
				continue
			} else if strings.Contains(line, "TEXT") && strings.Contains(line, "SB") {
				// function, but not visible from Go (didn't match asmTEXT), so stop checking
				flushRet()
				fn = nil
				fnName = ""
				continue
			}

			if strings.Contains(line, "RET") {
				retLine = append(retLine, lineno)
			}

			if fnName == "" {
				continue
			}

			if asmDATA.FindStringSubmatch(line) != nil {
				fn = nil
			}

			if archDef == nil {
				continue
			}

			if strings.Contains(line, ", "+archDef.stack) || strings.Contains(line, ",\t"+archDef.stack) {
				wroteSP = true
				continue
			}

			for _, m := range asmSP.FindAllStringSubmatch(line, -1) {
				if m[3] != archDef.stack || wroteSP {
					continue
				}
				off := 0
				if m[1] != "" {
					off, _ = strconv.Atoi(m[2])
				}
				if off >= localSize {
					if fn != nil {
						v := fn.varByOffset[off-localSize]
						if v != nil {
							badf("%s should be %s+%d(FP)", m[1], v.name, off-localSize)
							continue
						}
					}
					if off >= localSize+argSize {
						badf("use of %s points beyond argument frame", m[1])
						continue
					}
					badf("use of %s to access argument frame", m[1])
				}
			}

			if fn == nil {
				continue
			}

			for _, m := range asmUnnamedFP.FindAllStringSubmatch(line, -1) {
				off, _ := strconv.Atoi(m[2])
				v := fn.varByOffset[off]
				if v != nil {
					badf("use of unnamed argument %s; offset %d is %s+%d(FP)", m[1], off, v.name, v.off)
				} else {
					badf("use of unnamed argument %s", m[1])
				}
			}

			for _, m := range asmNamedFP.FindAllStringSubmatch(line, -1) {
				name := m[1]
				off := 0
				if m[2] != "" {
					off, _ = strconv.Atoi(m[2])
				}
				if name == "ret" || strings.HasPrefix(name, "ret_") {
					haveRetArg = true
				}
				v := fn.vars[name]
				if v == nil {
					// Allow argframe+0(FP).
					if name == "argframe" && off == 0 {
						continue
					}
					v = fn.varByOffset[off]
					if v != nil {
						badf("unknown variable %s; offset %d is %s+%d(FP)", name, off, v.name, v.off)
					} else {
						badf("unknown variable %s", name)
					}
					continue
				}
				asmCheckVar(badf, fn, line, m[0], off, v)
			}
		}
		flushRet()
	}
}

// asmParseDecl parses a function decl for expected assembly variables.
func (f *File) asmParseDecl(decl *ast.FuncDecl) map[string]*asmFunc {
	var (
		arch   *asmArch
		fn     *asmFunc
		offset int
		failed bool
	)

	addVar := func(outer string, v asmVar) {
		if vo := fn.vars[outer]; vo != nil {
			vo.inner = append(vo.inner, &v)
		}
		fn.vars[v.name] = &v
		for i := 0; i < v.size; i++ {
			fn.varByOffset[v.off+i] = &v
		}
	}

	addParams := func(list []*ast.Field) {
		for i, fld := range list {
			// Determine alignment, size, and kind of type in declaration.
			var align, size int
			var kind asmKind
			names := fld.Names
			typ := f.gofmt(fld.Type)
			switch t := fld.Type.(type) {
			default:
				switch typ {
				default:
					f.Warnf(fld.Type.Pos(), "unknown assembly argument type %s", typ)
					failed = true
					return
				case "int8", "uint8", "byte", "bool":
					size = 1
				case "int16", "uint16":
					size = 2
				case "int32", "uint32", "float32":
					size = 4
				case "int64", "uint64", "float64":
					align = arch.maxAlign
					size = 8
				case "int", "uint":
					size = arch.intSize
				case "uintptr", "iword", "Word", "Errno", "unsafe.Pointer":
					size = arch.ptrSize
				case "string", "ErrorString":
					size = arch.ptrSize * 2
					align = arch.ptrSize
					kind = asmString
				}
			case *ast.ChanType, *ast.FuncType, *ast.MapType, *ast.StarExpr:
				size = arch.ptrSize
			case *ast.InterfaceType:
				align = arch.ptrSize
				size = 2 * arch.ptrSize
				if len(t.Methods.List) > 0 {
					kind = asmInterface
				} else {
					kind = asmEmptyInterface
				}
			case *ast.ArrayType:
				if t.Len == nil {
					size = arch.ptrSize + 2*arch.intSize
					align = arch.ptrSize
					kind = asmSlice
					break
				}
				f.Warnf(fld.Type.Pos(), "unsupported assembly argument type %s", typ)
				failed = true
			case *ast.StructType:
				f.Warnf(fld.Type.Pos(), "unsupported assembly argument type %s", typ)
				failed = true
			}
			if align == 0 {
				align = size
			}
			if kind == 0 {
				kind = asmKind(size)
			}
			offset += -offset & (align - 1)

			// Create variable for each name being declared with this type.
			if len(names) == 0 {
				name := "unnamed"
				if decl.Type.Results != nil && len(decl.Type.Results.List) > 0 && &list[0] == &decl.Type.Results.List[0] && i == 0 {
					// Assume assembly will refer to single unnamed result as r.
					name = "ret"
				}
				names = []*ast.Ident{{Name: name}}
			}
			for _, id := range names {
				name := id.Name
				addVar("", asmVar{
					name: name,
					kind: kind,
					typ:  typ,
					off:  offset,
					size: size,
				})
				switch kind {
				case 8:
					if arch.ptrSize == 4 {
						w1, w2 := "lo", "hi"
						if arch.bigEndian {
							w1, w2 = w2, w1
						}
						addVar(name, asmVar{
							name: name + "_" + w1,
							kind: 4,
							typ:  "half " + typ,
							off:  offset,
							size: 4,
						})
						addVar(name, asmVar{
							name: name + "_" + w2,
							kind: 4,
							typ:  "half " + typ,
							off:  offset + 4,
							size: 4,
						})
					}

				case asmEmptyInterface:
					addVar(name, asmVar{
						name: name + "_type",
						kind: asmKind(arch.ptrSize),
						typ:  "interface type",
						off:  offset,
						size: arch.ptrSize,
					})
					addVar(name, asmVar{
						name: name + "_data",
						kind: asmKind(arch.ptrSize),
						typ:  "interface data",
						off:  offset + arch.ptrSize,
						size: arch.ptrSize,
					})

				case asmInterface:
					addVar(name, asmVar{
						name: name + "_itable",
						kind: asmKind(arch.ptrSize),
						typ:  "interface itable",
						off:  offset,
						size: arch.ptrSize,
					})
					addVar(name, asmVar{
						name: name + "_data",
						kind: asmKind(arch.ptrSize),
						typ:  "interface data",
						off:  offset + arch.ptrSize,
						size: arch.ptrSize,
					})

				case asmSlice:
					addVar(name, asmVar{
						name: name + "_base",
						kind: asmKind(arch.ptrSize),
						typ:  "slice base",
						off:  offset,
						size: arch.ptrSize,
					})
					addVar(name, asmVar{
						name: name + "_len",
						kind: asmKind(arch.intSize),
						typ:  "slice len",
						off:  offset + arch.ptrSize,
						size: arch.intSize,
					})
					addVar(name, asmVar{
						name: name + "_cap",
						kind: asmKind(arch.intSize),
						typ:  "slice cap",
						off:  offset + arch.ptrSize + arch.intSize,
						size: arch.intSize,
					})

				case asmString:
					addVar(name, asmVar{
						name: name + "_base",
						kind: asmKind(arch.ptrSize),
						typ:  "string base",
						off:  offset,
						size: arch.ptrSize,
					})
					addVar(name, asmVar{
						name: name + "_len",
						kind: asmKind(arch.intSize),
						typ:  "string len",
						off:  offset + arch.ptrSize,
						size: arch.intSize,
					})
				}
				offset += size
			}
		}
	}

	m := make(map[string]*asmFunc)
	for _, arch = range arches {
		fn = &asmFunc{
			arch:        arch,
			vars:        make(map[string]*asmVar),
			varByOffset: make(map[int]*asmVar),
		}
		offset = 0
		addParams(decl.Type.Params.List)
		if decl.Type.Results != nil && len(decl.Type.Results.List) > 0 {
			offset += -offset & (arch.maxAlign - 1)
			addParams(decl.Type.Results.List)
		}
		fn.size = offset
		m[arch.name] = fn
	}

	if failed {
		return nil
	}
	return m
}

// asmCheckVar checks a single variable reference.
func asmCheckVar(badf func(string, ...interface{}), fn *asmFunc, line, expr string, off int, v *asmVar) {
	m := asmOpcode.FindStringSubmatch(line)
	if m == nil {
		if !strings.HasPrefix(strings.TrimSpace(line), "//") {
			badf("cannot find assembly opcode")
		}
		return
	}

	// Determine operand sizes from instruction.
	// Typically the suffix suffices, but there are exceptions.
	var src, dst, kind asmKind
	op := m[1]
	switch fn.arch.name + "." + op {
	case "386.FMOVLP":
		src, dst = 8, 4
	case "arm.MOVD":
		src = 8
	case "arm.MOVW":
		src = 4
	case "arm.MOVH", "arm.MOVHU":
		src = 2
	case "arm.MOVB", "arm.MOVBU":
		src = 1
	// LEA* opcodes don't really read the second arg.
	// They just take the address of it.
	case "386.LEAL":
		dst = 4
	case "amd64.LEAQ":
		dst = 8
	case "amd64p32.LEAL":
		dst = 4
	default:
		switch fn.arch.name {
		case "386", "amd64":
			if strings.HasPrefix(op, "F") && (strings.HasSuffix(op, "D") || strings.HasSuffix(op, "DP")) {
				// FMOVDP, FXCHD, etc
				src = 8
				break
			}
			if strings.HasPrefix(op, "F") && (strings.HasSuffix(op, "F") || strings.HasSuffix(op, "FP")) {
				// FMOVFP, FXCHF, etc
				src = 4
				break
			}
			if strings.HasSuffix(op, "SD") {
				// MOVSD, SQRTSD, etc
				src = 8
				break
			}
			if strings.HasSuffix(op, "SS") {
				// MOVSS, SQRTSS, etc
				src = 4
				break
			}
			if strings.HasPrefix(op, "SET") {
				// SETEQ, etc
				src = 1
				break
			}
			switch op[len(op)-1] {
			case 'B':
				src = 1
			case 'W':
				src = 2
			case 'L':
				src = 4
			case 'D', 'Q':
				src = 8
			}
		case "power64", "power64le":
			// Strip standard suffixes to reveal size letter.
			m := power64Suff.FindStringSubmatch(op)
			if m != nil {
				switch m[1][0] {
				case 'B':
					src = 1
				case 'H':
					src = 2
				case 'W':
					src = 4
				case 'D':
					src = 8
				}
			}
		}
	}
	if dst == 0 {
		dst = src
	}

	// Determine whether the match we're holding
	// is the first or second argument.
	if strings.Index(line, expr) > strings.Index(line, ",") {
		kind = dst
	} else {
		kind = src
	}

	vk := v.kind
	vt := v.typ
	switch vk {
	case asmInterface, asmEmptyInterface, asmString, asmSlice:
		// allow reference to first word (pointer)
		vk = v.inner[0].kind
		vt = v.inner[0].typ
	}

	if off != v.off {
		var inner bytes.Buffer
		for i, vi := range v.inner {
			if len(v.inner) > 1 {
				fmt.Fprintf(&inner, ",")
			}
			fmt.Fprintf(&inner, " ")
			if i == len(v.inner)-1 {
				fmt.Fprintf(&inner, "or ")
			}
			fmt.Fprintf(&inner, "%s+%d(FP)", vi.name, vi.off)
		}
		badf("invalid offset %s; expected %s+%d(FP)%s", expr, v.name, v.off, inner.String())
		return
	}
	if kind != 0 && kind != vk {
		var inner bytes.Buffer
		if len(v.inner) > 0 {
			fmt.Fprintf(&inner, " containing")
			for i, vi := range v.inner {
				if i > 0 && len(v.inner) > 2 {
					fmt.Fprintf(&inner, ",")
				}
				fmt.Fprintf(&inner, " ")
				if i > 0 && i == len(v.inner)-1 {
					fmt.Fprintf(&inner, "and ")
				}
				fmt.Fprintf(&inner, "%s+%d(FP)", vi.name, vi.off)
			}
		}
		badf("invalid %s of %s; %s is %d-byte value%s", op, expr, vt, vk, inner.String())
	}
}
