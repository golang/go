// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package asmdecl defines an Analyzer that reports mismatches between
// assembly files and Go declarations.
package asmdecl

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"go/types"
	"log"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
)

const Doc = "report mismatches between assembly files and Go declarations"

var Analyzer = &analysis.Analyzer{
	Name: "asmdecl",
	Doc:  Doc,
	Run:  run,
}

// 'kind' is a kind of assembly variable.
// The kinds 1, 2, 4, 8 stand for values of that size.
type asmKind int

// These special kinds are not valid sizes.
const (
	asmString asmKind = 100 + iota
	asmSlice
	asmArray
	asmInterface
	asmEmptyInterface
	asmStruct
	asmComplex
)

// An asmArch describes assembly parameters for an architecture
type asmArch struct {
	name      string
	bigEndian bool
	stack     string
	lr        bool
	// retRegs is a list of registers for return value in register ABI (ABIInternal).
	// For now, as we only check whether we write to any result, here we only need to
	// include the first integer register and first floating-point register. Accessing
	// any of them counts as writing to result.
	retRegs []string
	// calculated during initialization
	sizes    types.Sizes
	intSize  int
	ptrSize  int
	maxAlign int
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
	asmArch386      = asmArch{name: "386", bigEndian: false, stack: "SP", lr: false}
	asmArchArm      = asmArch{name: "arm", bigEndian: false, stack: "R13", lr: true}
	asmArchArm64    = asmArch{name: "arm64", bigEndian: false, stack: "RSP", lr: true, retRegs: []string{"R0", "F0"}}
	asmArchAmd64    = asmArch{name: "amd64", bigEndian: false, stack: "SP", lr: false, retRegs: []string{"AX", "X0"}}
	asmArchMips     = asmArch{name: "mips", bigEndian: true, stack: "R29", lr: true}
	asmArchMipsLE   = asmArch{name: "mipsle", bigEndian: false, stack: "R29", lr: true}
	asmArchMips64   = asmArch{name: "mips64", bigEndian: true, stack: "R29", lr: true}
	asmArchMips64LE = asmArch{name: "mips64le", bigEndian: false, stack: "R29", lr: true}
	asmArchPpc64    = asmArch{name: "ppc64", bigEndian: true, stack: "R1", lr: true}
	asmArchPpc64LE  = asmArch{name: "ppc64le", bigEndian: false, stack: "R1", lr: true}
	asmArchRISCV64  = asmArch{name: "riscv64", bigEndian: false, stack: "SP", lr: true}
	asmArchS390X    = asmArch{name: "s390x", bigEndian: true, stack: "R15", lr: true}
	asmArchWasm     = asmArch{name: "wasm", bigEndian: false, stack: "SP", lr: false}

	arches = []*asmArch{
		&asmArch386,
		&asmArchArm,
		&asmArchArm64,
		&asmArchAmd64,
		&asmArchMips,
		&asmArchMipsLE,
		&asmArchMips64,
		&asmArchMips64LE,
		&asmArchPpc64,
		&asmArchPpc64LE,
		&asmArchRISCV64,
		&asmArchS390X,
		&asmArchWasm,
	}
)

func init() {
	for _, arch := range arches {
		arch.sizes = types.SizesFor("gc", arch.name)
		if arch.sizes == nil {
			// TODO(adonovan): fix: now that asmdecl is not in the standard
			// library we cannot assume types.SizesFor is consistent with arches.
			// For now, assume 64-bit norms and print a warning.
			// But this warning should really be deferred until we attempt to use
			// arch, which is very unlikely. Better would be
			// to defer size computation until we have Pass.TypesSizes.
			arch.sizes = types.SizesFor("gc", "amd64")
			log.Printf("unknown architecture %s", arch.name)
		}
		arch.intSize = int(arch.sizes.Sizeof(types.Typ[types.Int]))
		arch.ptrSize = int(arch.sizes.Sizeof(types.Typ[types.UnsafePointer]))
		arch.maxAlign = int(arch.sizes.Alignof(types.Typ[types.Int64]))
	}
}

var (
	re           = regexp.MustCompile
	asmPlusBuild = re(`//\s+\+build\s+([^\n]+)`)
	asmTEXT      = re(`\bTEXT\b(.*)·([^\(]+)\(SB\)(?:\s*,\s*([0-9A-Z|+()]+))?(?:\s*,\s*\$(-?[0-9]+)(?:-([0-9]+))?)?`)
	asmDATA      = re(`\b(DATA|GLOBL)\b`)
	asmNamedFP   = re(`\$?([a-zA-Z0-9_\xFF-\x{10FFFF}]+)(?:\+([0-9]+))\(FP\)`)
	asmUnnamedFP = re(`[^+\-0-9](([0-9]+)\(FP\))`)
	asmSP        = re(`[^+\-0-9](([0-9]+)\(([A-Z0-9]+)\))`)
	asmOpcode    = re(`^\s*(?:[A-Z0-9a-z_]+:)?\s*([A-Z]+)\s*([^,]*)(?:,\s*(.*))?`)
	ppc64Suff    = re(`([BHWD])(ZU|Z|U|BR)?$`)
	abiSuff      = re(`^(.+)<(ABI.+)>$`)
)

func run(pass *analysis.Pass) (interface{}, error) {
	// No work if no assembly files.
	var sfiles []string
	for _, fname := range pass.OtherFiles {
		if strings.HasSuffix(fname, ".s") {
			sfiles = append(sfiles, fname)
		}
	}
	if sfiles == nil {
		return nil, nil
	}

	// Gather declarations. knownFunc[name][arch] is func description.
	knownFunc := make(map[string]map[string]*asmFunc)

	for _, f := range pass.Files {
		for _, decl := range f.Decls {
			if decl, ok := decl.(*ast.FuncDecl); ok && decl.Body == nil {
				knownFunc[decl.Name.Name] = asmParseDecl(pass, decl)
			}
		}
	}

Files:
	for _, fname := range sfiles {
		content, tf, err := analysisutil.ReadFile(pass.Fset, fname)
		if err != nil {
			return nil, err
		}

		// Determine architecture from file name if possible.
		var arch string
		var archDef *asmArch
		for _, a := range arches {
			if strings.HasSuffix(fname, "_"+a.name+".s") {
				arch = a.name
				archDef = a
				break
			}
		}

		lines := strings.SplitAfter(string(content), "\n")
		var (
			fn                 *asmFunc
			fnName             string
			abi                string
			localSize, argSize int
			wroteSP            bool
			noframe            bool
			haveRetArg         bool
			retLine            []int
		)

		flushRet := func() {
			if fn != nil && fn.vars["ret"] != nil && !haveRetArg && len(retLine) > 0 {
				v := fn.vars["ret"]
				resultStr := fmt.Sprintf("%d-byte ret+%d(FP)", v.size, v.off)
				if abi == "ABIInternal" {
					resultStr = "result register"
				}
				for _, line := range retLine {
					pass.Reportf(analysisutil.LineStart(tf, line), "[%s] %s: RET without writing to %s", arch, fnName, resultStr)
				}
			}
			retLine = nil
		}
		trimABI := func(fnName string) (string, string) {
			m := abiSuff.FindStringSubmatch(fnName)
			if m != nil {
				return m[1], m[2]
			}
			return fnName, ""
		}
		for lineno, line := range lines {
			lineno++

			badf := func(format string, args ...interface{}) {
				pass.Reportf(analysisutil.LineStart(tf, lineno), "[%s] %s: %s", arch, fnName, fmt.Sprintf(format, args...))
			}

			if arch == "" {
				// Determine architecture from +build line if possible.
				if m := asmPlusBuild.FindStringSubmatch(line); m != nil {
					// There can be multiple architectures in a single +build line,
					// so accumulate them all and then prefer the one that
					// matches build.Default.GOARCH.
					var archCandidates []*asmArch
					for _, fld := range strings.Fields(m[1]) {
						for _, a := range arches {
							if a.name == fld {
								archCandidates = append(archCandidates, a)
							}
						}
					}
					for _, a := range archCandidates {
						if a.name == build.Default.GOARCH {
							archCandidates = []*asmArch{a}
							break
						}
					}
					if len(archCandidates) > 0 {
						arch = archCandidates[0].name
						archDef = archCandidates[0]
					}
				}
			}

			// Ignore comments and commented-out code.
			if i := strings.Index(line, "//"); i >= 0 {
				line = line[:i]
			}

			if m := asmTEXT.FindStringSubmatch(line); m != nil {
				flushRet()
				if arch == "" {
					// Arch not specified by filename or build tags.
					// Fall back to build.Default.GOARCH.
					for _, a := range arches {
						if a.name == build.Default.GOARCH {
							arch = a.name
							archDef = a
							break
						}
					}
					if arch == "" {
						log.Printf("%s: cannot determine architecture for assembly file", fname)
						continue Files
					}
				}
				fnName = m[2]
				if pkgPath := strings.TrimSpace(m[1]); pkgPath != "" {
					// The assembler uses Unicode division slash within
					// identifiers to represent the directory separator.
					pkgPath = strings.Replace(pkgPath, "∕", "/", -1)
					if pkgPath != pass.Pkg.Path() {
						// log.Printf("%s:%d: [%s] cannot check cross-package assembly function: %s is in package %s", fname, lineno, arch, fnName, pkgPath)
						fn = nil
						fnName = ""
						abi = ""
						continue
					}
				}
				// Trim off optional ABI selector.
				fnName, abi = trimABI(fnName)
				flag := m[3]
				fn = knownFunc[fnName][arch]
				if fn != nil {
					size, _ := strconv.Atoi(m[5])
					if size != fn.size && (flag != "7" && !strings.Contains(flag, "NOSPLIT") || size != 0) {
						badf("wrong argument size %d; expected $...-%d", size, fn.size)
					}
				}
				localSize, _ = strconv.Atoi(m[4])
				localSize += archDef.intSize
				if archDef.lr && !strings.Contains(flag, "NOFRAME") {
					// Account for caller's saved LR
					localSize += archDef.intSize
				}
				argSize, _ = strconv.Atoi(m[5])
				noframe = strings.Contains(flag, "NOFRAME")
				if fn == nil && !strings.Contains(fnName, "<>") && !noframe {
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
				abi = ""
				continue
			}

			if strings.Contains(line, "RET") && !strings.Contains(line, "(SB)") {
				// RET f(SB) is a tail call. It is okay to not write the results.
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

			if strings.Contains(line, ", "+archDef.stack) || strings.Contains(line, ",\t"+archDef.stack) || strings.Contains(line, "NOP "+archDef.stack) || strings.Contains(line, "NOP\t"+archDef.stack) {
				wroteSP = true
				continue
			}

			if arch == "wasm" && strings.Contains(line, "CallImport") {
				// CallImport is a call out to magic that can write the result.
				haveRetArg = true
			}

			if abi == "ABIInternal" && !haveRetArg {
				for _, reg := range archDef.retRegs {
					if strings.Contains(line, reg) {
						haveRetArg = true
						break
					}
				}
			}

			for _, m := range asmSP.FindAllStringSubmatch(line, -1) {
				if m[3] != archDef.stack || wroteSP || noframe {
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
				asmCheckVar(badf, fn, line, m[0], off, v, archDef)
			}
		}
		flushRet()
	}
	return nil, nil
}

func asmKindForType(t types.Type, size int) asmKind {
	switch t := t.Underlying().(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.String:
			return asmString
		case types.Complex64, types.Complex128:
			return asmComplex
		}
		return asmKind(size)
	case *types.Pointer, *types.Chan, *types.Map, *types.Signature:
		return asmKind(size)
	case *types.Struct:
		return asmStruct
	case *types.Interface:
		if t.Empty() {
			return asmEmptyInterface
		}
		return asmInterface
	case *types.Array:
		return asmArray
	case *types.Slice:
		return asmSlice
	}
	panic("unreachable")
}

// A component is an assembly-addressable component of a composite type,
// or a composite type itself.
type component struct {
	size   int
	offset int
	kind   asmKind
	typ    string
	suffix string // Such as _base for string base, _0_lo for lo half of first element of [1]uint64 on 32 bit machine.
	outer  string // The suffix for immediately containing composite type.
}

func newComponent(suffix string, kind asmKind, typ string, offset, size int, outer string) component {
	return component{suffix: suffix, kind: kind, typ: typ, offset: offset, size: size, outer: outer}
}

// componentsOfType generates a list of components of type t.
// For example, given string, the components are the string itself, the base, and the length.
func componentsOfType(arch *asmArch, t types.Type) []component {
	return appendComponentsRecursive(arch, t, nil, "", 0)
}

// appendComponentsRecursive implements componentsOfType.
// Recursion is required to correct handle structs and arrays,
// which can contain arbitrary other types.
func appendComponentsRecursive(arch *asmArch, t types.Type, cc []component, suffix string, off int) []component {
	s := t.String()
	size := int(arch.sizes.Sizeof(t))
	kind := asmKindForType(t, size)
	cc = append(cc, newComponent(suffix, kind, s, off, size, suffix))

	switch kind {
	case 8:
		if arch.ptrSize == 4 {
			w1, w2 := "lo", "hi"
			if arch.bigEndian {
				w1, w2 = w2, w1
			}
			cc = append(cc, newComponent(suffix+"_"+w1, 4, "half "+s, off, 4, suffix))
			cc = append(cc, newComponent(suffix+"_"+w2, 4, "half "+s, off+4, 4, suffix))
		}

	case asmEmptyInterface:
		cc = append(cc, newComponent(suffix+"_type", asmKind(arch.ptrSize), "interface type", off, arch.ptrSize, suffix))
		cc = append(cc, newComponent(suffix+"_data", asmKind(arch.ptrSize), "interface data", off+arch.ptrSize, arch.ptrSize, suffix))

	case asmInterface:
		cc = append(cc, newComponent(suffix+"_itable", asmKind(arch.ptrSize), "interface itable", off, arch.ptrSize, suffix))
		cc = append(cc, newComponent(suffix+"_data", asmKind(arch.ptrSize), "interface data", off+arch.ptrSize, arch.ptrSize, suffix))

	case asmSlice:
		cc = append(cc, newComponent(suffix+"_base", asmKind(arch.ptrSize), "slice base", off, arch.ptrSize, suffix))
		cc = append(cc, newComponent(suffix+"_len", asmKind(arch.intSize), "slice len", off+arch.ptrSize, arch.intSize, suffix))
		cc = append(cc, newComponent(suffix+"_cap", asmKind(arch.intSize), "slice cap", off+arch.ptrSize+arch.intSize, arch.intSize, suffix))

	case asmString:
		cc = append(cc, newComponent(suffix+"_base", asmKind(arch.ptrSize), "string base", off, arch.ptrSize, suffix))
		cc = append(cc, newComponent(suffix+"_len", asmKind(arch.intSize), "string len", off+arch.ptrSize, arch.intSize, suffix))

	case asmComplex:
		fsize := size / 2
		cc = append(cc, newComponent(suffix+"_real", asmKind(fsize), fmt.Sprintf("real(complex%d)", size*8), off, fsize, suffix))
		cc = append(cc, newComponent(suffix+"_imag", asmKind(fsize), fmt.Sprintf("imag(complex%d)", size*8), off+fsize, fsize, suffix))

	case asmStruct:
		tu := t.Underlying().(*types.Struct)
		fields := make([]*types.Var, tu.NumFields())
		for i := 0; i < tu.NumFields(); i++ {
			fields[i] = tu.Field(i)
		}
		offsets := arch.sizes.Offsetsof(fields)
		for i, f := range fields {
			cc = appendComponentsRecursive(arch, f.Type(), cc, suffix+"_"+f.Name(), off+int(offsets[i]))
		}

	case asmArray:
		tu := t.Underlying().(*types.Array)
		elem := tu.Elem()
		// Calculate offset of each element array.
		fields := []*types.Var{
			types.NewVar(token.NoPos, nil, "fake0", elem),
			types.NewVar(token.NoPos, nil, "fake1", elem),
		}
		offsets := arch.sizes.Offsetsof(fields)
		elemoff := int(offsets[1])
		for i := 0; i < int(tu.Len()); i++ {
			cc = appendComponentsRecursive(arch, elem, cc, suffix+"_"+strconv.Itoa(i), off+i*elemoff)
		}
	}

	return cc
}

// asmParseDecl parses a function decl for expected assembly variables.
func asmParseDecl(pass *analysis.Pass, decl *ast.FuncDecl) map[string]*asmFunc {
	var (
		arch   *asmArch
		fn     *asmFunc
		offset int
	)

	// addParams adds asmVars for each of the parameters in list.
	// isret indicates whether the list are the arguments or the return values.
	// TODO(adonovan): simplify by passing (*types.Signature).{Params,Results}
	// instead of list.
	addParams := func(list []*ast.Field, isret bool) {
		argnum := 0
		for _, fld := range list {
			t := pass.TypesInfo.Types[fld.Type].Type

			// Work around https://golang.org/issue/28277.
			if t == nil {
				if ell, ok := fld.Type.(*ast.Ellipsis); ok {
					t = types.NewSlice(pass.TypesInfo.Types[ell.Elt].Type)
				}
			}

			align := int(arch.sizes.Alignof(t))
			size := int(arch.sizes.Sizeof(t))
			offset += -offset & (align - 1)
			cc := componentsOfType(arch, t)

			// names is the list of names with this type.
			names := fld.Names
			if len(names) == 0 {
				// Anonymous args will be called arg, arg1, arg2, ...
				// Similarly so for return values: ret, ret1, ret2, ...
				name := "arg"
				if isret {
					name = "ret"
				}
				if argnum > 0 {
					name += strconv.Itoa(argnum)
				}
				names = []*ast.Ident{ast.NewIdent(name)}
			}
			argnum += len(names)

			// Create variable for each name.
			for _, id := range names {
				name := id.Name
				for _, c := range cc {
					outer := name + c.outer
					v := asmVar{
						name: name + c.suffix,
						kind: c.kind,
						typ:  c.typ,
						off:  offset + c.offset,
						size: c.size,
					}
					if vo := fn.vars[outer]; vo != nil {
						vo.inner = append(vo.inner, &v)
					}
					fn.vars[v.name] = &v
					for i := 0; i < v.size; i++ {
						fn.varByOffset[v.off+i] = &v
					}
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
		addParams(decl.Type.Params.List, false)
		if decl.Type.Results != nil && len(decl.Type.Results.List) > 0 {
			offset += -offset & (arch.maxAlign - 1)
			addParams(decl.Type.Results.List, true)
		}
		fn.size = offset
		m[arch.name] = fn
	}

	return m
}

// asmCheckVar checks a single variable reference.
func asmCheckVar(badf func(string, ...interface{}), fn *asmFunc, line, expr string, off int, v *asmVar, archDef *asmArch) {
	m := asmOpcode.FindStringSubmatch(line)
	if m == nil {
		if !strings.HasPrefix(strings.TrimSpace(line), "//") {
			badf("cannot find assembly opcode")
		}
		return
	}

	addr := strings.HasPrefix(expr, "$")

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
		addr = true
	case "amd64.LEAQ":
		dst = 8
		addr = true
	default:
		switch fn.arch.name {
		case "386", "amd64":
			if strings.HasPrefix(op, "F") && (strings.HasSuffix(op, "D") || strings.HasSuffix(op, "DP")) {
				// FMOVDP, FXCHD, etc
				src = 8
				break
			}
			if strings.HasPrefix(op, "P") && strings.HasSuffix(op, "RD") {
				// PINSRD, PEXTRD, etc
				src = 4
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
			if op == "MOVO" || op == "MOVOU" {
				src = 16
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
		case "ppc64", "ppc64le":
			// Strip standard suffixes to reveal size letter.
			m := ppc64Suff.FindStringSubmatch(op)
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
		case "mips", "mipsle", "mips64", "mips64le":
			switch op {
			case "MOVB", "MOVBU":
				src = 1
			case "MOVH", "MOVHU":
				src = 2
			case "MOVW", "MOVWU", "MOVF":
				src = 4
			case "MOVV", "MOVD":
				src = 8
			}
		case "s390x":
			switch op {
			case "MOVB", "MOVBZ":
				src = 1
			case "MOVH", "MOVHZ":
				src = 2
			case "MOVW", "MOVWZ", "FMOVS":
				src = 4
			case "MOVD", "FMOVD":
				src = 8
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
	vs := v.size
	vt := v.typ
	switch vk {
	case asmInterface, asmEmptyInterface, asmString, asmSlice:
		// allow reference to first word (pointer)
		vk = v.inner[0].kind
		vs = v.inner[0].size
		vt = v.inner[0].typ
	case asmComplex:
		// Allow a single instruction to load both parts of a complex.
		if int(kind) == vs {
			kind = asmComplex
		}
	}
	if addr {
		vk = asmKind(archDef.ptrSize)
		vs = archDef.ptrSize
		vt = "address"
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
		badf("invalid %s of %s; %s is %d-byte value%s", op, expr, vt, vs, inner.String())
	}
}
