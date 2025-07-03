// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gen command generates Go code (in the parent directory) for all
// the architecture-specific opcodes, blocks, and rewrites.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"log"
	"math/bits"
	"os"
	"path"
	"regexp"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"slices"
	"sort"
	"strings"
	"sync"
)

// TODO: capitalize these types, so that we can more easily tell variable names
// apart from type names, and avoid awkward func parameters like "arch arch".

type arch struct {
	name               string
	pkg                string // obj package to import for this arch.
	genfile            string // source file containing opcode code generation.
	ops                []opData
	blocks             []blockData
	regnames           []string
	ParamIntRegNames   string
	ParamFloatRegNames string
	gpregmask          regMask
	fpregmask          regMask
	fp32regmask        regMask
	fp64regmask        regMask
	specialregmask     regMask
	framepointerreg    int8
	linkreg            int8
	generic            bool
	imports            []string
}

type opData struct {
	name              string
	reg               regInfo
	asm               string
	typ               string // default result type
	aux               string
	rematerializeable bool
	argLength         int32  // number of arguments, if -1, then this operation has a variable number of arguments
	commutative       bool   // this operation is commutative on its first 2 arguments (e.g. addition)
	resultInArg0      bool   // (first, if a tuple) output of v and v.Args[0] must be allocated to the same register
	resultNotInArgs   bool   // outputs must not be allocated to the same registers as inputs
	clobberFlags      bool   // this op clobbers flags register
	needIntTemp       bool   // need a temporary free integer register
	call              bool   // is a function call
	tailCall          bool   // is a tail call
	nilCheck          bool   // this op is a nil check on arg0
	faultOnNilArg0    bool   // this op will fault if arg0 is nil (and aux encodes a small offset)
	faultOnNilArg1    bool   // this op will fault if arg1 is nil (and aux encodes a small offset)
	hasSideEffects    bool   // for "reasons", not to be eliminated.  E.g., atomic store, #19182.
	zeroWidth         bool   // op never translates into any machine code. example: copy, which may sometimes translate to machine code, is not zero-width.
	unsafePoint       bool   // this op is an unsafe point, i.e. not safe for async preemption
	fixedReg          bool   // this op will be assigned a fixed register
	symEffect         string // effect this op has on symbol in aux
	scale             uint8  // amd64/386 indexed load scale
}

type blockData struct {
	name     string // the suffix for this block ("EQ", "LT", etc.)
	controls int    // the number of control values this type of block requires
	aux      string // the type of the Aux/AuxInt value, if any
}

type regInfo struct {
	// inputs[i] encodes the set of registers allowed for the i'th input.
	// Inputs that don't use registers (flags, memory, etc.) should be 0.
	inputs []regMask
	// clobbers encodes the set of registers that are overwritten by
	// the instruction (other than the output registers).
	clobbers regMask
	// outputs[i] encodes the set of registers allowed for the i'th output.
	outputs []regMask
}

type regMask uint64

func (a arch) regMaskComment(r regMask) string {
	var buf strings.Builder
	for i := uint64(0); r != 0; i++ {
		if r&1 != 0 {
			if buf.Len() == 0 {
				buf.WriteString(" //")
			}
			buf.WriteString(" ")
			buf.WriteString(a.regnames[i])
		}
		r >>= 1
	}
	return buf.String()
}

var archs []arch

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
var memprofile = flag.String("memprofile", "", "write memory profile to `file`")
var tracefile = flag.String("trace", "", "write trace to `file`")
var outDir = flag.String("outdir", "..", "directory in which to write generated files")

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}
	if *tracefile != "" {
		f, err := os.Create(*tracefile)
		if err != nil {
			log.Fatalf("failed to create trace output file: %v", err)
		}
		defer func() {
			if err := f.Close(); err != nil {
				log.Fatalf("failed to close trace file: %v", err)
			}
		}()

		if err := trace.Start(f); err != nil {
			log.Fatalf("failed to start trace: %v", err)
		}
		defer trace.Stop()
	}

	if *outDir != ".." {
		err := os.MkdirAll(*outDir, 0755)
		if err != nil {
			log.Fatalf("failed to create output directory: %v", err)
		}
	}

	slices.SortFunc(archs, func(a, b arch) int {
		return strings.Compare(a.name, b.name)
	})

	// The generate tasks are run concurrently, since they are CPU-intensive
	// that can easily make use of many cores on a machine.
	//
	// Note that there is no limit on the concurrency at the moment. On a
	// four-core laptop at the time of writing, peak RSS usually reaches
	// ~200MiB, which seems doable by practically any machine nowadays. If
	// that stops being the case, we can cap this func to a fixed number of
	// architectures being generated at once.

	tasks := []func(){
		genOp,
		genAllocators,
	}
	for _, a := range archs {
		a := a // the funcs are ran concurrently at a later time
		tasks = append(tasks, func() {
			genRules(a)
			genSplitLoadRules(a)
			genLateLowerRules(a)
		})
	}
	var wg sync.WaitGroup
	for _, task := range tasks {
		task := task
		wg.Add(1)
		go func() {
			task()
			wg.Done()
		}()
	}
	wg.Wait()

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal("could not create memory profile: ", err)
		}
		defer f.Close()
		runtime.GC() // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
	}
}

func outFile(file string) string {
	return *outDir + "/" + file
}

func genOp() {
	w := new(bytes.Buffer)
	fmt.Fprintf(w, "// Code generated from _gen/*Ops.go using 'go generate'; DO NOT EDIT.\n")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "package ssa")

	fmt.Fprintln(w, "import (")
	fmt.Fprintln(w, "\"cmd/internal/obj\"")
	for _, a := range archs {
		if a.pkg != "" {
			fmt.Fprintf(w, "%q\n", a.pkg)
		}
	}
	fmt.Fprintln(w, ")")

	// generate Block* declarations
	fmt.Fprintln(w, "const (")
	fmt.Fprintln(w, "BlockInvalid BlockKind = iota")
	for _, a := range archs {
		fmt.Fprintln(w)
		for _, d := range a.blocks {
			fmt.Fprintf(w, "Block%s%s\n", a.Name(), d.name)
		}
	}
	fmt.Fprintln(w, ")")

	// generate block kind string method
	fmt.Fprintln(w, "var blockString = [...]string{")
	fmt.Fprintln(w, "BlockInvalid:\"BlockInvalid\",")
	for _, a := range archs {
		fmt.Fprintln(w)
		for _, b := range a.blocks {
			fmt.Fprintf(w, "Block%s%s:\"%s\",\n", a.Name(), b.name, b.name)
		}
	}
	fmt.Fprintln(w, "}")
	fmt.Fprintln(w, "func (k BlockKind) String() string {return blockString[k]}")

	// generate block kind auxint method
	fmt.Fprintln(w, "func (k BlockKind) AuxIntType() string {")
	fmt.Fprintln(w, "switch k {")
	for _, a := range archs {
		for _, b := range a.blocks {
			if b.auxIntType() == "invalid" {
				continue
			}
			fmt.Fprintf(w, "case Block%s%s: return \"%s\"\n", a.Name(), b.name, b.auxIntType())
		}
	}
	fmt.Fprintln(w, "}")
	fmt.Fprintln(w, "return \"\"")
	fmt.Fprintln(w, "}")

	// generate Op* declarations
	fmt.Fprintln(w, "const (")
	fmt.Fprintln(w, "OpInvalid Op = iota") // make sure OpInvalid is 0.
	for _, a := range archs {
		fmt.Fprintln(w)
		for _, v := range a.ops {
			if v.name == "Invalid" {
				continue
			}
			fmt.Fprintf(w, "Op%s%s\n", a.Name(), v.name)
		}
	}
	fmt.Fprintln(w, ")")

	// generate OpInfo table
	fmt.Fprintln(w, "var opcodeTable = [...]opInfo{")
	fmt.Fprintln(w, " { name: \"OpInvalid\" },")
	for _, a := range archs {
		fmt.Fprintln(w)

		pkg := path.Base(a.pkg)
		for _, v := range a.ops {
			if v.name == "Invalid" {
				continue
			}
			fmt.Fprintln(w, "{")
			fmt.Fprintf(w, "name:\"%s\",\n", v.name)

			// flags
			if v.aux != "" {
				fmt.Fprintf(w, "auxType: aux%s,\n", v.aux)
			}
			fmt.Fprintf(w, "argLen: %d,\n", v.argLength)

			if v.rematerializeable {
				if v.reg.clobbers != 0 {
					log.Fatalf("%s is rematerializeable and clobbers registers", v.name)
				}
				if v.clobberFlags {
					log.Fatalf("%s is rematerializeable and clobbers flags", v.name)
				}
				fmt.Fprintln(w, "rematerializeable: true,")
			}
			if v.commutative {
				fmt.Fprintln(w, "commutative: true,")
			}
			if v.resultInArg0 {
				fmt.Fprintln(w, "resultInArg0: true,")
				// OpConvert's register mask is selected dynamically,
				// so don't try to check it in the static table.
				if v.name != "Convert" && v.reg.inputs[0] != v.reg.outputs[0] {
					log.Fatalf("%s: input[0] and output[0] must use the same registers for %s", a.name, v.name)
				}
				if v.name != "Convert" && v.commutative && v.reg.inputs[1] != v.reg.outputs[0] {
					log.Fatalf("%s: input[1] and output[0] must use the same registers for %s", a.name, v.name)
				}
			}
			if v.resultNotInArgs {
				fmt.Fprintln(w, "resultNotInArgs: true,")
			}
			if v.clobberFlags {
				fmt.Fprintln(w, "clobberFlags: true,")
			}
			if v.needIntTemp {
				fmt.Fprintln(w, "needIntTemp: true,")
			}
			if v.call {
				fmt.Fprintln(w, "call: true,")
			}
			if v.tailCall {
				fmt.Fprintln(w, "tailCall: true,")
			}
			if v.nilCheck {
				fmt.Fprintln(w, "nilCheck: true,")
			}
			if v.faultOnNilArg0 {
				fmt.Fprintln(w, "faultOnNilArg0: true,")
				if v.aux != "Sym" && v.aux != "SymOff" && v.aux != "SymValAndOff" && v.aux != "Int64" && v.aux != "Int32" && v.aux != "" {
					log.Fatalf("faultOnNilArg0 with aux %s not allowed", v.aux)
				}
			}
			if v.faultOnNilArg1 {
				fmt.Fprintln(w, "faultOnNilArg1: true,")
				if v.aux != "Sym" && v.aux != "SymOff" && v.aux != "SymValAndOff" && v.aux != "Int64" && v.aux != "Int32" && v.aux != "" {
					log.Fatalf("faultOnNilArg1 with aux %s not allowed", v.aux)
				}
			}
			if v.hasSideEffects {
				fmt.Fprintln(w, "hasSideEffects: true,")
			}
			if v.zeroWidth {
				fmt.Fprintln(w, "zeroWidth: true,")
			}
			if v.fixedReg {
				fmt.Fprintln(w, "fixedReg: true,")
			}
			if v.unsafePoint {
				fmt.Fprintln(w, "unsafePoint: true,")
			}
			needEffect := strings.HasPrefix(v.aux, "Sym")
			if v.symEffect != "" {
				if !needEffect {
					log.Fatalf("symEffect with aux %s not allowed", v.aux)
				}
				fmt.Fprintf(w, "symEffect: Sym%s,\n", strings.ReplaceAll(v.symEffect, ",", "|Sym"))
			} else if needEffect {
				log.Fatalf("symEffect needed for aux %s", v.aux)
			}
			if a.name == "generic" {
				fmt.Fprintln(w, "generic:true,")
				fmt.Fprintln(w, "},") // close op
				// generic ops have no reg info or asm
				continue
			}
			if v.asm != "" {
				fmt.Fprintf(w, "asm: %s.A%s,\n", pkg, v.asm)
			}
			if v.scale != 0 {
				fmt.Fprintf(w, "scale: %d,\n", v.scale)
			}
			fmt.Fprintln(w, "reg:regInfo{")

			// Compute input allocation order. We allocate from the
			// most to the least constrained input. This order guarantees
			// that we will always be able to find a register.
			var s []intPair
			for i, r := range v.reg.inputs {
				if r != 0 {
					s = append(s, intPair{countRegs(r), i})
				}
			}
			if len(s) > 0 {
				sort.Sort(byKey(s))
				fmt.Fprintln(w, "inputs: []inputInfo{")
				for _, p := range s {
					r := v.reg.inputs[p.val]
					fmt.Fprintf(w, "{%d,%d},%s\n", p.val, r, a.regMaskComment(r))
				}
				fmt.Fprintln(w, "},")
			}

			if v.reg.clobbers > 0 {
				fmt.Fprintf(w, "clobbers: %d,%s\n", v.reg.clobbers, a.regMaskComment(v.reg.clobbers))
			}

			// reg outputs
			s = s[:0]
			for i, r := range v.reg.outputs {
				s = append(s, intPair{countRegs(r), i})
			}
			if len(s) > 0 {
				sort.Sort(byKey(s))
				fmt.Fprintln(w, "outputs: []outputInfo{")
				for _, p := range s {
					r := v.reg.outputs[p.val]
					fmt.Fprintf(w, "{%d,%d},%s\n", p.val, r, a.regMaskComment(r))
				}
				fmt.Fprintln(w, "},")
			}
			fmt.Fprintln(w, "},") // close reg info
			fmt.Fprintln(w, "},") // close op
		}
	}
	fmt.Fprintln(w, "}")

	fmt.Fprintln(w, "func (o Op) Asm() obj.As {return opcodeTable[o].asm}")
	fmt.Fprintln(w, "func (o Op) Scale() int16 {return int16(opcodeTable[o].scale)}")

	// generate op string method
	fmt.Fprintln(w, "func (o Op) String() string {return opcodeTable[o].name }")

	fmt.Fprintln(w, "func (o Op) SymEffect() SymEffect { return opcodeTable[o].symEffect }")
	fmt.Fprintln(w, "func (o Op) IsCall() bool { return opcodeTable[o].call }")
	fmt.Fprintln(w, "func (o Op) IsTailCall() bool { return opcodeTable[o].tailCall }")
	fmt.Fprintln(w, "func (o Op) HasSideEffects() bool { return opcodeTable[o].hasSideEffects }")
	fmt.Fprintln(w, "func (o Op) UnsafePoint() bool { return opcodeTable[o].unsafePoint }")
	fmt.Fprintln(w, "func (o Op) ResultInArg0() bool { return opcodeTable[o].resultInArg0 }")

	// generate registers
	for _, a := range archs {
		if a.generic {
			continue
		}
		fmt.Fprintf(w, "var registers%s = [...]Register {\n", a.name)
		num := map[string]int8{}
		for i, r := range a.regnames {
			num[r] = int8(i)
			pkg := a.pkg[len("cmd/internal/obj/"):]
			var objname string // name in cmd/internal/obj/$ARCH
			switch r {
			case "SB":
				// SB isn't a real register.  cmd/internal/obj expects 0 in this case.
				objname = "0"
			case "SP":
				objname = pkg + ".REGSP"
			case "g":
				objname = pkg + ".REGG"
			case "ZERO":
				objname = pkg + ".REGZERO"
			default:
				objname = pkg + ".REG_" + r
			}
			fmt.Fprintf(w, "  {%d, %s, \"%s\"},\n", i, objname, r)
		}
		parameterRegisterList := func(paramNamesString string) []int8 {
			paramNamesString = strings.TrimSpace(paramNamesString)
			if paramNamesString == "" {
				return nil
			}
			paramNames := strings.Split(paramNamesString, " ")
			var paramRegs []int8
			for _, regName := range paramNames {
				if regName == "" {
					// forgive extra spaces
					continue
				}
				if regNum, ok := num[regName]; ok {
					paramRegs = append(paramRegs, regNum)
					delete(num, regName)
				} else {
					log.Fatalf("parameter register %s for architecture %s not a register name (or repeated in parameter list)", regName, a.name)
				}
			}
			return paramRegs
		}

		paramIntRegs := parameterRegisterList(a.ParamIntRegNames)
		paramFloatRegs := parameterRegisterList(a.ParamFloatRegNames)

		fmt.Fprintln(w, "}")
		fmt.Fprintf(w, "var paramIntReg%s = %#v\n", a.name, paramIntRegs)
		fmt.Fprintf(w, "var paramFloatReg%s = %#v\n", a.name, paramFloatRegs)
		fmt.Fprintf(w, "var gpRegMask%s = regMask(%d)\n", a.name, a.gpregmask)
		fmt.Fprintf(w, "var fpRegMask%s = regMask(%d)\n", a.name, a.fpregmask)
		if a.fp32regmask != 0 {
			fmt.Fprintf(w, "var fp32RegMask%s = regMask(%d)\n", a.name, a.fp32regmask)
		}
		if a.fp64regmask != 0 {
			fmt.Fprintf(w, "var fp64RegMask%s = regMask(%d)\n", a.name, a.fp64regmask)
		}
		fmt.Fprintf(w, "var specialRegMask%s = regMask(%d)\n", a.name, a.specialregmask)
		fmt.Fprintf(w, "var framepointerReg%s = int8(%d)\n", a.name, a.framepointerreg)
		fmt.Fprintf(w, "var linkReg%s = int8(%d)\n", a.name, a.linkreg)
	}

	// gofmt result
	b := w.Bytes()
	var err error
	b, err = format.Source(b)
	if err != nil {
		fmt.Printf("%s\n", w.Bytes())
		panic(err)
	}

	if err := os.WriteFile(outFile("opGen.go"), b, 0666); err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}

	// Check that the arch genfile handles all the arch-specific opcodes.
	// This is very much a hack, but it is better than nothing.
	//
	// Do a single regexp pass to record all ops being handled in a map, and
	// then compare that with the ops list. This is much faster than one
	// regexp pass per opcode.
	for _, a := range archs {
		if a.genfile == "" {
			continue
		}

		pattern := fmt.Sprintf(`\Wssa\.Op%s([a-zA-Z0-9_]+)\W`, a.name)
		rxOp, err := regexp.Compile(pattern)
		if err != nil {
			log.Fatalf("bad opcode regexp %s: %v", pattern, err)
		}

		src, err := os.ReadFile(a.genfile)
		if err != nil {
			log.Fatalf("can't read %s: %v", a.genfile, err)
		}
		seen := make(map[string]bool, len(a.ops))
		for _, m := range rxOp.FindAllSubmatch(src, -1) {
			seen[string(m[1])] = true
		}
		for _, op := range a.ops {
			if !seen[op.name] {
				log.Fatalf("Op%s%s has no code generation in %s", a.name, op.name, a.genfile)
			}
		}
	}
}

// Name returns the name of the architecture for use in Op* and Block* enumerations.
func (a arch) Name() string {
	s := a.name
	if s == "generic" {
		s = ""
	}
	return s
}

// countRegs returns the number of set bits in the register mask.
func countRegs(r regMask) int {
	return bits.OnesCount64(uint64(r))
}

// for sorting a pair of integers by key
type intPair struct {
	key, val int
}
type byKey []intPair

func (a byKey) Len() int           { return len(a) }
func (a byKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byKey) Less(i, j int) bool { return a[i].key < a[j].key }
