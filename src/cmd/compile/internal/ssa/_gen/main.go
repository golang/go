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
	"path/filepath"
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

var splitPhase = phase6Rewrites

var splitTitle = identity

var splitOpPkg = "ssa"
var splitOpFile = "opGen.go"
var splitOpPrefix = ""

var splitCorePkg = "ssa"
var splitCorePath = "cmd/compile/internal/ssa"
var allocatorsFile = "allocators.go"
var splitCorePrefix = ""

var splitRewritesDir = ""
var splitRewritesPkg = "ssa"

func rewritesPkg(arch, suff string) string {
	if splitPhase < phase6Rewrites {
		return splitRewritesPkg
	}
	// Add rewrite to the beginning so we don't get
	// a package name starting with a number in the case of 386.
	name := strings.ToLower("rewrite" + arch + suff)
	return name
}

func rewritesDir(arch, suff string) string {
	if splitPhase < phase6Rewrites {
		return splitRewritesDir
	}
	return "../ssarewrite/" + rewritesPkg(arch, suff) + "/"
}

func rewriteFuncName(kind, arch, suff, rule string) string {
	if splitPhase < phase6Rewrites {
		return "rewrite" + kind + arch + suff + rule
	}
	if rule == "" {
		// Top level Value or Block rule. Exported.
		return "Rewrite" + kind
	}
	return "rewrite" + kind + rule
}

var registersFile = "opGen.go"
var registersPkg = "ssa"

func identity(s string) string {
	return s
}
func simpleTitle(s string) string { return strings.ToUpper(s[:1]) + s[1:] } // Unlike title, only title the first letter of the string.

const (
	phase0Start = iota
	phase0Export
	phase1Op
	phase2Core
	phase3Compile
	phase4CoreRename
	phase5Conv
	phase6Rewrites
)

func init() {
	if splitPhase >= phase0Export {
		splitTitle = simpleTitle
	}
	if splitPhase >= phase1Op {
		splitOpPkg = "ssaop"
		splitOpPrefix = "ssaop."
		splitOpFile = "ssaop/opGen.go"
	}
	if splitPhase >= phase2Core {
		splitCorePrefix = "ssacore."
		splitCorePkg = "ssacore"
		splitCorePath = "cmd/compile/internal/ssa/ssacore"
		allocatorsFile = filepath.Join(splitCorePkg, "allocators.go")
	}
	if splitPhase >= phase3Compile {
		splitRewritesDir = "../ssacompile/"
		splitRewritesPkg = "ssacompile"
		registersFile = splitRewritesDir + registersFile
		registersPkg = splitRewritesPkg
	}
	if splitPhase >= phase4CoreRename {
		splitCorePrefix = "ssa."
		splitCorePkg = "ssa"
		splitCorePath = "cmd/compile/internal/ssa"
		allocatorsFile = "allocators.go"
	}
}

type arch struct {
	name               string
	pkg                string // obj package to import for this arch.
	genfile            string // source file containing opcode code generation.
	genSIMDfile        string // source file containing opcode code generation for SIMD.
	ops                []opData
	blocks             []blockData
	regnames           []string
	ParamIntRegNames   string
	ParamFloatRegNames string
	gpregmask          regMask
	fpregmask          regMask
	fp32regmask        regMask
	fp64regmask        regMask
	simdregmask        regMask
	specialregmask     regMask
	framepointerreg    int8
	linkreg            int8
	generic            bool
	imports            []string
}

type comparableOpData struct {
	name              string
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
	earlyOk           bool   // executing this op in an earlier block is ok
	addrSinkArg0      bool   // the address in arg0 does not propagate to the result
	addrSinkArg1      bool   // the address in arg1 does not propagate to the result
	symEffect         string // effect this op has on symbol in aux
	scale             uint8  // amd64/386 indexed load scale
	zeroUpperBits     uint8  // the op writes a 64-bit GPR whose upper N bits are always zero (0, 32, 48 or 56); for a tuple op, this holds for every integer result
}

type opData struct {
	reg regInfo
	comparableOpData
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
	// Instruction clobbers the register containing input 0.
	clobbersArg0 bool
	// Instruction clobbers the register containing input 1.
	clobbersArg1 bool
	// outputs[i] encodes the set of registers allowed for the i'th output.
	outputs []regMask
}

type regMask struct {
	v1, v2 uint64
}

func regMaskAt(i uint) regMask {
	if i < 64 {
		return regMask{v1: 1 << i}
	}
	return regMask{v2: 1 << (i - 64)}
}

func (r regMask) empty() bool {
	return r.v1 == 0 && r.v2 == 0
}

func (r regMask) hasReg(i uint) bool {
	if i < 64 {
		return (r.v1>>i)&1 != 0
	}
	return (r.v2>>(i-64))&1 != 0
}

func (r regMask) addReg(i uint) regMask {
	if i < 64 {
		return regMask{r.v1 | 1<<i, r.v2}
	}
	return regMask{r.v1, r.v2 | 1<<(i-64)}
}

func (r regMask) union(s regMask) regMask {
	return regMask{r.v1 | s.v1, r.v2 | s.v2}
}

func (r regMask) minus(s regMask) regMask {
	return regMask{r.v1 &^ s.v1, r.v2 &^ s.v2}
}

func (a arch) regMaskComment(r regMask) string {
	var buf strings.Builder
	for i := uint(0); i < uint(len(a.regnames)); i++ {
		if r.hasReg(i) {
			if buf.Len() == 0 {
				buf.WriteString(" //")
			}
			buf.WriteString(" ")
			buf.WriteString(a.regnames[i])
		}
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

	// call this late so that all genericOps contributors have run their init functions.
	genericInit()

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

func mkdirOutFile(file string) {
	if err := os.MkdirAll(filepath.Dir(outFile(file)), 0777); err != nil {
		log.Fatalf("can't create output directory for %s: %v", file, err)
	}
}

func genOp() {
	w := new(bytes.Buffer)
	fmt.Fprintf(w, "// Code generated from _gen/*Ops.go using 'go generate'; DO NOT EDIT.\n")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "package block")

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

	// gofmt result
	blockb := w.Bytes()
	var blockerr error
	blockb, blockerr = format.Source(blockb)
	if blockerr != nil {
		fmt.Printf("%s\n", w.Bytes())
		panic(blockerr)
	}

	if err := os.MkdirAll(outFile("block"), 0777); err != nil {
		log.Fatal("can't create block directory")
	}
	if err := os.WriteFile(outFile("block/opGen.go"), blockb, 0666); err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}

	var opBuf bytes.Buffer
	w = &opBuf
	fmt.Fprintf(w, "// Code generated from _gen/*Ops.go using 'go generate'; DO NOT EDIT.\n")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "package %s\n", splitOpPkg)

	fmt.Fprintln(w, "import (")
	if splitPhase < phase1Op {
		fmt.Fprintln(w, `"cmd/compile/internal/ssa/ssabase"`)
	}
	fmt.Fprintln(w, `"cmd/internal/obj"`)
	for _, a := range archs {
		if a.pkg != "" {
			fmt.Fprintf(w, "%q\n", a.pkg)
		}
	}
	fmt.Fprintln(w, ")")
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

	auxPrefix := "aux"
	if splitPhase >= phase0Export {
		auxPrefix = "AuxType"
	}
	// generate OpInfo table
	fmt.Fprintf(w, "var %s = [...]%s{\n", splitTitle("opcodeTable"), splitTitle("opInfo"))
	fmt.Fprintf(w, " { %s: \"OpInvalid\" },\n", splitTitle("name"))
	for _, a := range archs {
		fmt.Fprintln(w)

		pkg := path.Base(a.pkg)
		for _, v := range a.ops {
			if v.name == "Invalid" {
				continue
			}
			fmt.Fprintln(w, "{")
			fmt.Fprintf(w, "%s:\"%s\",\n", splitTitle("name"), v.name)

			// flags
			if v.aux != "" {
				fmt.Fprintf(w, "%s: %s%s,\n", splitTitle("auxType"), auxPrefix, splitTitle(v.aux))
			}
			fmt.Fprintf(w, "%s: %d,\n", splitTitle("argLen"), v.argLength)

			if v.rematerializeable {
				if !v.reg.clobbers.empty() || v.reg.clobbersArg0 || v.reg.clobbersArg1 {
					log.Fatalf("%s is rematerializeable and clobbers registers", v.name)
				}
				if v.clobberFlags {
					log.Fatalf("%s is rematerializeable and clobbers flags", v.name)
				}
				fmt.Fprintln(w, splitTitle("rematerializeable: true,"))
			}
			if v.commutative {
				fmt.Fprintln(w, splitTitle("commutative: true,"))
			}
			if v.resultInArg0 {
				fmt.Fprintln(w, splitTitle("resultInArg0: true,"))
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
				fmt.Fprintln(w, splitTitle("resultNotInArgs: true,"))
			}
			if v.clobberFlags {
				fmt.Fprintln(w, splitTitle("clobberFlags: true,"))
			}
			if v.needIntTemp {
				fmt.Fprintln(w, splitTitle("needIntTemp: true,"))
			}
			if v.call {
				fmt.Fprintln(w, splitTitle("call: true,"))
			}
			if v.tailCall {
				fmt.Fprintln(w, "tailCall: true,")
			}
			if v.nilCheck {
				fmt.Fprintln(w, splitTitle("nilCheck: true,"))
			}
			if v.faultOnNilArg0 {
				fmt.Fprintln(w, splitTitle("faultOnNilArg0: true,"))
				if v.aux != "Sym" && v.aux != "SymOff" && v.aux != "SymValAndOff" && v.aux != "Int64" && v.aux != "Int32" && v.aux != "" {
					log.Fatalf("faultOnNilArg0 with aux %s not allowed", v.aux)
				}
			}
			if v.faultOnNilArg1 {
				fmt.Fprintln(w, splitTitle("faultOnNilArg1: true,"))
				if v.aux != "Sym" && v.aux != "SymOff" && v.aux != "SymValAndOff" && v.aux != "Int64" && v.aux != "Int32" && v.aux != "" {
					log.Fatalf("faultOnNilArg1 with aux %s not allowed", v.aux)
				}
			}
			if v.hasSideEffects {
				fmt.Fprintln(w, splitTitle("hasSideEffects: true,"))
			}
			if v.zeroWidth {
				fmt.Fprintln(w, splitTitle("zeroWidth: true,"))
			}
			if v.fixedReg {
				fmt.Fprintln(w, splitTitle("fixedReg: true,"))
			}
			if v.earlyOk {
				fmt.Fprintln(w, splitTitle("earlyOk: true,"))
			}
			if v.addrSinkArg0 {
				fmt.Fprintln(w, splitTitle("addrSinkArg0: true,"))
			}
			if v.addrSinkArg1 {
				fmt.Fprintln(w, splitTitle("addrSinkArg1: true,"))
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
				fmt.Fprintln(w, splitTitle("generic:true,"))
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
			if v.zeroUpperBits != 0 {
				switch v.zeroUpperBits {
				case 32, 48, 56:
				default:
					log.Fatalf("%s: zeroUpperBits must be 0, 32, 48 or 56, have %d", v.name, v.zeroUpperBits)
				}
				fmt.Fprintf(w, "%s: %d,\n", splitTitle("zeroUpperBits"), v.zeroUpperBits)
			}
			fmt.Fprintf(w, "%s:%s{\n", splitTitle("reg"), splitTitle("regInfo"))

			// Compute input allocation order. We allocate from the
			// most to the least constrained input. This order guarantees
			// that we will always be able to find a register.
			var s []intPair
			for i, r := range v.reg.inputs {
				if !r.empty() {
					s = append(s, intPair{countRegs(r), i})
				}
			}
			if len(s) > 0 {
				sort.Sort(byKey(s))
				fmt.Fprintf(w, "%s: []%s{\n", splitTitle("inputs"), splitTitle("inputInfo"))
				for _, p := range s {
					r := v.reg.inputs[p.val]
					fmt.Fprintf(w, "{%d,%s{%s: %d, %s: %d}},%s\n", p.val, splitTitle("regMask"), splitTitle("v1"), r.v1, splitTitle("v2"), r.v2, a.regMaskComment(r))
				}
				fmt.Fprintln(w, "},")
			}

			if !v.reg.clobbers.empty() {
				fmt.Fprintf(w, "%s: %s{%s: %d, %s: %d},%s\n", splitTitle("clobbers"), splitTitle("regMask"), splitTitle("v1"), v.reg.clobbers.v1, splitTitle("v2"), v.reg.clobbers.v2, a.regMaskComment(v.reg.clobbers))
			}
			if v.reg.clobbersArg0 {
				fmt.Fprintf(w, "%s: true,\n", splitTitle("clobbersArg0"))
			}
			if v.reg.clobbersArg1 {
				fmt.Fprintf(w, "%s: true,\n", splitTitle("clobbersArg1"))
			}

			// reg outputs
			s = s[:0]
			for i, r := range v.reg.outputs {
				s = append(s, intPair{countRegs(r), i})
			}
			if len(s) > 0 {
				sort.Sort(byKey(s))
				fmt.Fprintf(w, "%s: []%s{\n", splitTitle("outputs"), splitTitle("outputInfo"))
				for _, p := range s {
					r := v.reg.outputs[p.val]
					fmt.Fprintf(w, "{%d,%s{%s: %d, %s: %d}},%s\n", p.val, splitTitle("regMask"), splitTitle("v1"), r.v1, splitTitle("v2"), r.v2, a.regMaskComment(r))
				}
				fmt.Fprintln(w, "},")
			}
			fmt.Fprintln(w, "},") // close reg info
			fmt.Fprintln(w, "},") // close op
		}
	}
	fmt.Fprintln(w, "}")

	fmt.Fprintf(w, "func (o Op) Asm() obj.As {return %s[o].%s}\n", splitTitle("opcodeTable"), "asm")
	fmt.Fprintf(w, "func (o Op) Scale() int16 {return int16(%s[o].%s)}\n", splitTitle("opcodeTable"), "scale")

	// generate op string method
	fmt.Fprintf(w, "func (o Op) String() string {return %s[o].%s }\n", splitTitle("opcodeTable"), splitTitle("name"))

	fmt.Fprintf(w, "func (o Op) SymEffect() SymEffect { return %s[o].symEffect }\n", splitTitle("opcodeTable"))
	fmt.Fprintf(w, "func (o Op) IsCall() bool { return %s[o].%s }\n", splitTitle("opcodeTable"), splitTitle("call"))
	fmt.Fprintf(w, "func (o Op) IsTailCall() bool { return %s[o].tailCall }\n", splitTitle("opcodeTable"))
	fmt.Fprintf(w, "func (o Op) HasSideEffects() bool { return %s[o].%s }\n", splitTitle("opcodeTable"), splitTitle("hasSideEffects"))
	fmt.Fprintf(w, "func (o Op) UnsafePoint() bool { return %s[o].unsafePoint }\n", splitTitle("opcodeTable"))
	fmt.Fprintf(w, "func (o Op) ResultInArg0() bool { return %s[o].%s }\n", splitTitle("opcodeTable"), splitTitle("resultInArg0"))

	var registersBuf bytes.Buffer
	if registersFile != splitOpFile {
		w = &registersBuf

		fmt.Fprintf(w, "// Code generated from _gen/*Ops.go using 'go generate'; DO NOT EDIT.\n")
		fmt.Fprintln(w)
		fmt.Fprintln(w, "package "+registersPkg)

		fmt.Fprintln(w, "import (")
		fmt.Fprintln(w, `"cmd/compile/internal/ssa/ssabase"`)
		fmt.Fprintln(w, `"cmd/compile/internal/ssa/ssaop"`)
		for _, a := range archs {
			if a.pkg != "" {
				fmt.Fprintf(w, "%q\n", a.pkg)
			}
		}
		fmt.Fprintln(w, ")")
	}

	// generate registers
	for _, a := range archs {
		if a.generic {
			continue
		}
		fmt.Fprintf(w, "var registers%s = [...]ssabase.Register {\n", a.name)
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
			fmt.Fprintf(w, "  {Num: %d, ObjNum: %s, Name: \"%s\"},\n", i, objname, r)
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
		fmt.Fprintf(w, "var gpRegMask%s = %s{%s: %d, %s: %d}\n", a.name, splitOpPrefix+splitTitle("regMask"), splitTitle("v1"), a.gpregmask.v1, splitTitle("v2"), a.gpregmask.v2)
		fmt.Fprintf(w, "var fpRegMask%s = %s{%s: %d, %s: %d}\n", a.name, splitOpPrefix+splitTitle("regMask"), splitTitle("v1"), a.fpregmask.v1, splitTitle("v2"), a.fpregmask.v2)
		if !a.fp32regmask.empty() {
			fmt.Fprintf(w, "var fp32RegMask%s = %s{%s: %d, %s: %d}\n", a.name, splitOpPrefix+splitTitle("regMask"), splitTitle("v1"), a.fp32regmask.v1, splitTitle("v2"), a.fp32regmask.v2)
		}
		if !a.fp64regmask.empty() {
			fmt.Fprintf(w, "var fp64RegMask%s = %s{%s: %d, %s: %d}\n", a.name, splitOpPrefix+splitTitle("regMask"), splitTitle("v1"), a.fp64regmask.v1, splitTitle("v2"), a.fp64regmask.v2)
		}
		if !a.simdregmask.empty() {
			fmt.Fprintf(w, "var simdRegMask%s = %s{%s: %d, %s: %d}\n", a.name, splitOpPrefix+splitTitle("regMask"), splitTitle("v1"), a.simdregmask.v1, splitTitle("v2"), a.simdregmask.v2)
		}
		fmt.Fprintf(w, "var specialRegMask%s = %s{%s: %d, %s: %d}\n", a.name, splitOpPrefix+splitTitle("regMask"), splitTitle("v1"), a.specialregmask.v1, splitTitle("v2"), a.specialregmask.v2)
		fmt.Fprintf(w, "var framepointerReg%s = int8(%d)\n", a.name, a.framepointerreg)
		fmt.Fprintf(w, "var linkReg%s = int8(%d)\n", a.name, a.linkreg)
	}

	// gofmt result
	b := opBuf.Bytes()
	var err error
	b, err = format.Source(b)
	if err != nil {
		fmt.Printf("%s\n", w.Bytes())
		panic(err)
	}

	mkdirOutFile(splitOpFile)
	if err := os.WriteFile(outFile(splitOpFile), b, 0666); err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}

	if registersFile != splitOpFile {
		b := registersBuf.Bytes()
		var err error
		b, err = format.Source(b)
		if err != nil {
			fmt.Printf("%s\n", w.Bytes())
			panic(err)
		}

		mkdirOutFile(registersFile)
		if err := os.WriteFile(outFile(registersFile), b, 0666); err != nil {
			log.Fatalf("can't write output: %v\n", err)
		}
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

		pattern := fmt.Sprintf(`\W`+splitOpPkg+`\.Op%s([a-zA-Z0-9_]+)\W`, a.name)
		rxOp, err := regexp.Compile(pattern)
		if err != nil {
			log.Fatalf("bad opcode regexp %s: %v", pattern, err)
		}

		src, err := os.ReadFile(a.genfile)
		if err != nil {
			log.Fatalf("can't read %s: %v", a.genfile, err)
		}
		// Append the file(s) of simd operations, too. genSIMDfile may list
		// several space-separated paths (e.g. NEON plus SVE for ARM64).
		if a.genSIMDfile != "" {
			for _, f := range strings.Fields(a.genSIMDfile) {
				simdSrc, err := os.ReadFile(f)
				if err != nil {
					log.Fatalf("can't read %s: %v", f, err)
				}
				src = append(src, simdSrc...)
			}
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
	return bits.OnesCount64(r.v1) + bits.OnesCount64(r.v2)
}

// for sorting a pair of integers by key
type intPair struct {
	key, val int
}
type byKey []intPair

func (a byKey) Len() int           { return len(a) }
func (a byKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byKey) Less(i, j int) bool { return a[i].key < a[j].key }
