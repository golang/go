// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj"
	"crypto/sha1"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	arch            string                     // "amd64", etc.
	IntSize         int64                      // 4 or 8
	PtrSize         int64                      // 4 or 8
	RegSize         int64                      // 4 or 8
	lowerBlock      func(*Block, *Config) bool // lowering function
	lowerValue      func(*Value, *Config) bool // lowering function
	registers       []Register                 // machine registers
	gpRegMask       regMask                    // general purpose integer register mask
	fpRegMask       regMask                    // floating point register mask
	specialRegMask  regMask                    // special register mask
	FPReg           int8                       // register number of frame pointer, -1 if not used
	LinkReg         int8                       // register number of link register if it is a general purpose register, -1 if not used
	hasGReg         bool                       // has hardware g register
	fe              Frontend                   // callbacks into compiler frontend
	HTML            *HTMLWriter                // html writer, for debugging
	ctxt            *obj.Link                  // Generic arch information
	optimize        bool                       // Do optimization
	noDuffDevice    bool                       // Don't use Duff's device
	nacl            bool                       // GOOS=nacl
	use387          bool                       // GO386=387
	OldArch         bool                       // True for older versions of architecture, e.g. true for PPC64BE, false for PPC64LE
	NeedsFpScratch  bool                       // No direct move between GP and FP register sets
	BigEndian       bool                       //
	DebugTest       bool                       // default true unless $GOSSAHASH != ""; as a debugging aid, make new code conditional on this and use GOSSAHASH to binary search for failing cases
	sparsePhiCutoff uint64                     // Sparse phi location algorithm used above this #blocks*#variables score
	curFunc         *Func

	// TODO: more stuff. Compiler flags of interest, ...

	// Given an environment variable used for debug hash match,
	// what file (if any) receives the yes/no logging?
	logfiles map[string]*os.File

	// Storage for low-numbered values and blocks.
	values [2000]Value
	blocks [200]Block

	// Reusable stackAllocState.
	// See stackalloc.go's {new,put}StackAllocState.
	stackAllocState *stackAllocState

	domblockstore []ID         // scratch space for computing dominators
	scrSparse     []*sparseSet // scratch sparse sets to be re-used.
}

type TypeSource interface {
	TypeBool() Type
	TypeInt8() Type
	TypeInt16() Type
	TypeInt32() Type
	TypeInt64() Type
	TypeUInt8() Type
	TypeUInt16() Type
	TypeUInt32() Type
	TypeUInt64() Type
	TypeInt() Type
	TypeFloat32() Type
	TypeFloat64() Type
	TypeUintptr() Type
	TypeString() Type
	TypeBytePtr() Type // TODO: use unsafe.Pointer instead?

	CanSSA(t Type) bool
}

type Logger interface {
	// Logf logs a message from the compiler.
	Logf(string, ...interface{})

	// Log returns true if logging is not a no-op
	// some logging calls account for more than a few heap allocations.
	Log() bool

	// Fatal reports a compiler error and exits.
	Fatalf(line int32, msg string, args ...interface{})

	// Warnl writes compiler messages in the form expected by "errorcheck" tests
	Warnl(line int32, fmt_ string, args ...interface{})

	// Forwards the Debug flags from gc
	Debug_checknil() bool
	Debug_wb() bool
}

type Frontend interface {
	TypeSource
	Logger

	// StringData returns a symbol pointing to the given string's contents.
	StringData(string) interface{} // returns *gc.Sym

	// Auto returns a Node for an auto variable of the given type.
	// The SSA compiler uses this function to allocate space for spills.
	Auto(Type) GCNode

	// Given the name for a compound type, returns the name we should use
	// for the parts of that compound type.
	SplitString(LocalSlot) (LocalSlot, LocalSlot)
	SplitInterface(LocalSlot) (LocalSlot, LocalSlot)
	SplitSlice(LocalSlot) (LocalSlot, LocalSlot, LocalSlot)
	SplitComplex(LocalSlot) (LocalSlot, LocalSlot)
	SplitStruct(LocalSlot, int) LocalSlot
	SplitArray(LocalSlot) LocalSlot              // array must be length 1
	SplitInt64(LocalSlot) (LocalSlot, LocalSlot) // returns (hi, lo)

	// Line returns a string describing the given line number.
	Line(int32) string

	// AllocFrame assigns frame offsets to all live auto variables.
	AllocFrame(f *Func)

	// Syslook returns a symbol of the runtime function/variable with the
	// given name.
	Syslook(string) interface{} // returns *gc.Sym
}

// interface used to hold *gc.Node. We'd use *gc.Node directly but
// that would lead to an import cycle.
type GCNode interface {
	Typ() Type
	String() string
}

// NewConfig returns a new configuration object for the given architecture.
func NewConfig(arch string, fe Frontend, ctxt *obj.Link, optimize bool) *Config {
	c := &Config{arch: arch, fe: fe}
	switch arch {
	case "amd64":
		c.IntSize = 8
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockAMD64
		c.lowerValue = rewriteValueAMD64
		c.registers = registersAMD64[:]
		c.gpRegMask = gpRegMaskAMD64
		c.fpRegMask = fpRegMaskAMD64
		c.FPReg = framepointerRegAMD64
		c.LinkReg = linkRegAMD64
		c.hasGReg = false
	case "amd64p32":
		c.IntSize = 4
		c.PtrSize = 4
		c.RegSize = 8
		c.lowerBlock = rewriteBlockAMD64
		c.lowerValue = rewriteValueAMD64
		c.registers = registersAMD64[:]
		c.gpRegMask = gpRegMaskAMD64
		c.fpRegMask = fpRegMaskAMD64
		c.FPReg = framepointerRegAMD64
		c.LinkReg = linkRegAMD64
		c.hasGReg = false
		c.noDuffDevice = true
	case "386":
		c.IntSize = 4
		c.PtrSize = 4
		c.RegSize = 4
		c.lowerBlock = rewriteBlock386
		c.lowerValue = rewriteValue386
		c.registers = registers386[:]
		c.gpRegMask = gpRegMask386
		c.fpRegMask = fpRegMask386
		c.FPReg = framepointerReg386
		c.LinkReg = linkReg386
		c.hasGReg = false
	case "arm":
		c.IntSize = 4
		c.PtrSize = 4
		c.RegSize = 4
		c.lowerBlock = rewriteBlockARM
		c.lowerValue = rewriteValueARM
		c.registers = registersARM[:]
		c.gpRegMask = gpRegMaskARM
		c.fpRegMask = fpRegMaskARM
		c.FPReg = framepointerRegARM
		c.LinkReg = linkRegARM
		c.hasGReg = true
	case "arm64":
		c.IntSize = 8
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockARM64
		c.lowerValue = rewriteValueARM64
		c.registers = registersARM64[:]
		c.gpRegMask = gpRegMaskARM64
		c.fpRegMask = fpRegMaskARM64
		c.FPReg = framepointerRegARM64
		c.LinkReg = linkRegARM64
		c.hasGReg = true
		c.noDuffDevice = obj.GOOS == "darwin" // darwin linker cannot handle BR26 reloc with non-zero addend
	case "ppc64":
		c.OldArch = true
		c.BigEndian = true
		fallthrough
	case "ppc64le":
		c.IntSize = 8
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockPPC64
		c.lowerValue = rewriteValuePPC64
		c.registers = registersPPC64[:]
		c.gpRegMask = gpRegMaskPPC64
		c.fpRegMask = fpRegMaskPPC64
		c.FPReg = framepointerRegPPC64
		c.LinkReg = linkRegPPC64
		c.noDuffDevice = true // TODO: Resolve PPC64 DuffDevice (has zero, but not copy)
		c.NeedsFpScratch = true
		c.hasGReg = true
	case "mips64":
		c.BigEndian = true
		fallthrough
	case "mips64le":
		c.IntSize = 8
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockMIPS64
		c.lowerValue = rewriteValueMIPS64
		c.registers = registersMIPS64[:]
		c.gpRegMask = gpRegMaskMIPS64
		c.fpRegMask = fpRegMaskMIPS64
		c.specialRegMask = specialRegMaskMIPS64
		c.FPReg = framepointerRegMIPS64
		c.LinkReg = linkRegMIPS64
		c.hasGReg = true
	case "s390x":
		c.IntSize = 8
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockS390X
		c.lowerValue = rewriteValueS390X
		c.registers = registersS390X[:]
		c.gpRegMask = gpRegMaskS390X
		c.fpRegMask = fpRegMaskS390X
		c.FPReg = framepointerRegS390X
		c.LinkReg = linkRegS390X
		c.hasGReg = true
		c.noDuffDevice = true
		c.BigEndian = true
	case "mips":
		c.BigEndian = true
		fallthrough
	case "mipsle":
		c.IntSize = 4
		c.PtrSize = 4
		c.RegSize = 4
		c.lowerBlock = rewriteBlockMIPS
		c.lowerValue = rewriteValueMIPS
		c.registers = registersMIPS[:]
		c.gpRegMask = gpRegMaskMIPS
		c.fpRegMask = fpRegMaskMIPS
		c.specialRegMask = specialRegMaskMIPS
		c.FPReg = framepointerRegMIPS
		c.LinkReg = linkRegMIPS
		c.hasGReg = true
		c.noDuffDevice = true
	default:
		fe.Fatalf(0, "arch %s not implemented", arch)
	}
	c.ctxt = ctxt
	c.optimize = optimize
	c.nacl = obj.GOOS == "nacl"

	// Don't use Duff's device on Plan 9 AMD64, because floating
	// point operations are not allowed in note handler.
	if obj.GOOS == "plan9" && arch == "amd64" {
		c.noDuffDevice = true
	}

	if c.nacl {
		c.noDuffDevice = true // Don't use Duff's device on NaCl

		// runtime call clobber R12 on nacl
		opcodeTable[OpARMUDIVrtcall].reg.clobbers |= 1 << 12 // R12
	}

	// Assign IDs to preallocated values/blocks.
	for i := range c.values {
		c.values[i].ID = ID(i)
	}
	for i := range c.blocks {
		c.blocks[i].ID = ID(i)
	}

	c.logfiles = make(map[string]*os.File)

	// cutoff is compared with product of numblocks and numvalues,
	// if product is smaller than cutoff, use old non-sparse method.
	// cutoff == 0 implies all sparse.
	// cutoff == -1 implies none sparse.
	// Good cutoff values seem to be O(million) depending on constant factor cost of sparse.
	// TODO: get this from a flag, not an environment variable
	c.sparsePhiCutoff = 2500000 // 0 for testing. // 2500000 determined with crude experiments w/ make.bash
	ev := os.Getenv("GO_SSA_PHI_LOC_CUTOFF")
	if ev != "" {
		v, err := strconv.ParseInt(ev, 10, 64)
		if err != nil {
			fe.Fatalf(0, "Environment variable GO_SSA_PHI_LOC_CUTOFF (value '%s') did not parse as a number", ev)
		}
		c.sparsePhiCutoff = uint64(v) // convert -1 to maxint, for never use sparse
	}

	return c
}

func (c *Config) Set387(b bool) {
	c.NeedsFpScratch = b
	c.use387 = b
}

func (c *Config) Frontend() Frontend      { return c.fe }
func (c *Config) SparsePhiCutoff() uint64 { return c.sparsePhiCutoff }
func (c *Config) Ctxt() *obj.Link         { return c.ctxt }

// NewFunc returns a new, empty function object.
// Caller must call f.Free() before calling NewFunc again.
func (c *Config) NewFunc() *Func {
	// TODO(khr): should this function take name, type, etc. as arguments?
	if c.curFunc != nil {
		c.Fatalf(0, "NewFunc called without previous Free")
	}
	f := &Func{Config: c, NamedValues: map[LocalSlot][]*Value{}}
	c.curFunc = f
	return f
}

func (c *Config) Logf(msg string, args ...interface{})               { c.fe.Logf(msg, args...) }
func (c *Config) Log() bool                                          { return c.fe.Log() }
func (c *Config) Fatalf(line int32, msg string, args ...interface{}) { c.fe.Fatalf(line, msg, args...) }
func (c *Config) Warnl(line int32, msg string, args ...interface{})  { c.fe.Warnl(line, msg, args...) }
func (c *Config) Debug_checknil() bool                               { return c.fe.Debug_checknil() }
func (c *Config) Debug_wb() bool                                     { return c.fe.Debug_wb() }

func (c *Config) logDebugHashMatch(evname, name string) {
	file := c.logfiles[evname]
	if file == nil {
		file = os.Stdout
		tmpfile := os.Getenv("GSHS_LOGFILE")
		if tmpfile != "" {
			var ok error
			file, ok = os.Create(tmpfile)
			if ok != nil {
				c.Fatalf(0, "Could not open hash-testing logfile %s", tmpfile)
			}
		}
		c.logfiles[evname] = file
	}
	s := fmt.Sprintf("%s triggered %s\n", evname, name)
	file.WriteString(s)
	file.Sync()
}

// DebugHashMatch returns true if environment variable evname
// 1) is empty (this is a special more-quickly implemented case of 3)
// 2) is "y" or "Y"
// 3) is a suffix of the sha1 hash of name
// 4) is a suffix of the environment variable
//    fmt.Sprintf("%s%d", evname, n)
//    provided that all such variables are nonempty for 0 <= i <= n
// Otherwise it returns false.
// When true is returned the message
//  "%s triggered %s\n", evname, name
// is printed on the file named in environment variable
//  GSHS_LOGFILE
// or standard out if that is empty or there is an error
// opening the file.

func (c *Config) DebugHashMatch(evname, name string) bool {
	evhash := os.Getenv(evname)
	if evhash == "" {
		return true // default behavior with no EV is "on"
	}
	if evhash == "y" || evhash == "Y" {
		c.logDebugHashMatch(evname, name)
		return true
	}
	if evhash == "n" || evhash == "N" {
		return false
	}
	// Check the hash of the name against a partial input hash.
	// We use this feature to do a binary search to
	// find a function that is incorrectly compiled.
	hstr := ""
	for _, b := range sha1.Sum([]byte(name)) {
		hstr += fmt.Sprintf("%08b", b)
	}

	if strings.HasSuffix(hstr, evhash) {
		c.logDebugHashMatch(evname, name)
		return true
	}

	// Iteratively try additional hashes to allow tests for multi-point
	// failure.
	for i := 0; true; i++ {
		ev := fmt.Sprintf("%s%d", evname, i)
		evv := os.Getenv(ev)
		if evv == "" {
			break
		}
		if strings.HasSuffix(hstr, evv) {
			c.logDebugHashMatch(ev, name)
			return true
		}
	}
	return false
}

func (c *Config) DebugNameMatch(evname, name string) bool {
	return os.Getenv(evname) == name
}
