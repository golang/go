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
	lowerBlock      func(*Block) bool          // lowering function
	lowerValue      func(*Value, *Config) bool // lowering function
	registers       []Register                 // machine registers
	fe              Frontend                   // callbacks into compiler frontend
	HTML            *HTMLWriter                // html writer, for debugging
	ctxt            *obj.Link                  // Generic arch information
	optimize        bool                       // Do optimization
	noDuffDevice    bool                       // Don't use Duff's device
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

	// Unimplemented reports that the function cannot be compiled.
	// It will be removed once SSA work is complete.
	Unimplementedf(line int32, msg string, args ...interface{})

	// Warnl writes compiler messages in the form expected by "errorcheck" tests
	Warnl(line int32, fmt_ string, args ...interface{})

	// Fowards the Debug_checknil flag from gc
	Debug_checknil() bool
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

	// Line returns a string describing the given line number.
	Line(int32) string
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
		c.lowerBlock = rewriteBlockAMD64
		c.lowerValue = rewriteValueAMD64
		c.registers = registersAMD64[:]
	case "386":
		c.IntSize = 4
		c.PtrSize = 4
		c.lowerBlock = rewriteBlockAMD64
		c.lowerValue = rewriteValueAMD64 // TODO(khr): full 32-bit support
	case "arm":
		c.IntSize = 4
		c.PtrSize = 4
		c.lowerBlock = rewriteBlockARM
		c.lowerValue = rewriteValueARM
		c.registers = registersARM[:]
	default:
		fe.Unimplementedf(0, "arch %s not implemented", arch)
	}
	c.ctxt = ctxt
	c.optimize = optimize

	// Don't use Duff's device on Plan 9, because floating
	// point operations are not allowed in note handler.
	if obj.Getgoos() == "plan9" {
		c.noDuffDevice = true
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

func (c *Config) Frontend() Frontend      { return c.fe }
func (c *Config) SparsePhiCutoff() uint64 { return c.sparsePhiCutoff }

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
func (c *Config) Unimplementedf(line int32, msg string, args ...interface{}) {
	c.fe.Unimplementedf(line, msg, args...)
}
func (c *Config) Warnl(line int32, msg string, args ...interface{}) { c.fe.Warnl(line, msg, args...) }
func (c *Config) Debug_checknil() bool                              { return c.fe.Debug_checknil() }

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
