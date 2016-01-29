// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/internal/obj"

type Config struct {
	arch       string                     // "amd64", etc.
	IntSize    int64                      // 4 or 8
	PtrSize    int64                      // 4 or 8
	lowerBlock func(*Block) bool          // lowering function
	lowerValue func(*Value, *Config) bool // lowering function
	fe         Frontend                   // callbacks into compiler frontend
	HTML       *HTMLWriter                // html writer, for debugging
	ctxt       *obj.Link                  // Generic arch information
	optimize   bool                       // Do optimization
	curFunc    *Func

	// TODO: more stuff.  Compiler flags of interest, ...

	// Storage for low-numbered values and blocks.
	values [2000]Value
	blocks [200]Block
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
	Warnl(line int, fmt_ string, args ...interface{})

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

	// Line returns a string describing the given line number.
	Line(int32) string
}

// interface used to hold *gc.Node.  We'd use *gc.Node directly but
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
	case "386":
		c.IntSize = 4
		c.PtrSize = 4
		c.lowerBlock = rewriteBlockAMD64
		c.lowerValue = rewriteValueAMD64 // TODO(khr): full 32-bit support
	default:
		fe.Unimplementedf(0, "arch %s not implemented", arch)
	}
	c.ctxt = ctxt
	c.optimize = optimize

	// Assign IDs to preallocated values/blocks.
	for i := range c.values {
		c.values[i].ID = ID(i)
	}
	for i := range c.blocks {
		c.blocks[i].ID = ID(i)
	}

	return c
}

func (c *Config) Frontend() Frontend { return c.fe }

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
func (c *Config) Warnl(line int, msg string, args ...interface{}) { c.fe.Warnl(line, msg, args...) }
func (c *Config) Debug_checknil() bool                            { return c.fe.Debug_checknil() }
