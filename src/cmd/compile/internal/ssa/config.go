// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

type Config struct {
	arch       string                     // "amd64", etc.
	IntSize    int64                      // 4 or 8
	PtrSize    int64                      // 4 or 8
	lowerBlock func(*Block) bool          // lowering function
	lowerValue func(*Value, *Config) bool // lowering function
	fe         Frontend                   // callbacks into compiler frontend
	HTML       *HTMLWriter                // html writer, for debugging

	// TODO: more stuff.  Compiler flags of interest, ...
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
	TypeUintptr() Type
	TypeString() Type
	TypeBytePtr() Type // TODO: use unsafe.Pointer instead?
}

type Logger interface {
	// Log logs a message from the compiler.
	Logf(string, ...interface{})

	// Fatal reports a compiler error and exits.
	Fatalf(string, ...interface{})

	// Unimplemented reports that the function cannot be compiled.
	// It will be removed once SSA work is complete.
	Unimplementedf(msg string, args ...interface{})
}

type Frontend interface {
	TypeSource
	Logger

	// StringData returns a symbol pointing to the given string's contents.
	StringData(string) interface{} // returns *gc.Sym
}

// NewConfig returns a new configuration object for the given architecture.
func NewConfig(arch string, fe Frontend) *Config {
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
		fe.Unimplementedf("arch %s not implemented", arch)
	}

	return c
}

func (c *Config) Frontend() Frontend { return c.fe }

// NewFunc returns a new, empty function object
func (c *Config) NewFunc() *Func {
	// TODO(khr): should this function take name, type, etc. as arguments?
	return &Func{Config: c}
}

func (c *Config) Logf(msg string, args ...interface{})           { c.fe.Logf(msg, args...) }
func (c *Config) Fatalf(msg string, args ...interface{})         { c.fe.Fatalf(msg, args...) }
func (c *Config) Unimplementedf(msg string, args ...interface{}) { c.fe.Unimplementedf(msg, args...) }

// TODO(khr): do we really need a separate Config, or can we just
// store all its fields inside a Func?
