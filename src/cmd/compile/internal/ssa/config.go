// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "log"

type Config struct {
	arch       string                     // "amd64", etc.
	ptrSize    int64                      // 4 or 8
	Uintptr    Type                       // pointer arithmetic type
	lowerBlock func(*Block) bool          // lowering function
	lowerValue func(*Value, *Config) bool // lowering function
	fe         Frontend                   // callbacks into compiler frontend

	// TODO: more stuff.  Compiler flags of interest, ...
}

type Frontend interface {
	// StringSym returns a symbol pointing to the given string.
	// Strings are laid out in read-only memory with one word of pointer,
	// one word of length, then the contents of the string.
	StringSym(string) interface{} // returns *gc.Sym
}

// NewConfig returns a new configuration object for the given architecture.
func NewConfig(arch string, fe Frontend) *Config {
	c := &Config{arch: arch, fe: fe}
	switch arch {
	case "amd64":
		c.ptrSize = 8
		c.lowerBlock = lowerBlockAMD64
		c.lowerValue = lowerValueAMD64
	case "386":
		c.ptrSize = 4
		c.lowerBlock = lowerBlockAMD64
		c.lowerValue = lowerValueAMD64 // TODO(khr): full 32-bit support
	default:
		log.Fatalf("arch %s not implemented", arch)
	}

	// cache the intptr type in the config
	c.Uintptr = TypeUInt32
	if c.ptrSize == 8 {
		c.Uintptr = TypeUInt64
	}

	return c
}

// NewFunc returns a new, empty function object
func (c *Config) NewFunc() *Func {
	// TODO(khr): should this function take name, type, etc. as arguments?
	return &Func{Config: c}
}

// TODO(khr): do we really need a separate Config, or can we just
// store all its fields inside a Func?
