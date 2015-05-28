// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "log"

type Config struct {
	arch    string            // "amd64", etc.
	ptrSize int64             // 4 or 8
	Uintptr Type              // pointer arithmetic type
	lower   func(*Value) bool // lowering function

	// TODO: more stuff.  Compiler flags of interest, ...
}

// NewConfig returns a new configuration object for the given architecture.
func NewConfig(arch string) *Config {
	c := &Config{arch: arch}
	switch arch {
	case "amd64":
		c.ptrSize = 8
		c.lower = lowerAmd64
	case "386":
		c.ptrSize = 4
		c.lower = lowerAmd64 // TODO(khr): full 32-bit support
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
