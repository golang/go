// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package addr holds the definition of an instruction address.
package addr

// Addr represents a parsed address.
type Addr struct {
	IsStatic            bool    // symbol<>
	IsImmediateConstant bool    // $3
	IsImmediateAddress  bool    // $main·main(SB)
	IsIndirect          bool    // (R1)
	HasRegister         bool    // register is set
	HasRegister2        bool    // register2 is set
	HasFloat            bool    // float is set
	HasOffset           bool    // offset is set
	HasString           bool    // string is set
	Symbol              string  // "main·main"
	Register            int16   // R1
	Register2           int16   // R1 in R0:R1
	Offset              int64   // 3
	Float               float64 // 1.0e2 (floating constant)
	String              string  // "hi" (string constant)
	Index               int16   // R1 in (R1*8)
	Scale               int8    // 8 in (R1*8)
}

const (
	// IsStatic does not appear here; Is and Has methods ignore it.
	ImmediateConstant = 1 << iota
	ImmediateAddress
	Indirect
	Symbol
	Register
	Register2
	Offset
	Float
	String
	Index
	Scale
)

// Has reports whether the address has any of the specified elements.
// Indirect and immediate are not checked.
func (a *Addr) Has(mask int) bool {
	if mask&Symbol != 0 && a.Symbol != "" {
		return true
	}
	if mask&Register != 0 && a.HasRegister {
		return true
	}
	if mask&Register2 != 0 && a.HasRegister2 {
		return true
	}
	if mask&Offset != 0 && a.HasOffset {
		return true
	}
	if mask&Float != 0 && a.HasFloat {
		return true
	}
	if mask&String != 0 && a.HasString {
		return true
	}
	if mask&Index != 0 && a.Index != 0 {
		return true
	}
	if mask&Scale != 0 && a.Scale != 0 {
		return true
	}
	return false
}

// Is reports whether the address has all the specified elements.
// Indirect and immediate are checked.
func (a *Addr) Is(mask int) bool {
	if (mask&ImmediateConstant == 0) != !a.IsImmediateConstant {
		return false
	}
	if (mask&ImmediateAddress == 0) != !a.IsImmediateAddress {
		return false
	}
	if (mask&Indirect == 0) != !a.IsIndirect {
		return false
	}
	if (mask&Symbol == 0) != (a.Symbol == "") {
		return false
	}
	if (mask&Register == 0) != !a.HasRegister {
		return false
	}
	if (mask&Register2 == 0) != !a.HasRegister2 {
		return false
	}
	if (mask&Offset == 0) != !a.HasOffset {
		// $0 has the immediate bit but value 0.
		return false
	}
	if (mask&Float == 0) != !a.HasFloat {
		return false
	}
	if (mask&String == 0) != !a.HasString {
		return false
	}
	if (mask&Index == 0) != (a.Index == 0) {
		return false
	}
	if (mask&Scale == 0) != (a.Scale == 0) {
		return false
	}
	return true
}
