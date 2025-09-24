// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

//go:noinline
func testNoZicondUnderRva20u64(a, b int) int {
	result := b
	if a == 0 {
		result = a
	}
	// riscv64/rva20u64:-`CZEROEQZ`, -`CZERONEZ`
	return result
}

//go:noinline
func testNoZicondUnderRva22u64(a, b int) int {
	result := b
	if a == 0 {
		result = a
	}
	// riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	return result
}

//go:noinline
func testGenZicondUnderRva23u64(a, b int) int {
	result := b
	if a == 0 {
		result = a
	}
	// riscv64/rva23u64:`CZEROEQZ`, -`CZERONEZ`
	return result
}

//go:noinline
func testGenZicond(a, b int) int {
	var result int
	if a > b {
		result = a
	} else {
		result = b
	}
	// riscv64/rva23u64:`CZERONEZ`,`CZEROEQZ`
	return result
}

//go:noinline
func selectIfZero(cond, a, b int) int {
	result := a
	if cond == 0 {
		result = b
	}
	// riscv64/rva23u64:`CZERONEZ`,`CZEROEQZ`, `OR`
	// riscv64/rva23u64:-`SEQZ`
	return result
}

//go:noinline
func testSelectIfNotZero(cond, a, b int) int {
	result := a
	if cond != 0 {
		result = b
	}
	// riscv64/rva23u64:`CZERONEZ`,`CZEROEQZ`, `OR`
	// riscv64/rva23u64:-`SNEZ`
	return result
}

//go:noinline
func testCondAddZero(cond, a, b int) int {
	result := a
	if cond == 0 {
		result = a + b
	}
	// riscv64/rva23u64:`CZERONEZ`, `ADD`
	// riscv64/rva23u64:-`SEQZ`, -`CZEROEQZ`, -`OR`
	return result
}

//go:noinline
func testCondAddNonZero(cond, a, b int) int {
	var result int
	if cond != 0 {
		result = a + b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZEROEQZ`, `ADD`
	// riscv64/rva23u64:-`SNEZ`, -`CZERONEZ`, -`OR`
	return result
}

//go:noinline
func testCondSubZero(cond, a, b int) int {
	var result int
	if cond == 0 {
		result = a - b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZERONEZ`, `SUB`
	// riscv64/rva23u64:-`SEQZ`, -`CZEROEQZ`, -`OR`
	return result
}

//go:noinline
func testCondSubNonZero(cond, a, b int) int {
	var result int
	if cond != 0 {
		result = a - b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZEROEQZ`, `SUB`
	// riscv64/rva23u64:-`SNEZ`, -`CZERONEZ`, -`OR`
	return result
}

//go:noinline
func testCondOrZero(cond, a, b int) int {
	var result int
	if cond == 0 {
		result = a | b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZERONEZ`, `OR`
	// riscv64/rva23u64:-`SEQZ`, -`CZEROEQZ`
	return result
}

//go:noinline
func testCondOrNonZero(cond, a, b int) int {
	var result int
	if cond != 0 {
		result = a | b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZEROEQZ`, `OR`
	// riscv64/rva23u64:-`SNEZ`, -`CZERONEZ`
	return result
}

//go:noinline
func testCondXorZero(cond, a, b int) int {
	var result int
	if cond == 0 {
		result = a ^ b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZERONEZ`, `XOR`
	// riscv64/rva23u64:-`SEQZ`, -`CZEROEQZ`, -`OR`
	return result
}

//go:noinline
func testCondXorNonZero(cond, a, b int) int {
	var result int
	if cond != 0 {
		result = a ^ b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZEROEQZ`, `XOR`
	// riscv64/rva23u64:-`SNEZ`, -`CZERONEZ`, -`OR`
	return result
}

//go:noinline
func testCondAndZero(cond, a, b int) int {
	var result int
	if cond == 0 {
		result = a & b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZEROEQZ`, `AND`, `OR`
	// riscv64/rva23u64:-`SEQZ`, -`CZERONEZ`
	return result
}

//go:noinline
func testCondAndNonZero(cond, a, b int) int {
	var result int
	if cond != 0 {
		result = a & b
	} else {
		result = a
	}
	// riscv64/rva23u64:`CZERONEZ`, `AND`, `OR`
	// riscv64/rva23u64:-`SNEZ`, -`CZEROEQZ`
	return result
}
