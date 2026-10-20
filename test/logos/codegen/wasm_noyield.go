// asmcheck -gcflags=-logos.wasmprofile

package codegen

// M16-S1 attack, Go twin of wasm_noyield.lgo (R-U1): the profile also
// serves Go packages. Plain go keeps it off until the G3 flip, so this
// twin turns it on with the compiler flag; the shape checks are the same
// as the loGo/s file. The check on a func line covers the prologue, which
// carries the position of the func keyword.

import "unsafe"

//go:linkname gopark runtime.gopark
func gopark(unlockf func(unsafe.Pointer, unsafe.Pointer) bool, lock unsafe.Pointer, reason, traceReason uint8, traceskip int)

// A straight-line leaf computes with locals only: no dispatch.
//
//go:noinline
func leaf(a, b int) int { // js/wasm/:-"BrTable" -"Loop" wasip1/wasm/:-"BrTable" -"Loop"
	// js/wasm/:"I64Mul" "I64Add" wasip1/wasm/:"I64Mul" "I64Add"
	return a*b + 3
}

// A caller of pure leaves needs no resume point after either call: the call
// is native and an unwinding callee traps.
//
//go:noinline
func straight(x int) int { // js/wasm/:-"BrTable" wasip1/wasm/:-"BrTable"
	// js/wasm/:"Call \\S+\\.leaf" "UNDEF" -"BrIf" wasip1/wasm/:"Call \\S+\\.leaf" "UNDEF" -"BrIf"
	a := leaf(x, 1)
	// js/wasm/:"Call \\S+\\.leaf" "UNDEF" -"BrIf" wasip1/wasm/:"Call \\S+\\.leaf" "UNDEF" -"BrIf"
	b := leaf(x, 2)
	return a + b
}

// A channel operation may park the goroutine: dispatch stays.
//
//go:noinline
func recv() int { // js/wasm/:"BrTable" wasip1/wasm/:"BrTable"
	c := make(chan int, 1)
	c <- 1
	// js/wasm/:"Call runtime\\.chanrecv1" "BrIf" wasip1/wasm/:"Call runtime\\.chanrecv1" "BrIf"
	return <-c
}

// A defer needs the deferreturn resume point: dispatch stays.
//
//go:noinline
func deferred(p *int) { // js/wasm/:"BrTable" wasip1/wasm/:"BrTable"
	defer func() { *p++ }()
	*p = 1
}

// A function that parks calls runtime.gopark: dispatch stays, and so does
// the dispatch of its caller.
//
//go:noinline
func parks() { // js/wasm/:"BrTable" wasip1/wasm/:"BrTable"
	gopark(nil, nil, 0, 0, 0)
}

//go:noinline
func transitive() { // js/wasm/:"BrTable" wasip1/wasm/:"BrTable"
	// js/wasm/:"Call \\S+\\.parks" "BrIf" wasip1/wasm/:"Call \\S+\\.parks" "BrIf"
	parks()
}
