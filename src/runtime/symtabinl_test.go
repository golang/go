// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
	"internal/stringslite"
)

func XTestInlineUnwinder(t TestingT) {
	if TestenvOptimizationOff() {
		t.Skip("skipping test with inlining optimizations disabled")
	}

	pc1 := abi.FuncPCABIInternal(tiuTest)
	f := findfunc(pc1)
	if !f.valid() {
		t.Fatalf("failed to resolve tiuTest at PC %#x", pc1)
	}

	want := map[string]int{
		"tiuInlined1:3 tiuTest:10":               0,
		"tiuInlined1:3 tiuInlined2:6 tiuTest:11": 0,
		"tiuInlined2:7 tiuTest:11":               0,
		"tiuTest:12":                             0,
	}
	wantStart := map[string]int{
		"tiuInlined1": 2,
		"tiuInlined2": 5,
		"tiuTest":     9,
	}

	// Iterate over the PCs in tiuTest and walk the inline stack for each.
	prevStack := "x"
	for pc := pc1; pc < pc1+1024 && findfunc(pc) == f; pc += sys.PCQuantum {
		stack := ""
		u, uf := newInlineUnwinder(f, pc)
		if file, _ := u.fileLine(uf); file == "?" {
			// We're probably in the trailing function padding, where findfunc
			// still returns f but there's no symbolic information. Just keep
			// going until we definitely hit the end. If we see a "?" in the
			// middle of unwinding, that's a real problem.
			//
			// TODO: If we ever have function end information, use that to make
			// this robust.
			continue
		}
		for ; uf.valid(); uf = u.next(uf) {
			file, line := u.fileLine(uf)
			const wantFile = "symtabinl_test.go"
			if !stringslite.HasSuffix(file, wantFile) {
				t.Errorf("tiuTest+%#x: want file ...%s, got %s", pc-pc1, wantFile, file)
			}

			sf := u.srcFunc(uf)

			name := sf.name()
			const namePrefix = "runtime."
			if stringslite.HasPrefix(name, namePrefix) {
				name = name[len(namePrefix):]
			}
			if !stringslite.HasPrefix(name, "tiu") {
				t.Errorf("tiuTest+%#x: unexpected function %s", pc-pc1, name)
			}

			start := int(sf.startLine) - tiuStart
			if start != wantStart[name] {
				t.Errorf("tiuTest+%#x: want startLine %d, got %d", pc-pc1, wantStart[name], start)
			}
			if sf.funcID != abi.FuncIDNormal {
				t.Errorf("tiuTest+%#x: bad funcID %v", pc-pc1, sf.funcID)
			}

			if len(stack) > 0 {
				stack += " "
			}
			stack += FmtSprintf("%s:%d", name, line-tiuStart)
		}

		if stack != prevStack {
			prevStack = stack

			t.Logf("tiuTest+%#x: %s", pc-pc1, stack)

			if _, ok := want[stack]; ok {
				want[stack]++
			}
		}
	}

	// Check that we got all the stacks we wanted.
	for stack, count := range want {
		if count == 0 {
			t.Errorf("missing stack %s", stack)
		}
	}
}

func lineNumber() int {
	_, _, line, _ := Caller(1)
	return line // return 0 for error
}

// Below here is the test data for XTestInlineUnwinder

var tiuStart = lineNumber() // +0
var tiu1, tiu2, tiu3 int    // +1
func tiuInlined1() { // +2
	tiu1++ // +3
} // +4
func tiuInlined2() { // +5
	tiuInlined1() // +6
	tiu2++        // +7
} // +8
func tiuTest() { // +9
	tiuInlined1() // +10
	tiuInlined2() // +11
	tiu3++        // +12
} // +13
