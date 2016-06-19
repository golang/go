// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests continue and break.

package main

func continuePlain_ssa() int {
	var n int
	for i := 0; i < 10; i++ {
		if i == 6 {
			continue
		}
		n = i
	}
	return n
}

func continueLabeled_ssa() int {
	var n int
Next:
	for i := 0; i < 10; i++ {
		if i == 6 {
			continue Next
		}
		n = i
	}
	return n
}

func continuePlainInner_ssa() int {
	var n int
	for j := 0; j < 30; j += 10 {
		for i := 0; i < 10; i++ {
			if i == 6 {
				continue
			}
			n = i
		}
		n += j
	}
	return n
}

func continueLabeledInner_ssa() int {
	var n int
	for j := 0; j < 30; j += 10 {
	Next:
		for i := 0; i < 10; i++ {
			if i == 6 {
				continue Next
			}
			n = i
		}
		n += j
	}
	return n
}

func continueLabeledOuter_ssa() int {
	var n int
Next:
	for j := 0; j < 30; j += 10 {
		for i := 0; i < 10; i++ {
			if i == 6 {
				continue Next
			}
			n = i
		}
		n += j
	}
	return n
}

func breakPlain_ssa() int {
	var n int
	for i := 0; i < 10; i++ {
		if i == 6 {
			break
		}
		n = i
	}
	return n
}

func breakLabeled_ssa() int {
	var n int
Next:
	for i := 0; i < 10; i++ {
		if i == 6 {
			break Next
		}
		n = i
	}
	return n
}

func breakPlainInner_ssa() int {
	var n int
	for j := 0; j < 30; j += 10 {
		for i := 0; i < 10; i++ {
			if i == 6 {
				break
			}
			n = i
		}
		n += j
	}
	return n
}

func breakLabeledInner_ssa() int {
	var n int
	for j := 0; j < 30; j += 10 {
	Next:
		for i := 0; i < 10; i++ {
			if i == 6 {
				break Next
			}
			n = i
		}
		n += j
	}
	return n
}

func breakLabeledOuter_ssa() int {
	var n int
Next:
	for j := 0; j < 30; j += 10 {
		for i := 0; i < 10; i++ {
			if i == 6 {
				break Next
			}
			n = i
		}
		n += j
	}
	return n
}

var g, h int // globals to ensure optimizations don't collapse our switch statements

func switchPlain_ssa() int {
	var n int
	switch g {
	case 0:
		n = 1
		break
		n = 2
	}
	return n
}

func switchLabeled_ssa() int {
	var n int
Done:
	switch g {
	case 0:
		n = 1
		break Done
		n = 2
	}
	return n
}

func switchPlainInner_ssa() int {
	var n int
	switch g {
	case 0:
		n = 1
		switch h {
		case 0:
			n += 10
			break
		}
		n = 2
	}
	return n
}

func switchLabeledInner_ssa() int {
	var n int
	switch g {
	case 0:
		n = 1
	Done:
		switch h {
		case 0:
			n += 10
			break Done
		}
		n = 2
	}
	return n
}

func switchLabeledOuter_ssa() int {
	var n int
Done:
	switch g {
	case 0:
		n = 1
		switch h {
		case 0:
			n += 10
			break Done
		}
		n = 2
	}
	return n
}

func main() {
	tests := [...]struct {
		name string
		fn   func() int
		want int
	}{
		{"continuePlain_ssa", continuePlain_ssa, 9},
		{"continueLabeled_ssa", continueLabeled_ssa, 9},
		{"continuePlainInner_ssa", continuePlainInner_ssa, 29},
		{"continueLabeledInner_ssa", continueLabeledInner_ssa, 29},
		{"continueLabeledOuter_ssa", continueLabeledOuter_ssa, 5},

		{"breakPlain_ssa", breakPlain_ssa, 5},
		{"breakLabeled_ssa", breakLabeled_ssa, 5},
		{"breakPlainInner_ssa", breakPlainInner_ssa, 25},
		{"breakLabeledInner_ssa", breakLabeledInner_ssa, 25},
		{"breakLabeledOuter_ssa", breakLabeledOuter_ssa, 5},

		{"switchPlain_ssa", switchPlain_ssa, 1},
		{"switchLabeled_ssa", switchLabeled_ssa, 1},
		{"switchPlainInner_ssa", switchPlainInner_ssa, 2},
		{"switchLabeledInner_ssa", switchLabeledInner_ssa, 2},
		{"switchLabeledOuter_ssa", switchLabeledOuter_ssa, 11},

		// no select tests; they're identical to switch
	}

	var failed bool
	for _, test := range tests {
		if got := test.fn(); test.fn() != test.want {
			print(test.name, "()=", got, ", want ", test.want, "\n")
			failed = true
		}
	}

	if failed {
		panic("failed")
	}
}
