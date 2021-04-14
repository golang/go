// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Disabled for s390x because it uses assembly routines that are not
// accurate for huge arguments.

// +build !s390x

package math_test

import (
	. "math"
	"testing"
)

// Inputs to test trig_reduce
var trigHuge = []float64{
	1 << 28,
	1 << 29,
	1 << 30,
	1 << 35,
	1 << 120,
	1 << 240,
	1 << 480,
	1234567891234567 << 180,
	1234567891234567 << 300,
	MaxFloat64,
}

// Results for trigHuge[i] calculated with https://github.com/robpike/ivy
// using 4096 bits of working precision.   Values requiring less than
// 102 decimal digits (1 << 120, 1 << 240, 1 << 480, 1234567891234567 << 180)
// were confirmed via https://keisan.casio.com/
var cosHuge = []float64{
	-0.16556897949057876,
	-0.94517382606089662,
	0.78670712294118812,
	-0.76466301249635305,
	-0.92587902285483787,
	0.93601042593353793,
	-0.28282777640193788,
	-0.14616431394103619,
	-0.79456058210671406,
	-0.99998768942655994,
}

var sinHuge = []float64{
	-0.98619821183697566,
	0.32656766301856334,
	-0.61732641504604217,
	-0.64443035102329113,
	0.37782010936075202,
	-0.35197227524865778,
	0.95917070894368716,
	0.98926032637023618,
	-0.60718488235646949,
	0.00496195478918406,
}

var tanHuge = []float64{
	5.95641897939639421,
	-0.34551069233430392,
	-0.78469661331920043,
	0.84276385870875983,
	-0.40806638884180424,
	-0.37603456702698076,
	-3.39135965054779932,
	-6.76813854009065030,
	0.76417695016604922,
	-0.00496201587444489,
}

// Check that trig values of huge angles return accurate results.
// This confirms that argument reduction works for very large values
// up to MaxFloat64.
func TestHugeCos(t *testing.T) {
	for i := 0; i < len(trigHuge); i++ {
		f1 := cosHuge[i]
		f2 := Cos(trigHuge[i])
		if !close(f1, f2) {
			t.Errorf("Cos(%g) = %g, want %g", trigHuge[i], f2, f1)
		}
	}
}

func TestHugeSin(t *testing.T) {
	for i := 0; i < len(trigHuge); i++ {
		f1 := sinHuge[i]
		f2 := Sin(trigHuge[i])
		if !close(f1, f2) {
			t.Errorf("Sin(%g) = %g, want %g", trigHuge[i], f2, f1)
		}
	}
}

func TestHugeSinCos(t *testing.T) {
	for i := 0; i < len(trigHuge); i++ {
		f1, g1 := sinHuge[i], cosHuge[i]
		f2, g2 := Sincos(trigHuge[i])
		if !close(f1, f2) || !close(g1, g2) {
			t.Errorf("Sincos(%g) = %g, %g, want %g, %g", trigHuge[i], f2, g2, f1, g1)
		}
	}
}

func TestHugeTan(t *testing.T) {
	for i := 0; i < len(trigHuge); i++ {
		f1 := tanHuge[i]
		f2 := Tan(trigHuge[i])
		if !close(f1, f2) {
			t.Errorf("Tan(%g) = %g, want %g", trigHuge[i], f2, f1)
		}
	}
}
