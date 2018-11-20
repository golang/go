// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"math/rand"
	"testing"
)

func TestMUD(t *testing.T) {
	// Insert random uniforms and check histogram mass and
	// cumulative sum approximations.
	rnd := rand.New(rand.NewSource(42))
	mass := 0.0
	var mud mud
	for i := 0; i < 100; i++ {
		area, l, r := rnd.Float64(), rnd.Float64(), rnd.Float64()
		if rnd.Intn(10) == 0 {
			r = l
		}
		t.Log(l, r, area)
		mud.add(l, r, area)
		mass += area

		// Check total histogram weight.
		hmass := 0.0
		for _, val := range mud.hist {
			hmass += val
		}
		if !aeq(mass, hmass) {
			t.Fatalf("want mass %g, got %g", mass, hmass)
		}

		// Check inverse cumulative sum approximations.
		for j := 0.0; j < mass; j += mass * 0.099 {
			mud.setTrackMass(j)
			l, u, ok := mud.approxInvCumulativeSum()
			inv, ok2 := mud.invCumulativeSum(j)
			if !ok || !ok2 {
				t.Fatalf("inverse cumulative sum failed: approx %v, exact %v", ok, ok2)
			}
			if !(l <= inv && inv < u) {
				t.Fatalf("inverse(%g) = %g, not ∈ [%g, %g)", j, inv, l, u)
			}
		}
	}
}

func TestMUDTracking(t *testing.T) {
	// Test that the tracked mass is tracked correctly across
	// updates.
	rnd := rand.New(rand.NewSource(42))
	const uniforms = 100
	for trackMass := 0.0; trackMass < uniforms; trackMass += uniforms / 50 {
		var mud mud
		mass := 0.0
		mud.setTrackMass(trackMass)
		for i := 0; i < uniforms; i++ {
			area, l, r := rnd.Float64(), rnd.Float64(), rnd.Float64()
			mud.add(l, r, area)
			mass += area
			l, u, ok := mud.approxInvCumulativeSum()
			inv, ok2 := mud.invCumulativeSum(trackMass)

			if mass < trackMass {
				if ok {
					t.Errorf("approx(%g) = [%g, %g), but mass = %g", trackMass, l, u, mass)
				}
				if ok2 {
					t.Errorf("exact(%g) = %g, but mass = %g", trackMass, inv, mass)
				}
			} else {
				if !ok {
					t.Errorf("approx(%g) failed, but mass = %g", trackMass, mass)
				}
				if !ok2 {
					t.Errorf("exact(%g) failed, but mass = %g", trackMass, mass)
				}
				if ok && ok2 && !(l <= inv && inv < u) {
					t.Errorf("inverse(%g) = %g, not ∈ [%g, %g)", trackMass, inv, l, u)
				}
			}
		}
	}
}
