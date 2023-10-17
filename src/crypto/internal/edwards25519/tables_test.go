// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package edwards25519

import (
	"testing"
)

func TestProjLookupTable(t *testing.T) {
	var table projLookupTable
	table.FromP3(B)

	var tmp1, tmp2, tmp3 projCached
	table.SelectInto(&tmp1, 6)
	table.SelectInto(&tmp2, -2)
	table.SelectInto(&tmp3, -4)
	// Expect T1 + T2 + T3 = identity

	var accP1xP1 projP1xP1
	accP3 := NewIdentityPoint()

	accP1xP1.Add(accP3, &tmp1)
	accP3.fromP1xP1(&accP1xP1)
	accP1xP1.Add(accP3, &tmp2)
	accP3.fromP1xP1(&accP1xP1)
	accP1xP1.Add(accP3, &tmp3)
	accP3.fromP1xP1(&accP1xP1)

	if accP3.Equal(I) != 1 {
		t.Errorf("Consistency check on ProjLookupTable.SelectInto failed!  %x %x %x", tmp1, tmp2, tmp3)
	}
}

func TestAffineLookupTable(t *testing.T) {
	var table affineLookupTable
	table.FromP3(B)

	var tmp1, tmp2, tmp3 affineCached
	table.SelectInto(&tmp1, 3)
	table.SelectInto(&tmp2, -7)
	table.SelectInto(&tmp3, 4)
	// Expect T1 + T2 + T3 = identity

	var accP1xP1 projP1xP1
	accP3 := NewIdentityPoint()

	accP1xP1.AddAffine(accP3, &tmp1)
	accP3.fromP1xP1(&accP1xP1)
	accP1xP1.AddAffine(accP3, &tmp2)
	accP3.fromP1xP1(&accP1xP1)
	accP1xP1.AddAffine(accP3, &tmp3)
	accP3.fromP1xP1(&accP1xP1)

	if accP3.Equal(I) != 1 {
		t.Errorf("Consistency check on ProjLookupTable.SelectInto failed!  %x %x %x", tmp1, tmp2, tmp3)
	}
}

func TestNafLookupTable5(t *testing.T) {
	var table nafLookupTable5
	table.FromP3(B)

	var tmp1, tmp2, tmp3, tmp4 projCached
	table.SelectInto(&tmp1, 9)
	table.SelectInto(&tmp2, 11)
	table.SelectInto(&tmp3, 7)
	table.SelectInto(&tmp4, 13)
	// Expect T1 + T2 = T3 + T4

	var accP1xP1 projP1xP1
	lhs := NewIdentityPoint()
	rhs := NewIdentityPoint()

	accP1xP1.Add(lhs, &tmp1)
	lhs.fromP1xP1(&accP1xP1)
	accP1xP1.Add(lhs, &tmp2)
	lhs.fromP1xP1(&accP1xP1)

	accP1xP1.Add(rhs, &tmp3)
	rhs.fromP1xP1(&accP1xP1)
	accP1xP1.Add(rhs, &tmp4)
	rhs.fromP1xP1(&accP1xP1)

	if lhs.Equal(rhs) != 1 {
		t.Errorf("Consistency check on nafLookupTable5 failed")
	}
}

func TestNafLookupTable8(t *testing.T) {
	var table nafLookupTable8
	table.FromP3(B)

	var tmp1, tmp2, tmp3, tmp4 affineCached
	table.SelectInto(&tmp1, 49)
	table.SelectInto(&tmp2, 11)
	table.SelectInto(&tmp3, 35)
	table.SelectInto(&tmp4, 25)
	// Expect T1 + T2 = T3 + T4

	var accP1xP1 projP1xP1
	lhs := NewIdentityPoint()
	rhs := NewIdentityPoint()

	accP1xP1.AddAffine(lhs, &tmp1)
	lhs.fromP1xP1(&accP1xP1)
	accP1xP1.AddAffine(lhs, &tmp2)
	lhs.fromP1xP1(&accP1xP1)

	accP1xP1.AddAffine(rhs, &tmp3)
	rhs.fromP1xP1(&accP1xP1)
	accP1xP1.AddAffine(rhs, &tmp4)
	rhs.fromP1xP1(&accP1xP1)

	if lhs.Equal(rhs) != 1 {
		t.Errorf("Consistency check on nafLookupTable8 failed")
	}
}
