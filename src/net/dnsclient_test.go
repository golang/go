// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"math/rand"
	"testing"
)

func checkDistribution(t *testing.T, data []*SRV, margin float64) {
	sum := 0
	for _, srv := range data {
		sum += int(srv.Weight)
	}

	results := make(map[string]int)

	count := 1000
	for j := 0; j < count; j++ {
		d := make([]*SRV, len(data))
		copy(d, data)
		byPriorityWeight(d).shuffleByWeight()
		key := d[0].Target
		results[key] = results[key] + 1
	}

	actual := results[data[0].Target]
	expected := float64(count) * float64(data[0].Weight) / float64(sum)
	diff := float64(actual) - expected
	t.Logf("actual: %v diff: %v e: %v m: %v", actual, diff, expected, margin)
	if diff < 0 {
		diff = -diff
	}
	if diff > (expected * margin) {
		t.Errorf("missed target weight: expected %v, %v", expected, actual)
	}
}

func testUniformity(t *testing.T, size int, margin float64) {
	rand.Seed(1)
	data := make([]*SRV, size)
	for i := 0; i < size; i++ {
		data[i] = &SRV{Target: string('a' + i), Weight: 1}
	}
	checkDistribution(t, data, margin)
}

func TestDNSSRVUniformity(t *testing.T) {
	testUniformity(t, 2, 0.05)
	testUniformity(t, 3, 0.10)
	testUniformity(t, 10, 0.20)
	testWeighting(t, 0.05)
}

func testWeighting(t *testing.T, margin float64) {
	rand.Seed(1)
	data := []*SRV{
		{Target: "a", Weight: 60},
		{Target: "b", Weight: 30},
		{Target: "c", Weight: 10},
	}
	checkDistribution(t, data, margin)
}

func TestWeighting(t *testing.T) {
	testWeighting(t, 0.05)
}
