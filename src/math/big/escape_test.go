// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"internal/testenv"
	"math/rand"
	"os/exec"
	"strings"
	"testing"
)

func TestEscape(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	// The multiplication routines create many temporary Int values,
	// expecting them to be stack-allocated. Make sure none escape to the heap.
	out, err := exec.Command("go", "build", "-gcflags=-m").CombinedOutput()
	if err != nil {
		t.Fatalf("go build -gcflags=-m: %v\n%s", err, out)
	}
	for line := range strings.Lines(string(out)) {
		if strings.Contains(line, "natmul.go") && strings.Contains(line, "Int") && strings.Contains(line, "escapes") {
			t.Error(strings.TrimSpace(line))
		}
	}
}

func TestMulAlloc(t *testing.T) {
	r := rand.New(rand.NewSource(1234))
	sizes := []int{karatsubaThreshold / 2, karatsubaThreshold}
	for _, size := range sizes {
		x := randInt(r, uint(size))
		y := randInt(r, uint(size))
		z := &Int{abs: make(nat, 2*uint(size))}
		if n := testing.AllocsPerRun(10, func() { z.Mul(x, y) }); n >= 1 {
			t.Errorf("Mul(len %d, len %d) allocates %.2f objects", size, size, n)
		}
	}
}

func TestSqrAlloc(t *testing.T) {
	r := rand.New(rand.NewSource(1234))
	sizes := []int{basicSqrThreshold / 2, basicSqrThreshold, karatsubaSqrThreshold}
	for _, size := range sizes {
		x := randInt(r, uint(size))
		z := &Int{abs: make(nat, 2*uint(size))}
		if n := testing.AllocsPerRun(10, func() { z.Mul(x, x) }); n >= 1 {
			t.Errorf("Mul(len %d with itself) allocates %.2f objects", size, n)
		}
	}
}
