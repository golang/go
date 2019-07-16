// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test checks if the compiler's internal constant
// arithmetic correctly rounds denormal float32 values.

package main

import (
	"fmt"
	"math"
)

func main() {
	for _, t := range []struct {
		value float32
		bits  uint32
	}{
		{0e+00, 0x00000000},
		{1e-46, 0x00000000},
		{0.5e-45, 0x00000000},
		{0.8e-45, 0x00000001},
		{1e-45, 0x00000001},
		{2e-45, 0x00000001},
		{3e-45, 0x00000002},
		{4e-45, 0x00000003},
		{5e-45, 0x00000004},
		{6e-45, 0x00000004},
		{7e-45, 0x00000005},
		{8e-45, 0x00000006},
		{9e-45, 0x00000006},
		{1.0e-44, 0x00000007},
		{1.1e-44, 0x00000008},
		{1.2e-44, 0x00000009},
	} {
		got := math.Float32bits(t.value)
		want := t.bits
		if got != want {
			panic(fmt.Sprintf("bits(%g) = 0x%08x; want 0x%08x", t.value, got, want))
		}
	}
}
