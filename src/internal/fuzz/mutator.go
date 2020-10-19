// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import "math/rand"

func mutate(b []byte) []byte {
	mutated := make([]byte, len(b))
	copy(mutated, b)

	// Mutate a byte in a random position.
	pos := rand.Intn(len(mutated))
	mutated[pos] = byte(rand.Intn(256))

	return mutated
}
