// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// TestTree provides basic coverage of the AVL tree operations.
func TestTree(t *testing.T) {
	// Use a reproducible PRNG.
	seed1, seed2 := rand.Uint64(), rand.Uint64()
	t.Logf("random seeds: %d, %d", seed1, seed2)
	rng := rand.New(rand.NewPCG(seed1, seed2))

	// Create a number of Files of arbitrary size.
	files := make([]*File, 500)
	var base int
	for i := range files {
		base++
		size := 1000
		files[i] = &File{base: base, size: size}
		base += size
	}

	// Add them all to the tree in random order.
	var tr tree
	{
		files2 := slices.Clone(files)
		Shuffle(rng, files2)
		for _, f := range files2 {
			tr.add(f)
		}
	}

	// Randomly delete a subset of them.
	for range 100 {
		i := rng.IntN(len(files))
		file := files[i]
		if file == nil {
			continue // already deleted
		}
		files[i] = nil

		pn, _ := tr.locate(file.key())
		if (*pn).file != file {
			t.Fatalf("locate returned wrong file")
		}
		tr.delete(pn)
	}

	// Check some position lookups within each file.
	for _, file := range files {
		if file == nil {
			continue // deleted
		}
		for _, pos := range []int{
			file.base,               // start
			file.base + file.size/2, // midpoint
			file.base + file.size,   // end
		} {
			pn, _ := tr.locate(key{pos, pos})
			if (*pn).file != file {
				t.Fatalf("lookup %s@%d returned wrong file %s",
					file.name, pos,
					(*pn).file.name)
			}
		}
	}

	// Check that the sequence is the same.
	files = slices.DeleteFunc(files, func(f *File) bool { return f == nil })
	if !slices.Equal(slices.Collect(tr.all()), files) {
		t.Fatalf("incorrect tree.all sequence")
	}
}

func Shuffle[T any](rng *rand.Rand, slice []*T) {
	rng.Shuffle(len(slice), func(i, j int) {
		slice[i], slice[j] = slice[j], slice[i]
	})
}
