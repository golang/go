// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"reflect"
)

func isMinimizable(t reflect.Type) bool {
	return t == reflect.TypeOf("") || t == reflect.TypeOf([]byte(nil))
}

func minimizeBytes(v []byte, try func([]byte) bool, shouldStop func() bool) {
	tmp := make([]byte, len(v))
	// If minimization was successful at any point during minimizeBytes,
	// then the vals slice in (*workerServer).minimizeInput will point to
	// tmp. Since tmp is altered while making new candidates, we need to
	// make sure that it is equal to the correct value, v, before exiting
	// this function.
	defer copy(tmp, v)

	// First, try to cut the tail.
	for n := 1024; n != 0; n /= 2 {
		for len(v) > n {
			if shouldStop() {
				return
			}
			candidate := v[:len(v)-n]
			if !try(candidate) {
				break
			}
			// Set v to the new value to continue iterating.
			v = candidate
		}
	}

	// Then, try to remove each individual byte.
	for i := 0; i < len(v)-1; i++ {
		if shouldStop() {
			return
		}
		candidate := tmp[:len(v)-1]
		copy(candidate[:i], v[:i])
		copy(candidate[i:], v[i+1:])
		if !try(candidate) {
			continue
		}
		// Update v to delete the value at index i.
		copy(v[i:], v[i+1:])
		v = v[:len(candidate)]
		// v[i] is now different, so decrement i to redo this iteration
		// of the loop with the new value.
		i--
	}

	// Then, try to remove each possible subset of bytes.
	for i := 0; i < len(v)-1; i++ {
		copy(tmp, v[:i])
		for j := len(v); j > i+1; j-- {
			if shouldStop() {
				return
			}
			candidate := tmp[:len(v)-j+i]
			copy(candidate[i:], v[j:])
			if !try(candidate) {
				continue
			}
			// Update v and reset the loop with the new length.
			copy(v[i:], v[j:])
			v = v[:len(candidate)]
			j = len(v)
		}
	}

	// Then, try to make it more simplified and human-readable by trying to replace each
	// byte with a printable character.
	printableChars := []byte("012789ABCXYZabcxyz !\"#$%&'()*+,.")
	for i, b := range v {
		if shouldStop() {
			return
		}

		for _, pc := range printableChars {
			v[i] = pc
			if try(v) {
				// Successful. Move on to the next byte in v.
				break
			}
			// Unsuccessful. Revert v[i] back to original value.
			v[i] = b
		}
	}
}
