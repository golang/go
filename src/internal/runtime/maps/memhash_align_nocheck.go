// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(amd64 || 386)

package maps

func checkMasksAndShiftsAlignment() bool {
	// This check is only meaningful on amd64/386, where the AES memhash
	// implementation depends on these globals being properly aligned.
	//
	// Return false here so any accidental use on other architectures fails
	// loudly rather than silently succeeding.
	return false
}
