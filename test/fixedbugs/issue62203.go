// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"maps"
)

func main() {
	m := map[string]struct{}{}

	// Fill m up to the max for 4 buckets = 48 entries.
	for i := 0; i < 48; i++ {
		m[fmt.Sprintf("%d", i)] = struct{}{}
	}

	// Add a 49th entry, to start a grow to 8 buckets.
	m["foo"] = struct{}{}

	// Remove that 49th entry. m is still growing to 8 buckets,
	// but a clone of m will only have 4 buckets because it
	// only needs to fit 48 entries.
	delete(m, "foo")

	// Clone an 8-bucket map to a 4-bucket map.
	_ = maps.Clone(m)
}
