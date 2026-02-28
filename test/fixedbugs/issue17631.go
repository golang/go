// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "time"

func main() {
	_ = struct {
		about      string
		before     map[string]uint
		update     map[string]int
		updateTime time.Time
		expect     map[string]int
	}{
		about:   "this one",
		updates: map[string]int{"gopher": 10}, // ERROR "unknown field 'updates' in struct literal of type|unknown field .*updates.* in .*unnamed struct.*"
	}
}
