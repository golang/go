// errorcheck -0 -race

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sample

type Html struct {
	headerIDs map[string]int
}

// We don't want to see:
//    internal error: (*Html).xyzzy autotmp_3 (type *int) recorded as live on entry, p.Pc=0
// or (now, with the error caught earlier)
//    Treating auto as if it were arg, func (*Html).xyzzy, node ...
// caused by racewalker inserting instrumentation before an OAS where the Ninit
// of the OAS defines part of its right-hand-side. (I.e., the race instrumentation
// references a variable before it is defined.)
func (options *Html) xyzzy(id string) string {
	for count, found := options.headerIDs[id]; found; count, found = options.headerIDs[id] {
		_ = count
	}
	return ""
}
