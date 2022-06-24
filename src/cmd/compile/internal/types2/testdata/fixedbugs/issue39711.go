// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Do not report a duplicate type error for this term list.
// (Check types after interfaces have been completed.)
type _ interface {
	// TODO(gri) Once we have full type sets we can enable this again.
	// Fow now we don't permit interfaces in term lists.
	// type interface{ Error() string }, interface{ String() string }
}
