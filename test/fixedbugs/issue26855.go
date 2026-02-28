// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we get the correct (T vs &T) literal specification
// in the error message.

package p

type S struct {
	f T
}

type P struct {
	f *T
}

type T struct{}

var _ = S{
	f: &T{}, // ERROR "cannot use &T{}|incompatible type"
}

var _ = P{
	f: T{}, // ERROR "cannot use T{}|incompatible type"
}
