// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// expose internals for testing

package sync

func Semacquire(s *int32) {
	semacquire(s)
}

func Semrelease(s *int32) {
	semrelease(s)
}
