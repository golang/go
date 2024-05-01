// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure this doesn't crash the compiler.
// (This test should be part of the internal/syntax
// package, but we don't have a general test framework
// there yet, so put it here for now. See also #20800.)

package e
func([<-chan<-[func u){go // ERROR "unexpected `u'"