// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure this doesn't crash the compiler.
// Line 9 must end in EOF for this test (no newline).

package e
func([<-chan<-[func /* ERROR unexpected u */ u){go