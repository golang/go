// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !mips64,!mips64le

// Declarations for routines that are implemented in noasm.go.

package runtime

func cmpstring(s1, s2 string) int
