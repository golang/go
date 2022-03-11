// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package a

import "testing"

var fuzzTargets = []testing.InternalFuzzTarget{
	{"Fuzz", Fuzz},
}

func Fuzz(f *testing.F) {}
