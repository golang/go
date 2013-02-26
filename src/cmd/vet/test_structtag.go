// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the structtag checker.

// +build vet_test

// This file contains the test for canonical struct tags.

package main

type StructTagTest struct {
	X int "hello" // ERROR "not compatible with reflect.StructTag.Get"
}
