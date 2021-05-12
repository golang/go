// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.1
// +build go1.1

package main

// Test that go1.1 tag above is included in builds. main.go refers to this definition.
const go11tag = true
