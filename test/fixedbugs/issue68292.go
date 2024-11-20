// errorcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[S any, T any](T) {}
func g() {
	f(0) // ERROR "in call to f, cannot infer S \(declared at issue68292.go:9:8\)"
}
