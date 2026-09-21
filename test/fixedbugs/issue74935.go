// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type W struct {
	E struct{}
	X *byte
}

func main() {
	w := reflect.ValueOf(W{})
	_ = w.Field(0).Interface()
}
