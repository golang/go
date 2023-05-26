// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21
// +build go1.21

package tests

func init() {
	builtins["clear"] = true
	builtins["max"] = true
	builtins["min"] = true
}
