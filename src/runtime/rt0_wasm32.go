// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm32

package runtime

// mainFunc is statically initialized so rt0_go can pass its function value
// to newproc before package initialization runs.
var mainFunc = main
