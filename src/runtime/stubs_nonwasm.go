// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm && !wasm32

package runtime

// pause is only used on WebAssembly.
func pause(newsp uintptr) { panic("unreachable") }
