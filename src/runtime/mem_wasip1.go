// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package runtime

func resetMemoryDataView() {
	// This function is a no-op on WASI, it is only used to notify the browser
	// that its view of the WASM memory needs to be updated when compiling for
	// GOOS=js.
}
