// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// logosWasmMode reports whether the binary was linked with -gc=wasm, the
// mode that splits the wasm heap into a low permanent and a high scratch
// region and marks in slices during idle time of the JS event loop. The
// linker sets logosGCMode; an empty value keeps the upstream collector.
func logosWasmMode() bool {
	return logosGCMode == "wasm"
}
