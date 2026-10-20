// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

// WasmFeatures is the value of the --wasm-features build flag: a
// comma-separated list of optional wasm features for GOARCH=wasm. It is
// empty by default; stack-switching is the only entry the compiler accepts.
var WasmFeatures string
