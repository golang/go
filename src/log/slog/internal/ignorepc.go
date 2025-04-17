// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

// If IgnorePC is true, do not invoke runtime.Callers to get the pc.
// This is solely for benchmarking the slowdown from runtime.Callers.
var IgnorePC = false
