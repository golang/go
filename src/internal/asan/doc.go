// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package asan contains helper functions for manually instrumenting
// code for the address sanitizer.
// The runtime package intentionally exports these functions only in the
// asan build; this package exports them unconditionally but without the
// "asan" build tag they are no-ops.
package asan
