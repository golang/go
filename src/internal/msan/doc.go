// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package msan contains helper functions for manually instrumenting code
// for the memory sanitizer.
// This package exports the private msan routines in runtime unconditionally
// but without the "msan" build tag they are no-ops.
package msan
