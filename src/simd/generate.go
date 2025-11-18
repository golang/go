// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simd

// Invoke code generators.
//
// This file intentionally has no goexperiment.simd build tag, so that go
// generate can run without a GOEXPERIMENT set.

//go:generate go run -C _gen . -tmplgen -simdgen
