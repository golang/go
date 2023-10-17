// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package typeparams

// Note: this constant is in a separate file as this is the only acceptable
// diff between the <1.18 API of this package and the 1.18 API.

// Enabled reports whether type parameters are enabled in the current build
// environment.
const Enabled = true
