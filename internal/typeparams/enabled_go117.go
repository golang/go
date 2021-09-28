// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !typeparams || !go1.18
// +build !typeparams !go1.18

package typeparams

// Enabled reports whether type parameters are enabled in the current build
// environment.
const Enabled = false
