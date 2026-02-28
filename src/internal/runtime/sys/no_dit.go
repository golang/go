// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64

package sys

var DITSupported = false

func EnableDIT() bool  { return false }
func DITEnabled() bool { return false }
func DisableDIT()      {}
