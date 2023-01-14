// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// want +1 `invalid space '\\u00a0' in //go:debug directive`
//go:debug 00a0

package main

