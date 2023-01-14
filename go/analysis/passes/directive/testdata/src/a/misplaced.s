// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// want +1 `//go:debug directive only valid in Go source files`
//go:debug panicnil=1

/*
can skip over comments
//go:debug doesn't matter here
*/

// want +1 `//go:debug directive only valid in Go source files`
//go:debug panicnil=1

package a

// no error here because we can't parse this far
//go:debug panicnil=1
