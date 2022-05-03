// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.17
// +build !go1.17

package cache

// The parser.SkipObjectResolution mode flag is not supported before Go 1.17.
const skipObjectResolution = 0
