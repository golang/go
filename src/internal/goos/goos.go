// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package goos contains GOOS-specific constants.
package goos

// The next line makes 'go generate' write the zgoos*.go files with
// per-OS information, including constants named Is$GOOS for every
// known GOOS. The constant is 1 on the current system, 0 otherwise;
// multiplying by them is useful for defining GOOS-specific constants.
//
//go:generate go run gengoos.go
