// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

//go:embed issue48230.go // ERROR `go:embed only allowed in Go files that import "embed"`
var _ string
