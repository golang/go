// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xerrors implements functions to manipulate errors.
//
// This package is based on the Go 2 proposal for error values:
//   https://golang.org/design/29934-error-values
//
// These functions were incorporated into the standard library's errors package
// in Go 1.13:
// - Is
// - As
// - Unwrap
//
// Also, Errorf's %w verb was incorporated into fmt.Errorf.
//
// Use this package to get equivalent behavior in all supported Go versions.
//
// No other features of this package were included in Go 1.13, and at present
// there are no plans to include any of them.
package xerrors // import "golang.org/x/xerrors"
