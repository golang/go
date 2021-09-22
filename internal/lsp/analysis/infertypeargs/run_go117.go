// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package infertypeargs

import "golang.org/x/tools/go/analysis"

// This analyzer only relates to go1.18+, and uses the types.CheckExpr API that
// was added in Go 1.13.
func run(pass *analysis.Pass) (interface{}, error) {
	return nil, nil
}
