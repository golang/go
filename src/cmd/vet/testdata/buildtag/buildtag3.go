// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the buildtag checker.

//go:build good
// ERRORNEXT "[+]build lines do not match //go:build condition"
// +build bad

package testdata

var _ = `
// +build notacomment
`
