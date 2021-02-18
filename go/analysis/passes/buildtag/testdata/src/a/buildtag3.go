// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// want +3 `[+]build lines do not match //go:build condition`

//go:build good
// +build bad

package a

var _ = `
// +build notacomment
`
