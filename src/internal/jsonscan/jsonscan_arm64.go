// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonscan

//go:noescape
func indexStringSpecial(p *byte, n int) int

//go:noescape
func indexEscapeHTML(p *byte, n int) int
