// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonscan

import "internal/cpu"

const avx2BlockLen = 32

func indexStringSpecial(p *byte, n int) int {
	if n >= avx2BlockLen && cpu.X86.HasAVX2 {
		return indexStringSpecialAVX2(p, n)
	}
	return indexStringSpecialSSE2(p, n)
}

func indexEscapeHTML(p *byte, n int) int {
	if n >= avx2BlockLen && cpu.X86.HasAVX2 {
		return indexEscapeHTMLAVX2(p, n)
	}
	return indexEscapeHTMLSSE2(p, n)
}

//go:noescape
func indexStringSpecialSSE2(p *byte, n int) int

//go:noescape
func indexStringSpecialAVX2(p *byte, n int) int

//go:noescape
func indexEscapeHTMLSSE2(p *byte, n int) int

//go:noescape
func indexEscapeHTMLAVX2(p *byte, n int) int
