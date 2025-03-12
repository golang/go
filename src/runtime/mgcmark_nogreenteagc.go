// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.greenteagc

package runtime

func (s *mspan) markBitsForIndex(objIndex uintptr) markBits {
	bytep, mask := s.gcmarkBits.bitp(objIndex)
	return markBits{bytep, mask, objIndex}
}

func (s *mspan) markBitsForBase() markBits {
	return markBits{&s.gcmarkBits.x, uint8(1), 0}
}

func tryDeferToSpanScan(p uintptr, gcw *gcWork) bool {
	return false
}

func (s *mspan) initInlineMarkBits() {
}

func (s *mspan) mergeInlineMarks(to *gcBits) {
	throw("unimplemented")
}

func gcUsesSpanInlineMarkBits(_ uintptr) bool {
	return false
}

func (s *mspan) inlineMarkBits() *spanInlineMarkBits {
	return nil
}

func (s *mspan) scannedBitsForIndex(objIndex uintptr) markBits {
	throw("unimplemented")
	return markBits{}
}

type spanInlineMarkBits struct {
}

func (q *spanInlineMarkBits) tryAcquire() bool {
	return false
}

type spanQueue struct {
	_ uint32 // To match alignment padding requirements for atomically-accessed variables in workType.
}

func (q *spanQueue) empty() bool {
	return true
}

func (q *spanQueue) size() int {
	return 0
}

type localSpanQueue struct {
}

func (q *localSpanQueue) drain() bool {
	return false
}

func (q *localSpanQueue) empty() bool {
	return true
}

type objptr uintptr

func (w *gcWork) tryGetSpan(steal bool) objptr {
	return 0
}

func scanSpan(p objptr, gcw *gcWork) {
	throw("unimplemented")
}
