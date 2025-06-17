// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// This type and constants are for encoding different
// kinds of bounds check failures.
type BoundsErrorCode uint8

const (
	BoundsIndex      BoundsErrorCode = iota // s[x], 0 <= x < len(s) failed
	BoundsSliceAlen                         // s[?:x], 0 <= x <= len(s) failed
	BoundsSliceAcap                         // s[?:x], 0 <= x <= cap(s) failed
	BoundsSliceB                            // s[x:y], 0 <= x <= y failed (but boundsSliceA didn't happen)
	BoundsSlice3Alen                        // s[?:?:x], 0 <= x <= len(s) failed
	BoundsSlice3Acap                        // s[?:?:x], 0 <= x <= cap(s) failed
	BoundsSlice3B                           // s[?:x:y], 0 <= x <= y failed (but boundsSlice3A didn't happen)
	BoundsSlice3C                           // s[x:y:?], 0 <= x <= y failed (but boundsSlice3A/B didn't happen)
	BoundsConvert                           // (*[x]T)(s), 0 <= x <= len(s) failed
)
