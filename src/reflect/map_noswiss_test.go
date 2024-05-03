// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.swissmap

package reflect_test

import (
	"internal/abi"
	"internal/goarch"
	. "reflect"
	"testing"
)

func testGCBitsMap(t *testing.T) {
	const bucketCount = abi.OldMapBucketCount

	hdr := make([]byte, bucketCount/goarch.PtrSize)

	verifyMapBucket := func(t *testing.T, k, e Type, m any, want []byte) {
		verifyGCBits(t, MapBucketOf(k, e), want)
		verifyGCBits(t, CachedBucketOf(TypeOf(m)), want)
	}
	verifyMapBucket(t,
		Tscalar, Tptr,
		map[Xscalar]Xptr(nil),
		join(hdr, rep(bucketCount, lit(0)), rep(bucketCount, lit(1)), lit(1)))
	verifyMapBucket(t,
		Tscalarptr, Tptr,
		map[Xscalarptr]Xptr(nil),
		join(hdr, rep(bucketCount, lit(0, 1)), rep(bucketCount, lit(1)), lit(1)))
	verifyMapBucket(t, Tint64, Tptr,
		map[int64]Xptr(nil),
		join(hdr, rep(bucketCount, rep(8/goarch.PtrSize, lit(0))), rep(bucketCount, lit(1)), lit(1)))
	verifyMapBucket(t,
		Tscalar, Tscalar,
		map[Xscalar]Xscalar(nil),
		empty)
	verifyMapBucket(t,
		ArrayOf(2, Tscalarptr), ArrayOf(3, Tptrscalar),
		map[[2]Xscalarptr][3]Xptrscalar(nil),
		join(hdr, rep(bucketCount*2, lit(0, 1)), rep(bucketCount*3, lit(1, 0)), lit(1)))
	verifyMapBucket(t,
		ArrayOf(64/goarch.PtrSize, Tscalarptr), ArrayOf(64/goarch.PtrSize, Tptrscalar),
		map[[64 / goarch.PtrSize]Xscalarptr][64 / goarch.PtrSize]Xptrscalar(nil),
		join(hdr, rep(bucketCount*64/goarch.PtrSize, lit(0, 1)), rep(bucketCount*64/goarch.PtrSize, lit(1, 0)), lit(1)))
	verifyMapBucket(t,
		ArrayOf(64/goarch.PtrSize+1, Tscalarptr), ArrayOf(64/goarch.PtrSize, Tptrscalar),
		map[[64/goarch.PtrSize + 1]Xscalarptr][64 / goarch.PtrSize]Xptrscalar(nil),
		join(hdr, rep(bucketCount, lit(1)), rep(bucketCount*64/goarch.PtrSize, lit(1, 0)), lit(1)))
	verifyMapBucket(t,
		ArrayOf(64/goarch.PtrSize, Tscalarptr), ArrayOf(64/goarch.PtrSize+1, Tptrscalar),
		map[[64 / goarch.PtrSize]Xscalarptr][64/goarch.PtrSize + 1]Xptrscalar(nil),
		join(hdr, rep(bucketCount*64/goarch.PtrSize, lit(0, 1)), rep(bucketCount, lit(1)), lit(1)))
	verifyMapBucket(t,
		ArrayOf(64/goarch.PtrSize+1, Tscalarptr), ArrayOf(64/goarch.PtrSize+1, Tptrscalar),
		map[[64/goarch.PtrSize + 1]Xscalarptr][64/goarch.PtrSize + 1]Xptrscalar(nil),
		join(hdr, rep(bucketCount, lit(1)), rep(bucketCount, lit(1)), lit(1)))
}
