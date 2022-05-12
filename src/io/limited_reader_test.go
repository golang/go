// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func testLimitedReader(t *testing.T, data, limit int) {
	if data < 0 {
		return
	}
	sample := strings.Repeat("a", data)
	r := strings.NewReader(sample)
	sentinel := errors.New("reached read limit")
	lr := &io.LimitedReader{R: r, N: int64(limit), Err: sentinel}

	var buf strings.Builder
	_, err := io.Copy(&buf, lr)
	wantlen := limit
	wanterr := sentinel
	if limit >= data {
		wanterr = nil
		wantlen = data
	}
	if wantlen < 0 {
		wantlen = 0
	}

	if s := buf.String(); len(s) != wantlen {
		t.Fatalf("want len %d; got %d", wantlen, len(s))
	}
	if err != wanterr {
		t.Fatalf("want err %v; got %v", wanterr, err)
	}
}

func TestLimitedReader(t *testing.T) {
	for _, tc := range [][2]int{
		{0, -1},
		{0, 0},
		{0, 5},
		{5, -1},
		{5, 4},
		{5, 5},
		{5, 6},
		{15_000, 14_999},
		{15_000, 15_000},
		{15_000, 15_001},
	} {
		testLimitedReader(t, tc[0], tc[1])
	}
}
