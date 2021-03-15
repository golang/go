// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gofuzz
// +build gofuzz

package big

import (
	"strings"
)

func FuzzCmp(data []byte) int {
	if !IsDivisibleBy(len(data), 2) {
		return -1
	}
	half := len(data) / 2

	halfOne := data[:half]
	x, err := natFromString(string(halfOne))
	if err != nil {
		return 0
	}

	halfTwo := data[half:]
	y, err := natFromString(string(halfTwo))
	if err != nil {
		return 0
	}

	x.cmp(y)
	return 1
}

func FuzzExpNN(data []byte) int {
	if !IsDivisibleBy(len(data), 3) {
		return -1
	}
	firstThird := len(data) / 3
	x, err := natFromString(string(data[:firstThird]))
	if err != nil {
		return 0
	}

	secondThird := firstThird * 2
	y, err := natFromString(string(data[firstThird:secondThird]))
	if err != nil {
		return 0
	}

	z, err := natFromString(string(data[secondThird:]))
	if err != nil {
		return 0
	}

	p := nat(nil).expNN(x, y, z)

	const n = 165
	p = p.shl(p, n)
	return 1
}

func IsDivisibleBy(n int, divisibleby int) bool {
	return (n % divisibleby) == 0
}

func natFromString(s string) (nat, error) {
	x, _, _, err := nat(nil).scan(strings.NewReader(s), 0, false)
	if err != nil {
		return nil, err
	}
	return x, nil
}
