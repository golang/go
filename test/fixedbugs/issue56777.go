// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func fn(setText []rune, negate bool) int {
	ranges := []singleRange{}

	if len(setText) > 0 {
		fillFirst := false
		l := len(setText)
		if negate {
			if setText[0] == 0 {
				setText = setText[1:]
			} else {
				l++
				fillFirst = true
			}
		}

		if l%2 == 0 {
			ranges = make([]singleRange, l/2)
		} else {
			ranges = make([]singleRange, l/2+1)
		}

		first := true
		if fillFirst {
			ranges[0] = singleRange{first: 0}
			first = false
		}

		i := 0
		for _, r := range setText {
			if first {
				// lower bound in a new range
				ranges[i] = singleRange{first: r}
				first = false
			} else {
				ranges[i].last = r - 1
				i++
				first = true
			}
		}
	}

	return len(ranges)
}

type singleRange struct {
	first rune
	last  rune
}
