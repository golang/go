// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"sort"
	"strings"
)

func f[int any](x int) {
	x = 0 /* ERRORx "cannot use 0.*(as int.*with int declared at|type parameter)" */
}

// test case from issue

type Set[T comparable] map[T]struct{}

func (s *Set[string]) String() string {
	keys := make([]string, 0, len(*s))
	for k := range *s {
		keys = append(keys, k)
	}
	sort.Strings(keys /* ERRORx "cannot use keys.*with string declared at.*|type parameter" */ )
	return strings /* ERROR "cannot use strings.Join" */ .Join(keys /* ERRORx "cannot use keys.*with string declared at.*|type parameter" */ , ",")
}
