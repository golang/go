// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type _ interface {
	int
	(int)
	(*int)
	*([]byte)
	~(int)
	(int) | (string)
	(int) | ~(string)
	(/* ERROR unexpected ~ */ ~int)
	(int /* ERROR unexpected \| */ | /* ERROR unexpected string */ string /* ERROR unexpected \) */ )
}
