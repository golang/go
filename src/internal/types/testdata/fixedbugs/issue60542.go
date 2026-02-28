// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func Clip[S ~[]E, E any](s S) S {
	return s
}

var versions func()
var _ = Clip /* ERROR "in call to Clip, S (type func()) does not satisfy ~[]E" */ (versions)
