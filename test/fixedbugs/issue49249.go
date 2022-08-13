// compile -l

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() int {
	var a, b struct {
		s struct {
			s struct {
				byte
				float32
			}
		}
	}
	_ = a

	return func() int {
		return func() int {
			a = struct {
				s struct {
					s struct {
						byte
						float32
					}
				}
			}{b.s}
			return 0
		}()
	}()
}

func g() int {
	var a, b struct {
		s [1][1]struct {
			byte
			float32
		}
	}
	_ = a

	return func() int {
		return func() int {
			a = struct {
				s [1][1]struct {
					byte
					float32
				}
			}{b.s}
			return 0
		}()
	}()
}
