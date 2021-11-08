// Copyright 2021 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file

package a

type W struct {
	M func(string) string
}

func FM(m string) func(W) {
	return func(pw W) {
		pw.M = func(string) string {
			return m
		}
	}
}
