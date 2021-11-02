// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type s struct {
	f func()
}

func f() {
	ch := make(chan struct{}, 1)
	_ = [...]struct{ slice []s }{
		{}, {}, {}, {},
		{
			slice: []s{
				{
					f: func() { ch <- struct{}{} },
				},
			},
		},
	}
}
