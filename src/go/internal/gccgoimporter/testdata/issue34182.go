// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue34182

type T1 struct {
	f *T2
}

type T2 struct {
	f T3
}

type T3 struct {
	*T2
}
