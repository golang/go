// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T1 struct{}
type T2 struct{}

func f() {
	switch (T1{}) {
	case T1(T2{}):
	}
}
