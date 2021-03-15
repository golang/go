// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type ATypeWithALoooooongName interface { // a long name, so the type descriptor symbol name is mangled
	M()
}
