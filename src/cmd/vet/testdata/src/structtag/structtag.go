// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package structtag

import "encoding/json"

type StructTagTest struct {
	A int "hello" // ERROR "`hello` not compatible with reflect.StructTag.Get: bad syntax for struct tag pair"
}

func Issue30465() {
	type T1 struct {
		X string `json:"x"`
	}
	type T2 struct {
		T1
		X string `json:"x"`
	}
	var t2 T2
	json.Marshal(&t2)
}
