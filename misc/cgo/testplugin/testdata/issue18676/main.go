// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The bug happened like this:
// 1) The main binary adds an itab for *json.UnsupportedValueError / error
//    (concrete type / interface type).  This itab goes in hash bucket 0x111.
// 2) The plugin adds that same itab again.  That makes a cycle in the itab
//    chain rooted at hash bucket 0x111.
// 3) The main binary then asks for the itab for *dynamodbstreamsevt.Event /
//    json.Unmarshaler.  This itab happens to also live in bucket 0x111.
//    The lookup code goes into an infinite loop searching for this itab.
// The code is carefully crafted so that the two itabs are both from the
// same bucket, and so that the second itab doesn't exist in
// the itab hashmap yet (so the entire linked list must be searched).
package main

import (
	"encoding/json"
	"plugin"
	"testplugin/issue18676/dynamodbstreamsevt"
)

func main() {
	plugin.Open("plugin.so")

	var x interface{} = (*dynamodbstreamsevt.Event)(nil)
	if _, ok := x.(json.Unmarshaler); !ok {
		println("something")
	}
}
