// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//gofmt

package main

var _ = []struct{
	S       string
	Integer int
}{
	{
		S: "Hello World",
		Integer: 42,
	},
	{
		S: "\t",
		Integer: 42,
	},
	{
		S: "	", // an actual <tab>
		Integer: 42,
	},
	{
		S: `	`, // an actual <tab>
		Integer: 42,
	},
	{
		S: "\u0009",
		Integer: 42,
	},
}
