// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package asmdecl

var asmArchLoong64 = asmArch{name: "loong64", bigEndian: false, stack: "R3", lr: true}

func additionalArches() []*asmArch {
	return []*asmArch{&asmArchLoong64}
}
