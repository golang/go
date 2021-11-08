// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c

import (
	"a"
	"b"
)

type HandlerFunc func(*string)

func RouterInit() {
	//home API
	homeIndex := &a.IndexController{}
	GET("/home/index/index", homeIndex.Index)
	//admin API
	adminIndex := &b.IndexController{}
	GET("/admin/index/index", adminIndex.Index)
	return
}

func GET(path string, handlers ...HandlerFunc) {
	return
}
