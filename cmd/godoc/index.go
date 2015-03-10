// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

func indexDirectoryDefault(dir string) bool {
	return dir != "/pkg" && !strings.HasPrefix(dir, "/pkg/")
}
