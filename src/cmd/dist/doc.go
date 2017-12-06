// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Dist helps bootstrap, build, and test the Go distribution.
//
// Usage:
//   go tool dist [command]
//
// The commands are:
//   banner         print installation banner
//   bootstrap      rebuild everything
//   clean          deletes all built files
//   env [-p]       print environment (-p: include $PATH)
//   install [dir]  install individual directory
//   list [-json]   list all supported platforms
//   test [-h]      run Go test(s)
//   version        print Go version
package main
