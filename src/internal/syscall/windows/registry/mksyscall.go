// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build generate

package registry

//go:generate go run ../../../../syscall/mksyscall_windows.go -output zsyscall_windows.go syscall.go
