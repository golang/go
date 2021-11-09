// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build generate
// +build generate

package windows

//go:generate go run golang.org/x/sys/windows/mkwinsyscall -output zsyscall_windows.go eventlog.go service.go syscall_windows.go security_windows.go
