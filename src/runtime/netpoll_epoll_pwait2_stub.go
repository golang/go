// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package runtime

// netpollEpollPwait2Init is a no-op on platforms that do not support epoll_pwait2.
func netpollEpollPwait2Init() {}
