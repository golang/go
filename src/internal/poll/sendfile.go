// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

var TestHookDidSendFile = func(dstFD *FD, src int, written int64, err error, handled bool) {}
