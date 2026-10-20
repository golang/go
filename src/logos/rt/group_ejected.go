// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build logosejected

package rt

// logosTagGroup is a no-op in ejected code: goroutine tags exist only in
// binaries built by logos with the runtime group field.
func logosTagGroup(id uint64) {}

// logosCancelGroup is a no-op in ejected code: netpoll waiters are woken
// by the logos runtime only.
func logosCancelGroup(id uint64) {}
