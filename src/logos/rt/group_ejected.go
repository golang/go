// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build logosejected

package rt

// logosGroupAcquire lends no slot in ejected code: goroutine tags exist
// only in binaries built by logos with the runtime group field.
func logosGroupAcquire() uint32 { return 0 }

// logosGroupRelease is a no-op in ejected code: no slot was lent.
func logosGroupRelease(slot uint32) {}

// logosTagGroup is a no-op in ejected code: goroutine tags exist only in
// binaries built by logos with the runtime group field.
func logosTagGroup(slot uint32) {}

// logosCancelGroup is a no-op in ejected code: netpoll waiters are woken
// by the logos runtime only.
func logosCancelGroup(slot uint32) {}
