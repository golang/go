// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build logosejected

package rt

func logosGroupAcquire() uint32 { return 0 }

func logosGroupRelease(slot uint32) {}

func logosTagGroup(slot uint32) {}

func logosCancelGroup(slot uint32) {}
