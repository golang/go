// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import "sync/atomic"

var _ *atomic.Pointer[int] = &(&Hot[int]{}).p
