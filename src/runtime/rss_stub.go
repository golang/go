// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux && !darwin

package runtime

func sysRSSInit() {}

// sysRSS is unavailable on this platform.
func sysRSS() (rss, shared uint64, hasShared bool) { return 0, 0, false }
