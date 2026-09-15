// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exists only so the Go compiler permits bodyless function
// declarations for `lastmoduleinit` and `doInit` in plugin_windows.go
// (their bodies are provided by package runtime via //go:linkname,
// resolved at link time).
