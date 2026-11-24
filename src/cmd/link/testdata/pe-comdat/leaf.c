// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// clang --target=aarch64-w64-windows-gnu -O2 -ffunction-sections -c leaf.c -o leaf_windows_arm64.syso
int coff_leaf(int n) { return n + 1; }
