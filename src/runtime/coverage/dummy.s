// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The runtime package uses //go:linkname to push a few functions into this
// package but we still need a .s file so the Go tool does not pass -complete
// to 'go tool compile' so the latter does not complain about Go functions
// with no bodies.
