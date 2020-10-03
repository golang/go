// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

// Custom code actions that aren't explicitly stated in LSP
const (
	GoTest CodeActionKind = "goTest"
	// TODO: Add GoGenerate, RegenerateCgo etc.
)
