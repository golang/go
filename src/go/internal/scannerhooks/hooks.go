// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scannerhooks defines nonexported channels between parser and scanner.
// Ideally this package could be eliminated by adding API to scanner.
package scannerhooks

import "go/token"

var StringEnd func(scanner any) token.Pos
