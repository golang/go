// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This tells the go tool that this package builds using cgo.
// TODO: Once we stop using Make, this import can move into cgo.go.

package cgo

import "C"
