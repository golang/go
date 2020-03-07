// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides support for telemetry tagging.
// This package is a thin shim over contexts with the main addition being the
// the ability to observe when contexts get tagged with new values.
package tag

import (
	"golang.org/x/tools/internal/telemetry/event"
)

type Key = event.Key

var (
	With = event.Label
	Get  = event.Tags
	Of   = event.TagOf
)
