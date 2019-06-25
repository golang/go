// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"time"
)

// detatch returns a context that keeps all the values of its parent context
// but detatches from the cancellation and error handling.
func detatchContext(ctx context.Context) context.Context { return detatchedContext{ctx} }

type detatchedContext struct{ parent context.Context }

func (v detatchedContext) Deadline() (time.Time, bool)       { return time.Time{}, false }
func (v detatchedContext) Done() <-chan struct{}             { return nil }
func (v detatchedContext) Err() error                        { return nil }
func (v detatchedContext) Value(key interface{}) interface{} { return v.parent.Value(key) }
