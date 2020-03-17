// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"sync/atomic"
	"unsafe"
)

type Exporter interface {
	// ProcessEvent is a function that handles all events.
	// This is called with all events that should be delivered to the exporter
	// along with the context in which that event ocurred.
	// This method is called synchronously from the event call site, so it should
	// return quickly so as not to hold up user code.
	ProcessEvent(context.Context, Event) (context.Context, Event)
}

var (
	exporter unsafe.Pointer
)

func SetExporter(e Exporter) {
	p := unsafe.Pointer(&e)
	if e == nil {
		// &e is always valid, and so p is always valid, but for the early abort
		// of ProcessEvent to be efficient it needs to make the nil check on the
		// pointer without having to dereference it, so we make the nil interface
		// also a nil pointer
		p = nil
	}
	atomic.StorePointer(&exporter, p)
}

func ProcessEvent(ctx context.Context, ev Event) (context.Context, Event) {
	exporterPtr := (*Exporter)(atomic.LoadPointer(&exporter))
	if exporterPtr == nil {
		return ctx, ev
	}
	// and now also hand the event of to the current exporter
	return (*exporterPtr).ProcessEvent(ctx, ev)
}
