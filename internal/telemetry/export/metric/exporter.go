// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package metric aggregates events into metrics that can be exported.
package metric

import (
	"context"

	"golang.org/x/tools/internal/telemetry/event"
)

var Entries = event.NewKey("metric_entries", "The set of metrics calculated for an event")

type Exporter struct {
	subscribers map[interface{}][]subscriber
}

type subscriber func(context.Context, event.Event, event.Tag) Data

func (e *Exporter) subscribe(key event.Key, s subscriber) {
	if e.subscribers == nil {
		e.subscribers = make(map[interface{}][]subscriber)
	}
	ident := key.Identity()
	e.subscribers[ident] = append(e.subscribers[ident], s)
}

func (e *Exporter) ProcessEvent(ctx context.Context, ev event.Event) (context.Context, event.Event) {
	if !ev.IsRecord() {
		return ctx, ev
	}
	var metrics []Data
	for i := ev.Tags.Iterator(); i.Next(); {
		tag := i.Value()
		id := tag.Key().Identity()
		if list := e.subscribers[id]; len(list) > 0 {
			for _, s := range list {
				metrics = append(metrics, s(ctx, ev, tag))
			}
		}
	}
	ev.Tags = ev.Tags.Add(Entries.Of(metrics))
	return ctx, ev
}
