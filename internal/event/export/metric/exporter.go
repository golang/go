// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package metric aggregates events into metrics that can be exported.
package metric

import (
	"context"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
)

var Entries = keys.New("metric_entries", "The set of metrics calculated for an event")

type Config struct {
	subscribers map[interface{}][]subscriber
}

type subscriber func(time.Time, label.Map, label.Label) Data

func (e *Config) subscribe(key label.Key, s subscriber) {
	if e.subscribers == nil {
		e.subscribers = make(map[interface{}][]subscriber)
	}
	e.subscribers[key] = append(e.subscribers[key], s)
}

func (e *Config) Exporter(output event.Exporter) event.Exporter {
	var mu sync.Mutex
	return func(ctx context.Context, ev core.Event, lm label.Map) context.Context {
		if !event.IsMetric(ev) {
			return output(ctx, ev, lm)
		}
		mu.Lock()
		defer mu.Unlock()
		var metrics []Data
		for index := 0; ev.Valid(index); index++ {
			l := ev.Label(index)
			if !l.Valid() {
				continue
			}
			id := l.Key()
			if list := e.subscribers[id]; len(list) > 0 {
				for _, s := range list {
					metrics = append(metrics, s(ev.At(), lm, l))
				}
			}
		}
		lm = label.MergeMaps(label.NewMap(Entries.Of(metrics)), lm)
		return output(ctx, ev, lm)
	}
}
