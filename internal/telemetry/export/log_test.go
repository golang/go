// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export_test

import (
	"context"
	"errors"
	"os"
	"time"

	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/telemetry/export"
)

func ExampleLog() {
	ctx := context.Background()
	event.SetExporter(timeFixer(export.LogWriter(os.Stdout, false)))
	anInt := event.NewIntKey("myInt", "an integer")
	aString := event.NewStringKey("myString", "a string")
	event.Print(ctx, "my event", anInt.Of(6))
	event.Error(ctx, "error event", errors.New("an error"), aString.Of("some string value"))
	// Output:
	// 2020/03/05 14:27:48 my event
	// 	myInt=6
	// 2020/03/05 14:27:48 error event: an error
	// 	myString="some string value"
}

func timeFixer(output event.Exporter) event.Exporter {
	at, _ := time.Parse(time.RFC3339Nano, "2020-03-05T14:27:48Z")
	return func(ctx context.Context, ev event.Event, tagMap event.TagMap) context.Context {
		ev.At = at
		return output(ctx, ev, tagMap)
	}
}
