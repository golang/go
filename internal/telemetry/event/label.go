// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"time"
)

// Label sends a label event to the exporter with the supplied tags.
func Label(ctx context.Context, tags ...Tag) context.Context {
	return ProcessEvent(ctx, Event{
		Type: LabelType,
		At:   time.Now(),
		Tags: tags,
	})
}
