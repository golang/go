// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"time"
)

func Record(ctx context.Context, tags ...Tag) {
	ProcessEvent(ctx, Event{
		Type: RecordType,
		At:   time.Now(),
		Tags: newTagSet(tags),
	})
}
