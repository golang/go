// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
)

// Record sends a label event to the exporter with the supplied tags.
func Record(ctx context.Context, tags ...Tag) context.Context {
	return dispatch(ctx, makeEvent(RecordType, sTags{}, tags))
}

// Record1 sends a label event to the exporter with the supplied tags.
func Record1(ctx context.Context, t1 Tag) context.Context {
	return dispatch(ctx, makeEvent(RecordType, sTags{t1}, nil))
}

// Record2 sends a label event to the exporter with the supplied tags.
func Record2(ctx context.Context, t1, t2 Tag) context.Context {
	return dispatch(ctx, makeEvent(RecordType, sTags{t1, t2}, nil))
}

// Record3 sends a label event to the exporter with the supplied tags.
func Record3(ctx context.Context, t1, t2, t3 Tag) context.Context {
	return dispatch(ctx, makeEvent(RecordType, sTags{t1, t2, t3}, nil))
}
