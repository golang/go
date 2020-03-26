// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
)

// Label sends a label event to the exporter with the supplied tags.
func Label(ctx context.Context, tags ...Tag) context.Context {
	return dispatch(ctx, makeEvent(LabelType, sTags{}, tags))
}

// Label1 sends a label event to the exporter with the supplied tags.
func Label1(ctx context.Context, t1 Tag) context.Context {
	return dispatch(ctx, makeEvent(LabelType, sTags{t1}, nil))
}

// Label2 sends a label event to the exporter with the supplied tags.
func Label2(ctx context.Context, t1, t2 Tag) context.Context {
	return dispatch(ctx, makeEvent(LabelType, sTags{t1, t2}, nil))
}

// Label3 sends a label event to the exporter with the supplied tags.
func Label3(ctx context.Context, t1, t2, t3 Tag) context.Context {
	return dispatch(ctx, makeEvent(LabelType, sTags{t1, t2, t3}, nil))
}
