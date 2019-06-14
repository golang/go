// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag adds support for telemetry tags.
package tag

import "context"

type Map interface{}

type Key interface {
	Name() string
}

type Mutator interface {
	Mutate(Map) (Map, error)
}

type nullMutator struct{}

func (nullMutator) Mutate(Map) (Map, error) { return nil, nil }

var (
	New         = func(ctx context.Context, mutator ...Mutator) (context.Context, error) { return ctx, nil }
	NewContext  = func(ctx context.Context, m Map) context.Context { return ctx }
	FromContext = func(ctx context.Context) Map { return nil }
	Delete      = func(k Key) Mutator { return nullMutator{} }
	Insert      = func(k Key, v string) Mutator { return nullMutator{} }
	Update      = func(k Key, v string) Mutator { return nullMutator{} }
	Upsert      = func(k Key, v string) Mutator { return nullMutator{} }
)
