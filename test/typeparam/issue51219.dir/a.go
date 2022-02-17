// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// Type I is the first basic test for the issue, which relates to a type that is recursive
// via a type constraint.  (In this test, I -> IConstraint -> MyStruct -> I.)
type JsonRaw []byte

type MyStruct struct {
	x *I[JsonRaw]
}

type IConstraint interface {
	JsonRaw | MyStruct
}

type I[T IConstraint] struct {
}

// The following types form an even more complex recursion (through two type
// constraints), and model the actual types in the issue (#51219) more closely.
// However, they don't reveal any new issue. But it seems useful to leave this
// complex set of types in a test in case it might be broken by future changes.

type Message struct {
	Interaction *Interaction[JsonRaw] `json:"interaction,omitempty"`
}

type ResolvedDataConstraint interface {
	User | Message
}

type Snowflake uint64

type ResolvedData[T ResolvedDataConstraint] map[Snowflake]T

type User struct {
}

type Resolved struct {
	Users ResolvedData[User] `json:"users,omitempty"`
}

type resolvedInteractionWithOptions struct {
	Resolved Resolved `json:"resolved,omitempty"`
}

type UserCommandInteractionData struct {
	resolvedInteractionWithOptions
}

type InteractionDataConstraint interface {
	JsonRaw | UserCommandInteractionData
}

type Interaction[DataT InteractionDataConstraint] struct {
}
