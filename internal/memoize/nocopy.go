// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memoize

// NoCopy is a type with no public methods that will trigger a vet check if it
// is ever copied.
// You can embed this in any type intended to be used as a value. This helps
// avoid accidentally holding a copy of a value instead of the value itself.
type NoCopy struct {
	noCopy noCopy
}

// noCopy may be embedded into structs which must not be copied
// after the first use.
//
// See https://golang.org/issues/8005#issuecomment-190753527
// for details.
type noCopy struct{}

// Lock is a no-op used by -copylocks checker from `go vet`.
func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}
