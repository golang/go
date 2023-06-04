// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package toolchain

import (
	"context"
	"fmt"
	"os"

	"cmd/go/internal/base"
	"cmd/go/internal/gover"
)

// A Switcher collects errors to be reported and then decides
// between reporting the errors or switching to a new toolchain
// to resolve them.
//
// The client calls [Switcher.Error] repeatedly with errors encountered
// and then calls [Switcher.Switch]. If the errors included any
// *gover.TooNewErrors (potentially wrapped) and switching is
// permitted by GOTOOLCHAIN, Switch switches to a new toolchain.
// Otherwise Switch prints all the errors using base.Error.
type Switcher struct {
	TooNew *gover.TooNewError // max go requirement observed
	Errors []error            // errors collected so far
}

// Error reports the error to the Switcher,
// which saves it for processing during Switch.
func (s *Switcher) Error(err error) {
	s.Errors = append(s.Errors, err)
	s.addTooNew(err)
}

// addTooNew adds any TooNew errors that can be found in err.
func (s *Switcher) addTooNew(err error) {
	switch err := err.(type) {
	case interface{ Unwrap() []error }:
		for _, e := range err.Unwrap() {
			s.addTooNew(e)
		}

	case interface{ Unwrap() error }:
		s.addTooNew(err.Unwrap())

	case *gover.TooNewError:
		if s.TooNew == nil ||
			gover.Compare(err.GoVersion, s.TooNew.GoVersion) > 0 ||
			gover.Compare(err.GoVersion, s.TooNew.GoVersion) == 0 && err.What < s.TooNew.What {
			s.TooNew = err
		}
	}
}

// NeedSwitch reports whether Switch would attempt to switch toolchains.
func (s *Switcher) NeedSwitch() bool {
	return s.TooNew != nil && (HasAuto() || HasPath())
}

// Switch decides whether to switch to a newer toolchain
// to resolve any of the saved errors.
// It switches if toolchain switches are permitted and there is at least one TooNewError.
//
// If Switch decides not to switch toolchains, it prints the errors using base.Error and returns.
//
// If Switch decides to switch toolchains but cannot identify a toolchain to use.
// it prints the errors along with one more about not being able to find the toolchain
// and returns.
//
// Otherwise, Switch prints an informational message giving a reason for the
// switch and the toolchain being invoked and then switches toolchains.
// This operation never returns.
func (s *Switcher) Switch(ctx context.Context) {
	if !s.NeedSwitch() {
		for _, err := range s.Errors {
			base.Error(err)
		}
		return
	}

	// Switch to newer Go toolchain if necessary and possible.
	tv, err := NewerToolchain(ctx, s.TooNew.GoVersion)
	if err != nil {
		for _, err := range s.Errors {
			base.Error(err)
		}
		base.Error(fmt.Errorf("switching to go >= %v: %w", s.TooNew.GoVersion, err))
		return
	}

	fmt.Fprintf(os.Stderr, "go: %v requires go >= %v; switching to %v\n", s.TooNew.What, s.TooNew.GoVersion, tv)
	SwitchTo(tv)
	panic("unreachable")
}

// SwitchOrFatal attempts a toolchain switch based on the information in err
// and otherwise falls back to base.Fatal(err).
func SwitchOrFatal(ctx context.Context, err error) {
	var s Switcher
	s.Error(err)
	s.Switch(ctx)
	base.Exit()
}
