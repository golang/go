// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the slog checker.

//go:build go1.21

package a

import (
	"fmt"
	"log/slog"
)

func F() {
	var (
		l *slog.Logger
		r slog.Record
	)

	// Unrelated call.
	fmt.Println("ok")

	// Valid calls.
	slog.Info("msg")
	slog.Info("msg", "a", 1)
	l.Debug("msg", "a", 1)
	l.With("a", 1)
	slog.Warn("msg", slog.Int("a", 1), "k", 2)
	l.WarnCtx(nil, "msg", "a", 1, slog.Int("b", 2), slog.Int("c", 3), "d", 4)
	r.Add("a", 1, "b", 2)

	// bad
	slog.Info("msg", 1)                     // want `slog.Info arg "1" should be a string or a slog.Attr`
	l.Info("msg", 2)                        // want `slog.Logger.Info arg "2" should be a string or a slog.Attr`
	slog.Debug("msg", "a")                  // want `call to slog.Debug missing a final value`
	slog.Warn("msg", slog.Int("a", 1), "k") // want `call to slog.Warn missing a final value`
	slog.ErrorCtx(nil, "msg", "a", 1, "b")  // want `call to slog.ErrorCtx missing a final value`
	r.Add("K", "v", "k")                    // want `call to slog.Record.Add missing a final value`
	l.With("a", "b", 2)                     // want `slog.Logger.With arg "2" should be a string or a slog.Attr`

	slog.Log(nil, slog.LevelWarn, "msg", "a", "b", 2) // want `slog.Log arg "2" should be a string or a slog.Attr`

	// Skip calls with spread args.
	var args []any
	slog.Info("msg", args...)

	// The variadic part of all the calls below begins with an argument of
	// static type any, followed by an integer.
	// Even though the we don't know the dynamic type of the first arg, and thus
	// whether it is a key, an Attr, or something else, the fact that the
	// following integer arg cannot be a key allows us to assume that we should
	// expect a key to follow.
	var a any = "key"

	// This is a valid call for which  we correctly produce no diagnostic.
	slog.Info("msg", a, 7, "key2", 5)

	// This is an invalid call because the final value is missing, but we can't
	// be sure that's the reason.
	slog.Info("msg", a, 7, "key2") // want `call to slog.Info has a missing or misplaced value`

	// Here our guess about the unknown arg (a) is wrong: we assume it's a string, but it's an Attr.
	// Therefore the second argument should be a key, but it is a number.
	// Ideally our diagnostic would pinpoint the problem, but we don't have enough information.
	a = slog.Int("a", 1)
	slog.Info("msg", a, 7, "key2") // want `call to slog.Info has a missing or misplaced value`

	// This call is invalid for the same reason as the one above, but we can't
	// detect that.
	slog.Info("msg", a, 7, "key2", 5)

	// Another invalid call we can't detect. Here the first argument is wrong.
	a = 1
	slog.Info("msg", a, 7, "b", 5)
}
