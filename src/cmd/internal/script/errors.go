// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"errors"
	"fmt"
)

// ErrUnexpectedSuccess indicates that a script command that was expected to
// fail (as indicated by a "!" prefix) instead completed successfully.
var ErrUnexpectedSuccess = errors.New("unexpected success")

// A CommandError describes an error resulting from attempting to execute a
// specific command.
type CommandError struct {
	File string
	Line int
	Op   string
	Args []string
	Err  error
}

func cmdError(cmd *command, err error) *CommandError {
	return &CommandError{
		File: cmd.file,
		Line: cmd.line,
		Op:   cmd.name,
		Args: cmd.args,
		Err:  err,
	}
}

func (e *CommandError) Error() string {
	if len(e.Args) == 0 {
		return fmt.Sprintf("%s:%d: %s: %v", e.File, e.Line, e.Op, e.Err)
	}
	return fmt.Sprintf("%s:%d: %s %s: %v", e.File, e.Line, e.Op, quoteArgs(e.Args), e.Err)
}

func (e *CommandError) Unwrap() error { return e.Err }

// A UsageError reports the valid arguments for a command.
//
// It may be returned in response to invalid arguments.
type UsageError struct {
	Name    string
	Command Cmd
}

func (e *UsageError) Error() string {
	usage := e.Command.Usage()
	suffix := ""
	if usage.Async {
		suffix = " [&]"
	}
	return fmt.Sprintf("usage: %s %s%s", e.Name, usage.Args, suffix)
}

// ErrUsage may be returned by a Command to indicate that it was called with
// invalid arguments; its Usage method may be called to obtain details.
var ErrUsage = errors.New("invalid usage")
