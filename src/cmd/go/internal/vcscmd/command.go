// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vcscmd constructs VCS command lines, separating constant command
// words from option values and positional arguments supplied at run time.
package vcscmd

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
)

func New(bin lit, words ...lit) *Command {
	if bin == "" {
		return &Command{err: errors.New("empty command name")}
	}
	return (&Command{argv: []string{string(bin)}}).Add(words...)
}

// lit restricts callers outside this package to untyped string constants.
type lit string

// Command provides methods that record intent and help
// prevent building unsafe argv for VCS sub-process calls.
type Command struct {
	argv      []string
	separated bool
	err       error
}

// Add appends constant command words.
//
// It is an error to specify "--" or "--end-of-options".
func (c *Command) Add(words ...lit) *Command {
	if !c.beforeOperands() {
		return c
	}
	for _, w := range words {
		if w == "--" || w == "--end-of-options" {
			c.err = fmt.Errorf("unexpected operand separator %q", w)
			return c
		}
		c.argv = append(c.argv, string(w))
	}
	return c
}

func (c *Command) beforeOperands() bool {
	if c.err != nil {
		return false
	}
	switch {
	case len(c.argv) == 0:
		c.err = errors.New("empty command")
	case c.separated:
		c.err = errors.New("cannot add arguments after operands")
	}
	return c.err == nil
}

// Option appends a long option in the form --name=value.
//
// If [New] was invoked with "fossil", it also accepts "-r"
// written as "-r=value" and "-R" followed by an absolute
// repository path.
func (c *Command) Option(name lit, value string) *Command {
	if !c.beforeOperands() {
		return c
	}
	if c.argv[0] == "fossil" {
		switch name {
		case "-R":
			if !filepath.IsAbs(value) {
				c.err = fmt.Errorf("path %q is not absolute", value)
				return c
			}
			c.argv = append(c.argv, "-R", value)
			return c
		case "-r":
			c.argv = append(c.argv, "-r="+value)
			return c
		}
	}
	if !strings.HasPrefix(string(name), "--") || name == "--" || name == "--end-of-options" || strings.Contains(string(name), "=") {
		c.err = fmt.Errorf("invalid long option name %q", name)
		return c
	}
	c.argv = append(c.argv, string(name)+"="+value)
	return c
}

// Operands appends args and places the correct end of arguments
// separator.
//
// For git commands that parse a revision followed by paths, use
// [Command.GitRevision].
func (c *Command) Operands(args ...string) *Command {
	if !c.beforeOperands() {
		return c
	}
	c.separated = true
	if c.argv[0] == "git" {
		c.argv = append(c.argv, "--end-of-options")
	} else {
		c.argv = append(c.argv, "--")
	}
	c.argv = append(c.argv, args...)
	return c
}

// GitRevision allows safe specification of paths in a rev.
//
// It is an error to use GitRevision for non-git commands.
func (c *Command) GitRevision(rev string, paths ...string) *Command {
	if !c.beforeOperands() {
		return c
	}
	if c.argv[0] != "git" {
		c.err = errors.New("GitRevision requires a git command")
		return c
	}
	if rev == "--" {
		c.err = errors.New("Git revision cannot be --")
		return c
	}
	c.separated = true
	c.argv = append(c.argv, "--end-of-options", rev, "--")
	c.argv = append(c.argv, paths...)
	return c
}

// Build returns the full argv with VCS binary in argv[0].
//
// If Build returns an error, it is unsafe to use [Command].
func (c *Command) Build() ([]string, error) {
	if len(c.argv) == 0 && c.err == nil {
		return nil, errors.New("empty command")
	}
	return slices.Clone(c.argv), c.err
}
