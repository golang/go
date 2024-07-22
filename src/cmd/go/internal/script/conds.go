// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"fmt"
	"internal/syslist"
	"os"
	"runtime"
	"sync"
)

// DefaultConds returns a set of broadly useful script conditions.
//
// Run the 'help' command within a script engine to view a list of the available
// conditions.
func DefaultConds() map[string]Cond {
	conds := make(map[string]Cond)

	conds["GOOS"] = PrefixCondition(
		"runtime.GOOS == <suffix>",
		func(_ *State, suffix string) (bool, error) {
			if suffix == runtime.GOOS {
				return true, nil
			}
			if _, ok := syslist.KnownOS[suffix]; !ok {
				return false, fmt.Errorf("unrecognized GOOS %q", suffix)
			}
			return false, nil
		})

	conds["GOARCH"] = PrefixCondition(
		"runtime.GOARCH == <suffix>",
		func(_ *State, suffix string) (bool, error) {
			if suffix == runtime.GOARCH {
				return true, nil
			}
			if _, ok := syslist.KnownArch[suffix]; !ok {
				return false, fmt.Errorf("unrecognized GOOS %q", suffix)
			}
			return false, nil
		})

	conds["compiler"] = PrefixCondition(
		"runtime.Compiler == <suffix>",
		func(_ *State, suffix string) (bool, error) {
			if suffix == runtime.Compiler {
				return true, nil
			}
			switch suffix {
			case "gc", "gccgo":
				return false, nil
			default:
				return false, fmt.Errorf("unrecognized compiler %q", suffix)
			}
		})

	conds["root"] = BoolCondition("os.Geteuid() == 0", os.Geteuid() == 0)

	return conds
}

// Condition returns a Cond with the given summary and evaluation function.
func Condition(summary string, eval func(*State) (bool, error)) Cond {
	return &funcCond{eval: eval, usage: CondUsage{Summary: summary}}
}

type funcCond struct {
	eval  func(*State) (bool, error)
	usage CondUsage
}

func (c *funcCond) Usage() *CondUsage { return &c.usage }

func (c *funcCond) Eval(s *State, suffix string) (bool, error) {
	if suffix != "" {
		return false, ErrUsage
	}
	return c.eval(s)
}

// PrefixCondition returns a Cond with the given summary and evaluation function.
func PrefixCondition(summary string, eval func(*State, string) (bool, error)) Cond {
	return &prefixCond{eval: eval, usage: CondUsage{Summary: summary, Prefix: true}}
}

type prefixCond struct {
	eval  func(*State, string) (bool, error)
	usage CondUsage
}

func (c *prefixCond) Usage() *CondUsage { return &c.usage }

func (c *prefixCond) Eval(s *State, suffix string) (bool, error) {
	return c.eval(s, suffix)
}

// BoolCondition returns a Cond with the given truth value and summary.
// The Cond rejects the use of condition suffixes.
func BoolCondition(summary string, v bool) Cond {
	return &boolCond{v: v, usage: CondUsage{Summary: summary}}
}

type boolCond struct {
	v     bool
	usage CondUsage
}

func (b *boolCond) Usage() *CondUsage { return &b.usage }

func (b *boolCond) Eval(s *State, suffix string) (bool, error) {
	if suffix != "" {
		return false, ErrUsage
	}
	return b.v, nil
}

// OnceCondition returns a Cond that calls eval the first time the condition is
// evaluated. Future calls reuse the same result.
//
// The eval function is not passed a *State because the condition is cached
// across all execution states and must not vary by state.
func OnceCondition(summary string, eval func() (bool, error)) Cond {
	return &onceCond{eval: eval, usage: CondUsage{Summary: summary}}
}

type onceCond struct {
	once  sync.Once
	v     bool
	err   error
	eval  func() (bool, error)
	usage CondUsage
}

func (l *onceCond) Usage() *CondUsage { return &l.usage }

func (l *onceCond) Eval(s *State, suffix string) (bool, error) {
	if suffix != "" {
		return false, ErrUsage
	}
	l.once.Do(func() { l.v, l.err = l.eval() })
	return l.v, l.err
}

// CachedCondition is like Condition but only calls eval the first time the
// condition is evaluated for a given suffix.
// Future calls with the same suffix reuse the earlier result.
//
// The eval function is not passed a *State because the condition is cached
// across all execution states and must not vary by state.
func CachedCondition(summary string, eval func(string) (bool, error)) Cond {
	return &cachedCond{eval: eval, usage: CondUsage{Summary: summary, Prefix: true}}
}

type cachedCond struct {
	m     sync.Map
	eval  func(string) (bool, error)
	usage CondUsage
}

func (c *cachedCond) Usage() *CondUsage { return &c.usage }

func (c *cachedCond) Eval(_ *State, suffix string) (bool, error) {
	for {
		var ready chan struct{}

		v, loaded := c.m.Load(suffix)
		if !loaded {
			ready = make(chan struct{})
			v, loaded = c.m.LoadOrStore(suffix, (<-chan struct{})(ready))

			if !loaded {
				inPanic := true
				defer func() {
					if inPanic {
						c.m.Delete(suffix)
					}
					close(ready)
				}()

				b, err := c.eval(suffix)
				inPanic = false

				if err == nil {
					c.m.Store(suffix, b)
					return b, nil
				} else {
					c.m.Store(suffix, err)
					return false, err
				}
			}
		}

		switch v := v.(type) {
		case bool:
			return v, nil
		case error:
			return false, v
		case <-chan struct{}:
			<-v
		}
	}
}
