// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mvs implements Minimal Version Selection.
// See https://research.swtch.com/vgo-mvs.
package mvs

import (
	"fmt"
	"sort"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/module"
	"cmd/go/internal/par"
)

// A Reqs is the requirement graph on which Minimal Version Selection (MVS) operates.
//
// The version strings are opaque except for the special version "none"
// (see the documentation for module.Version). In particular, MVS does not
// assume that the version strings are semantic versions; instead, the Max method
// gives access to the comparison operation.
//
// It must be safe to call methods on a Reqs from multiple goroutines simultaneously.
// Because a Reqs may read the underlying graph from the network on demand,
// the MVS algorithms parallelize the traversal to overlap network delays.
type Reqs interface {
	// Required returns the module versions explicitly required by m itself.
	// The caller must not modify the returned list.
	Required(m module.Version) ([]module.Version, error)

	// Max returns the maximum of v1 and v2 (it returns either v1 or v2).
	//
	// For all versions v, Max(v, "none") must be v,
	// and for the tanget passed as the first argument to MVS functions,
	// Max(target, v) must be target.
	//
	// Note that v1 < v2 can be written Max(v1, v2) != v1
	// and similarly v1 <= v2 can be written Max(v1, v2) == v2.
	Max(v1, v2 string) string

	// Upgrade returns the upgraded version of m,
	// for use during an UpgradeAll operation.
	// If m should be kept as is, Upgrade returns m.
	// If m is not yet used in the build, then m.Version will be "none".
	// More typically, m.Version will be the version required
	// by some other module in the build.
	//
	// If no module version is available for the given path,
	// Upgrade returns a non-nil error.
	// TODO(rsc): Upgrade must be able to return errors,
	// but should "no latest version" just return m instead?
	Upgrade(m module.Version) (module.Version, error)

	// Previous returns the version of m.Path immediately prior to m.Version,
	// or "none" if no such version is known.
	Previous(m module.Version) (module.Version, error)
}

type MissingModuleError struct {
	Module module.Version
}

func (e *MissingModuleError) Error() string {
	return fmt.Sprintf("missing module: %v", e.Module)
}

// BuildList returns the build list for the target module.
func BuildList(target module.Version, reqs Reqs) ([]module.Version, error) {
	return buildList(target, reqs, nil)
}

func buildList(target module.Version, reqs Reqs, upgrade func(module.Version) module.Version) ([]module.Version, error) {
	// Explore work graph in parallel in case reqs.Required
	// does high-latency network operations.
	var work par.Work
	work.Add(target)
	var (
		mu       sync.Mutex
		min      = map[string]string{target.Path: target.Version}
		firstErr error
	)
	work.Do(10, func(item interface{}) {
		m := item.(module.Version)
		required, err := reqs.Required(m)

		mu.Lock()
		if err != nil && firstErr == nil {
			firstErr = err
		}
		if firstErr != nil {
			mu.Unlock()
			return
		}
		if v, ok := min[m.Path]; !ok || reqs.Max(v, m.Version) != v {
			min[m.Path] = m.Version
		}
		mu.Unlock()

		for _, r := range required {
			if r.Path == "" {
				base.Errorf("Required(%v) returned zero module in list", m)
				continue
			}
			work.Add(r)
		}

		if upgrade != nil {
			u := upgrade(m)
			if u.Path == "" {
				base.Errorf("Upgrade(%v) returned zero module", m)
				return
			}
			work.Add(u)
		}
	})

	if firstErr != nil {
		return nil, firstErr
	}
	if v := min[target.Path]; v != target.Version {
		panic(fmt.Sprintf("mistake: chose version %q instead of target %+v", v, target)) // TODO: Don't panic.
	}

	list := []module.Version{target}
	listed := map[string]bool{target.Path: true}
	for i := 0; i < len(list); i++ {
		m := list[i]
		required, err := reqs.Required(m)
		if err != nil {
			return nil, err
		}
		for _, r := range required {
			v := min[r.Path]
			if r.Path != target.Path && reqs.Max(v, r.Version) != v {
				panic(fmt.Sprintf("mistake: version %q does not satisfy requirement %+v", v, r)) // TODO: Don't panic.
			}
			if !listed[r.Path] {
				list = append(list, module.Version{Path: r.Path, Version: v})
				listed[r.Path] = true
			}
		}
	}

	tail := list[1:]
	sort.Slice(tail, func(i, j int) bool {
		return tail[i].Path < tail[j].Path
	})
	return list, nil
}

// Req returns the minimal requirement list for the target module
// that results in the given build list, with the constraint that all
// module paths listed in base must appear in the returned list.
func Req(target module.Version, list []module.Version, base []string, reqs Reqs) ([]module.Version, error) {
	// Note: Not running in parallel because we assume
	// that list came from a previous operation that paged
	// in all the requirements, so there's no I/O to overlap now.

	// Compute postorder, cache requirements.
	var postorder []module.Version
	reqCache := map[module.Version][]module.Version{}
	reqCache[target] = nil
	var walk func(module.Version) error
	walk = func(m module.Version) error {
		_, ok := reqCache[m]
		if ok {
			return nil
		}
		required, err := reqs.Required(m)
		if err != nil {
			return err
		}
		reqCache[m] = required
		for _, m1 := range required {
			if err := walk(m1); err != nil {
				return err
			}
		}
		postorder = append(postorder, m)
		return nil
	}
	for _, m := range list {
		if err := walk(m); err != nil {
			return nil, err
		}
	}

	// Walk modules in reverse post-order, only adding those not implied already.
	have := map[string]string{}
	walk = func(m module.Version) error {
		if v, ok := have[m.Path]; ok && reqs.Max(m.Version, v) == v {
			return nil
		}
		have[m.Path] = m.Version
		for _, m1 := range reqCache[m] {
			walk(m1)
		}
		return nil
	}
	max := map[string]string{}
	for _, m := range list {
		if v, ok := max[m.Path]; ok {
			max[m.Path] = reqs.Max(m.Version, v)
		} else {
			max[m.Path] = m.Version
		}
	}
	// First walk the base modules that must be listed.
	var min []module.Version
	for _, path := range base {
		m := module.Version{Path: path, Version: max[path]}
		min = append(min, m)
		walk(m)
	}
	// Now the reverse postorder to bring in anything else.
	for i := len(postorder) - 1; i >= 0; i-- {
		m := postorder[i]
		if max[m.Path] != m.Version {
			// Older version.
			continue
		}
		if have[m.Path] != m.Version {
			min = append(min, m)
			walk(m)
		}
	}
	sort.Slice(min, func(i, j int) bool {
		return min[i].Path < min[j].Path
	})
	return min, nil
}

// UpgradeAll returns a build list for the target module
// in which every module is upgraded to its latest version.
func UpgradeAll(target module.Version, reqs Reqs) ([]module.Version, error) {
	return buildList(target, reqs, func(m module.Version) module.Version {
		if m.Path == target.Path {
			return target
		}

		latest, err := reqs.Upgrade(m)
		if err != nil {
			panic(err) // TODO
		}
		m.Version = latest.Version
		return m
	})
}

// Upgrade returns a build list for the target module
// in which the given additional modules are upgraded.
func Upgrade(target module.Version, reqs Reqs, upgrade ...module.Version) ([]module.Version, error) {
	list, err := reqs.Required(target)
	if err != nil {
		panic(err) // TODO
	}
	// TODO: Maybe if an error is given,
	// rerun with BuildList(upgrade[0], reqs) etc
	// to find which ones are the buggy ones.
	list = append([]module.Version(nil), list...)
	list = append(list, upgrade...)
	return BuildList(target, &override{target, list, reqs})
}

// Downgrade returns a build list for the target module
// in which the given additional modules are downgraded.
//
// The versions to be downgraded may be unreachable from reqs.Latest and
// reqs.Previous, but the methods of reqs must otherwise handle such versions
// correctly.
func Downgrade(target module.Version, reqs Reqs, downgrade ...module.Version) ([]module.Version, error) {
	list, err := reqs.Required(target)
	if err != nil {
		panic(err) // TODO
	}
	max := make(map[string]string)
	for _, r := range list {
		max[r.Path] = r.Version
	}
	for _, d := range downgrade {
		if v, ok := max[d.Path]; !ok || reqs.Max(v, d.Version) != d.Version {
			max[d.Path] = d.Version
		}
	}

	var (
		added    = make(map[module.Version]bool)
		rdeps    = make(map[module.Version][]module.Version)
		excluded = make(map[module.Version]bool)
	)
	var exclude func(module.Version)
	exclude = func(m module.Version) {
		if excluded[m] {
			return
		}
		excluded[m] = true
		for _, p := range rdeps[m] {
			exclude(p)
		}
	}
	var add func(module.Version)
	add = func(m module.Version) {
		if added[m] {
			return
		}
		added[m] = true
		if v, ok := max[m.Path]; ok && reqs.Max(m.Version, v) != v {
			exclude(m)
			return
		}
		list, err := reqs.Required(m)
		if err != nil {
			panic(err) // TODO
		}
		for _, r := range list {
			add(r)
			if excluded[r] {
				exclude(m)
				return
			}
			rdeps[r] = append(rdeps[r], m)
		}
	}

	var out []module.Version
	out = append(out, target)
List:
	for _, r := range list {
		add(r)
		for excluded[r] {
			p, err := reqs.Previous(r)
			if err != nil {
				return nil, err // TODO
			}
			// If the target version is a pseudo-version, it may not be
			// included when iterating over prior versions using reqs.Previous.
			// Insert it into the right place in the iteration.
			// If v is excluded, p should be returned again by reqs.Previous on the next iteration.
			if v := max[r.Path]; reqs.Max(v, r.Version) != v && reqs.Max(p.Version, v) != p.Version {
				p.Version = v
			}
			if p.Version == "none" {
				continue List
			}
			add(p)
			r = p
		}
		out = append(out, r)
	}

	return out, nil
}

type override struct {
	target module.Version
	list   []module.Version
	Reqs
}

func (r *override) Required(m module.Version) ([]module.Version, error) {
	if m == r.target {
		return r.list, nil
	}
	return r.Reqs.Required(m)
}
