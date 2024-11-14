// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mvs

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/mod/module"
)

var tests = `
# Scenario from blog.
name: blog
A: B1 C2
B1: D3
C1: D2
C2: D4
C3: D5
C4: G1
D2: E1
D3: E2
D4: E2 F1
D5: E2
G1: C4
A2: B1 C4 D4
build A:       A B1 C2 D4 E2 F1
upgrade* A:    A B1 C4 D5 E2 F1 G1
upgrade A C4:  A B1 C4 D4 E2 F1 G1
build A2:     A2 B1 C4 D4 E2 F1 G1
downgrade A2 D2: A2 C4 D2 E2 F1 G1

name: trim
A: B1 C2
B1: D3
C2: B2
B2:
build A: A B2 C2 D3

# Cross-dependency between D and E.
# No matter how it arises, should get result of merging all build lists via max,
# which leads to including both D2 and E2.

name: cross1
A: B C
B: D1
C: D2
D1: E2
D2: E1
build A: A B C D2 E2

name: cross1V
A: B2 C D2 E1
B1:
B2: D1
C: D2
D1: E2
D2: E1
build A: A B2 C D2 E2

name: cross1U
A: B1 C
B1:
B2: D1
C: D2
D1: E2
D2: E1
build A:      A B1 C D2 E1
upgrade A B2: A B2 C D2 E2

name: cross1R
A: B C
B: D2
C: D1
D1: E2
D2: E1
build A: A B C D2 E2

name: cross1X
A: B C
B: D1 E2
C: D2
D1: E2
D2: E1
build A: A B C D2 E2

name: cross2
A: B D2
B: D1
D1: E2
D2: E1
build A: A B D2 E2

name: cross2X
A: B D2
B: D1 E2
C: D2
D1: E2
D2: E1
build A: A B D2 E2

name: cross3
A: B D2 E1
B: D1
D1: E2
D2: E1
build A: A B D2 E2

name: cross3X
A: B D2 E1
B: D1 E2
D1: E2
D2: E1
build A: A B D2 E2

# Should not get E2 here, because B has been updated
# not to depend on D1 anymore.
name: cross4
A1: B1 D2
A2: B2 D2
B1: D1
B2: D2
D1: E2
D2: E1
build A1: A1 B1 D2 E2
build A2: A2 B2 D2 E1

# But the upgrade from A1 preserves the E2 dep explicitly.
upgrade A1 B2: A1 B2 D2 E2
upgradereq A1 B2: B2 E2

name: cross5
A: D1
D1: E2
D2: E1
build A:       A D1 E2
upgrade* A:    A D2 E2
upgrade A D2:  A D2 E2
upgradereq A D2: D2 E2

name: cross6
A: D2
D1: E2
D2: E1
build A:      A D2 E1
upgrade* A:   A D2 E2
upgrade A E2: A D2 E2

name: cross7
A: B C
B: D1
C: E1
D1: E2
E1: D2
build A: A B C D2 E2

# golang.org/issue/31248:
# Even though we select X2, the requirement on I1
# via X1 should be preserved.
name: cross8
M: A1 B1
A1: X1
B1: X2
X1: I1
X2:
build M: M A1 B1 I1 X2

# Upgrade from B1 to B2 should not drop the transitive dep on D.
name: drop
A: B1 C1
B1: D1
B2:
C2:
D2:
build A:    A B1 C1 D1
upgrade* A: A B2 C2 D2

name: simplify
A: B1 C1
B1: C2
C1: D1
C2:
build A: A B1 C2 D1

name: up1
A: B1 C1
B1:
B2:
B3:
B4:
B5.hidden:
C2:
C3:
build A:    A B1 C1
upgrade* A: A B4 C3

name: up2
A: B5.hidden C1
B1:
B2:
B3:
B4:
B5.hidden:
C2:
C3:
build A:    A B5.hidden C1
upgrade* A: A B5.hidden C3

name: down1
A: B2
B1: C1
B2: C2
build A:        A B2 C2
downgrade A C1: A B1 C1

name: down2
A: B2 E2
B1:
B2: C2 F2
C1:
D1:
C2: D2 E2
D2: B2
E2: D2
E1:
F1:
build A:        A B2 C2 D2 E2 F2
downgrade A F1: A B1 C1 D1 E1 F1

# https://research.swtch.com/vgo-mvs#algorithm_4:
# “[D]owngrades are constrained to only downgrade packages, not also upgrade
# them; if an upgrade before downgrade is needed, the user must ask for it
# explicitly.”
#
# Here, downgrading B2 to B1 upgrades C1 to C2, and C2 does not depend on D2.
# However, C2 would be an upgrade — not a downgrade — so B1 must also be
# rejected.
name: downcross1
A: B2 C1
B1: C2
B2: C1
C1: D2
C2:
D1:
D2:
build A:        A B2 C1 D2
downgrade A D1: A       D1

# https://research.swtch.com/vgo-mvs#algorithm_4:
# “Unlike upgrades, downgrades must work by removing requirements, not adding
# them.”
#
# However, downgrading a requirement may introduce a new requirement on a
# previously-unrequired module. If each dependency's requirements are complete
# (“tidy”), that can't change the behavior of any other package whose version is
# not also being downgraded, so we should allow it.
name: downcross2
A: B2
B1: C1
B2: D2
C1:
D1:
D2:
build A:        A B2    D2
downgrade A D1: A B1 C1 D1

name: downcycle
A: A B2
B2: A
B1:
build A:        A B2
downgrade A B1: A B1

# Both B3 and C2 require D2.
# If we downgrade D to D1, then in isolation B3 would downgrade to B1,
# because B2 is hidden — B1 is the next-highest version that is not hidden.
# However, if we downgrade D, we will also downgrade C to C1.
# And C1 requires B2.hidden, and B2.hidden also meets our requirements:
# it is compatible with D1 and a strict downgrade from B3.
#
# Since neither the initial nor the final build list includes B1,
# and the nothing in the final downgraded build list requires E at all,
# no dependency on E1 (required by only B1) should be introduced.
#
name: downhiddenartifact
A: B3 C2
A1: B3
B1: E1
B2.hidden:
B3: D2
C1: B2.hidden
C2: D2
D1:
D2:
build A1: A1 B3 D2
downgrade A1 D1: A1 B1 D1 E1
build A: A B3 C2 D2
downgrade A D1: A B2.hidden C1 D1

# Both B3 and C3 require D2.
# If we downgrade D to D1, then in isolation B3 would downgrade to B1,
# and C3 would downgrade to C1.
# But C1 requires B2.hidden, and B1 requires C2.hidden, so we can't
# downgrade to either of those without pulling the other back up a little.
#
# B2.hidden and C2.hidden are both compatible with D1, so that still
# meets our requirements — but then we're in an odd state in which
# B and C have both been downgraded to hidden versions, without any
# remaining requirements to explain how those hidden versions got there.
#
# TODO(bcmills): Would it be better to force downgrades to land on non-hidden
# versions?
# In this case, that would remove the dependencies on B and C entirely.
#
name: downhiddencross
A: B3 C3
B1: C2.hidden
B2.hidden:
B3: D2
C1: B2.hidden
C2.hidden:
C3: D2
D1:
D2:
build A: A B3 C3 D2
downgrade A D1: A B2.hidden C2.hidden D1

# golang.org/issue/25542.
name: noprev1
A: B4 C2
B2.hidden:
C2:
build A:               A B4        C2
downgrade A B2.hidden: A B2.hidden C2

name: noprev2
A: B4 C2
B2.hidden:
B1:
C2:
build A:               A B4        C2
downgrade A B2.hidden: A B2.hidden C2

name: noprev3
A: B4 C2
B3:
B2.hidden:
C2:
build A:               A B4        C2
downgrade A B2.hidden: A B2.hidden C2

# Cycles involving the target.

# The target must be the newest version of itself.
name: cycle1
A: B1
B1: A1
B2: A2
B3: A3
build A:      A B1
upgrade A B2: A B2
upgrade* A:   A B3

# golang.org/issue/29773:
# Requirements of older versions of the target
# must be carried over.
name: cycle2
A: B1
A1: C1
A2: D1
B1: A1
B2: A2
C1: A2
C2:
D2:
build A:    A B1 C1 D1
upgrade* A: A B2 C2 D2

# Cycles with multiple possible solutions.
# (golang.org/issue/34086)
name: cycle3
M: A1 C2
A1: B1
B1: C1
B2: C2
C1:
C2: B2
build M: M A1 B2 C2
req M:     A1 B2
req M A:   A1 B2
req M C:   A1 C2

# Requirement minimization.

name: req1
A: B1 C1 D1 E1 F1
B1: C1 E1 F1
req A:   B1    D1
req A C: B1 C1 D1

name: req2
A: G1 H1
G1: H1
H1: G1
req A:   G1
req A G: G1
req A H: H1

name: req3
M: A1 B1
A1: X1
B1: X2
X1: I1
X2:
req M: A1 B1

name: reqnone
M: Anone B1 D1 E1
B1: Cnone D1
E1: Fnone
build M: M B1 D1 E1
req M:     B1    E1

name: reqdup
M: A1 B1
A1: B1
B1:
req M A A: A1

name: reqcross
M: A1 B1 C1
A1: B1 C1
B1: C1
C1:
req M A B: A1 B1
`

func Test(t *testing.T) {
	var (
		name string
		reqs reqsMap
		fns  []func(*testing.T)
	)
	flush := func() {
		if name != "" {
			t.Run(name, func { t ->
				for _, fn := range fns {
					fn(t)
				}
				if len(fns) == 0 {
					t.Errorf("no functions tested")
				}
			})
		}
	}
	m := func(s string) module.Version {
		return module.Version{Path: s[:1], Version: s[1:]}
	}
	ms := func(list []string) []module.Version {
		var mlist []module.Version
		for _, s := range list {
			mlist = append(mlist, m(s))
		}
		return mlist
	}
	checkList := func(t *testing.T, desc string, list []module.Version, err error, val string) {
		if err != nil {
			t.Fatalf("%s: %v", desc, err)
		}
		vs := ms(strings.Fields(val))
		if !reflect.DeepEqual(list, vs) {
			t.Errorf("%s = %v, want %v", desc, list, vs)
		}
	}

	for _, line := range strings.Split(tests, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		i := strings.Index(line, ":")
		if i < 0 {
			t.Fatalf("missing colon: %q", line)
		}
		key := strings.TrimSpace(line[:i])
		val := strings.TrimSpace(line[i+1:])
		if key == "" {
			t.Fatalf("missing key: %q", line)
		}
		kf := strings.Fields(key)
		switch kf[0] {
		case "name":
			if len(kf) != 1 {
				t.Fatalf("name takes no arguments: %q", line)
			}
			flush()
			reqs = make(reqsMap)
			fns = nil
			name = val
			continue
		case "build":
			if len(kf) != 2 {
				t.Fatalf("build takes one argument: %q", line)
			}
			fns = append(fns, func { t ->
				list, err := BuildList([]module.Version{m(kf[1])}, reqs)
				checkList(t, key, list, err, val)
			})
			continue
		case "upgrade*":
			if len(kf) != 2 {
				t.Fatalf("upgrade* takes one argument: %q", line)
			}
			fns = append(fns, func { t ->
				list, err := UpgradeAll(m(kf[1]), reqs)
				checkList(t, key, list, err, val)
			})
			continue
		case "upgradereq":
			if len(kf) < 2 {
				t.Fatalf("upgrade takes at least one argument: %q", line)
			}
			fns = append(fns, func { t ->
				list, err := Upgrade(m(kf[1]), reqs, ms(kf[2:])...)
				if err == nil {
					// Copy the reqs map, but substitute the upgraded requirements in
					// place of the target's original requirements.
					upReqs := make(reqsMap, len(reqs))
					for m, r := range reqs {
						upReqs[m] = r
					}
					upReqs[m(kf[1])] = list

					list, err = Req(m(kf[1]), nil, upReqs)
				}
				checkList(t, key, list, err, val)
			})
			continue
		case "upgrade":
			if len(kf) < 2 {
				t.Fatalf("upgrade takes at least one argument: %q", line)
			}
			fns = append(fns, func { t ->
				list, err := Upgrade(m(kf[1]), reqs, ms(kf[2:])...)
				checkList(t, key, list, err, val)
			})
			continue
		case "downgrade":
			if len(kf) < 2 {
				t.Fatalf("downgrade takes at least one argument: %q", line)
			}
			fns = append(fns, func { t ->
				list, err := Downgrade(m(kf[1]), reqs, ms(kf[1:])...)
				checkList(t, key, list, err, val)
			})
			continue
		case "req":
			if len(kf) < 2 {
				t.Fatalf("req takes at least one argument: %q", line)
			}
			fns = append(fns, func { t ->
				list, err := Req(m(kf[1]), kf[2:], reqs)
				checkList(t, key, list, err, val)
			})
			continue
		}
		if len(kf) == 1 && 'A' <= key[0] && key[0] <= 'Z' {
			var rs []module.Version
			for _, f := range strings.Fields(val) {
				r := m(f)
				if reqs[r] == nil {
					reqs[r] = []module.Version{}
				}
				rs = append(rs, r)
			}
			reqs[m(key)] = rs
			continue
		}
		t.Fatalf("bad line: %q", line)
	}
	flush()
}

type reqsMap map[module.Version][]module.Version

func (r reqsMap) Max(_, v1, v2 string) string {
	if v1 == "none" || v2 == "" {
		return v2
	}
	if v2 == "none" || v1 == "" {
		return v1
	}
	if v1 < v2 {
		return v2
	}
	return v1
}

func (r reqsMap) Upgrade(m module.Version) (module.Version, error) {
	u := module.Version{Version: "none"}
	for k := range r {
		if k.Path == m.Path && r.Max(k.Path, u.Version, k.Version) == k.Version && !strings.HasSuffix(k.Version, ".hidden") {
			u = k
		}
	}
	if u.Path == "" {
		return module.Version{}, fmt.Errorf("missing module: %v", module.Version{Path: m.Path})
	}
	return u, nil
}

func (r reqsMap) Previous(m module.Version) (module.Version, error) {
	var p module.Version
	for k := range r {
		if k.Path == m.Path && p.Version < k.Version && k.Version < m.Version && !strings.HasSuffix(k.Version, ".hidden") {
			p = k
		}
	}
	if p.Path == "" {
		return module.Version{Path: m.Path, Version: "none"}, nil
	}
	return p, nil
}

func (r reqsMap) Required(m module.Version) ([]module.Version, error) {
	rr, ok := r[m]
	if !ok {
		return nil, fmt.Errorf("missing module: %v", m)
	}
	return rr, nil
}
