// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mvs

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"cmd/go/internal/module"
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
build A: A B1 C2 D4 E2 F1
upgrade* A: A B1 C4 D5 E2 F1 G1
upgrade A C4: A B1 C4 D4 E2 F1 G1
downgrade A2 D2: A2 C4 D2

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
build A: A B1 C D2 E1
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
build A: A D1 E2
upgrade* A: A D2 E2
upgrade A D2: A D2 E2
upgradereq A D2: D2 E2

name: cross6
A: D2
D1: E2
D2: E1
build A: A D2 E1
upgrade* A: A D2 E2
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
build A: A B1 C1 D1
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
build A: A B1 C1
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
build A: A B5.hidden C1
upgrade* A: A B5.hidden C3

name: down1
A: B2
B1: C1
B2: C2
build A: A B2 C2
downgrade A C1: A B1

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
downgrade A F1: A B1 E1

name: down3
A: 

# golang.org/issue/25542.
name: noprev1
A: B4 C2
B2.hidden: 
C2: 
downgrade A B2.hidden: A B2.hidden C2

name: noprev2
A: B4 C2
B2.hidden: 
B1: 
C2: 
downgrade A B2.hidden: A B2.hidden C2

name: noprev3
A: B4 C2
B3: 
B2.hidden: 
C2: 
downgrade A B2.hidden: A B2.hidden C2

# Cycles involving the target.

# The target must be the newest version of itself.
name: cycle1
A: B1
B1: A1
B2: A2
B3: A3
build A: A B1
upgrade A B2: A B2
upgrade* A: A B3

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
build A: A B1 C1 D1
upgrade* A: A B2 C2 D2

# Requirement minimization.

name: req1
A: B1 C1 D1 E1 F1
B1: C1 E1 F1
req A: B1 D1
req A C: B1 C1 D1

name: req2
A: G1 H1
G1: H1
H1: G1
req A: G1
req A G: G1
req A H: H1

name: req3
M: A1 B1
A1: X1
B1: X2
X1: I1
X2: 
req M: A1 B1
`

func Test(t *testing.T) {
	var (
		name string
		reqs reqsMap
		fns  []func(*testing.T)
	)
	flush := func() {
		if name != "" {
			t.Run(name, func(t *testing.T) {
				for _, fn := range fns {
					fn(t)
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
			fns = append(fns, func(t *testing.T) {
				list, err := BuildList(m(kf[1]), reqs)
				checkList(t, key, list, err, val)
			})
			continue
		case "upgrade*":
			if len(kf) != 2 {
				t.Fatalf("upgrade* takes one argument: %q", line)
			}
			fns = append(fns, func(t *testing.T) {
				list, err := UpgradeAll(m(kf[1]), reqs)
				checkList(t, key, list, err, val)
			})
			continue
		case "upgradereq":
			if len(kf) < 2 {
				t.Fatalf("upgrade takes at least one argument: %q", line)
			}
			fns = append(fns, func(t *testing.T) {
				list, err := Upgrade(m(kf[1]), reqs, ms(kf[2:])...)
				if err == nil {
					list, err = Req(m(kf[1]), list, nil, reqs)
				}
				checkList(t, key, list, err, val)
			})
			continue
		case "upgrade":
			if len(kf) < 2 {
				t.Fatalf("upgrade takes at least one argument: %q", line)
			}
			fns = append(fns, func(t *testing.T) {
				list, err := Upgrade(m(kf[1]), reqs, ms(kf[2:])...)
				checkList(t, key, list, err, val)
			})
			continue
		case "downgrade":
			if len(kf) < 2 {
				t.Fatalf("downgrade takes at least one argument: %q", line)
			}
			fns = append(fns, func(t *testing.T) {
				list, err := Downgrade(m(kf[1]), reqs, ms(kf[1:])...)
				checkList(t, key, list, err, val)
			})
			continue
		case "req":
			if len(kf) < 2 {
				t.Fatalf("req takes at least one argument: %q", line)
			}
			fns = append(fns, func(t *testing.T) {
				list, err := BuildList(m(kf[1]), reqs)
				if err != nil {
					t.Fatal(err)
				}
				list, err = Req(m(kf[1]), list, kf[2:], reqs)
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

func (r reqsMap) Max(v1, v2 string) string {
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
	var u module.Version
	for k := range r {
		if k.Path == m.Path && u.Version < k.Version && !strings.HasSuffix(k.Version, ".hidden") {
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
