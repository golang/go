// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"
)

func requireLowered(t *testing.T, out string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(out, want) {
			t.Errorf("lowered text misses %q:\n%s", want, out)
		}
	}
}

func TestPlatformStorePassesStaticTypeName(t *testing.T) {
	src := `package main

import (
	"flag"
	"sync/atomic"
)

func stored() *flag.Flag {
	var p *flag.Flag = flag.Lookup("x")
	return p
}

func generic[E any](a *atomic.Pointer[E]) *E {
	var p *E = a.Load()
	return p
}
`
	out := lowerSnippetSrc(t, "platformname.lgo", src)
	requireLowered(t, out,
		`_logos_rt.MustNonNil(flag.Lookup("x"), "*flag.Flag", &_logos_site`,
		`_logos_rt.MustNonNil(a.Load(), "", &_logos_site`,
	)
}

func TestUnionAssertionPassesStaticTypeName(t *testing.T) {
	src := `package main

type U interface{ int | string }

func named(u U) float64 { return u.(float64) }

func generic[E any](u U) E { return u.(E) }

func commaOk[E any](u U) (E, bool) {
	v, ok := u.(E)
	return v, ok
}
`
	out := lowerSnippetSrc(t, "unionname.lgo", src)
	requireLowered(t, out,
		`_logos_assertfrom_U[float64](u, "float64")`,
		`_logos_assertfrom_U[E](u, "")`,
		`_logos_assertfrom2_U[E](u)`,
	)
}

func TestInterpHelperFollowsStaticOperandType(t *testing.T) {
	src := `package main

import "fmt"

type valErr struct{}

func (valErr) Error() string { return "v" }

type ptrErr struct{}

func (*ptrErr) Error() string { return "p" }

type ptrStr struct{}

func (*ptrStr) String() string { return "s" }

func main() {
	var iface error = valErr{}
	ve := valErr{}
	pe := &ptrErr{}
	ps := &ptrStr{}
	var st fmt.Stringer = ps
	out := $"{iface} {ve} {pe} {ps} {st}"
	_ = out
}
`
	out := lowerSnippetSrc(t, "interphelper.lgo", src)
	requireLowered(t, out,
		`_logos_rt.InterpError(_logos_v0, `,
		`_logos_rt.InterpErrorValue(_logos_v1, `,
		`_logos_rt.InterpErrorPtr(_logos_v2, `,
		`_logos_rt.InterpStringerPtr(_logos_v3, `,
		`_logos_rt.InterpStringer(_logos_v4, `,
	)
}
