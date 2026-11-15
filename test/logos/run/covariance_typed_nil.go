// run

package main

import (
	"fmt"
	"os"

	"logos/rt"
)

var failed bool

func check(name string, cond bool, got, want any) {
	if !cond {
		failed = true
		fmt.Println("FAIL", name+":", "got", got, "want", want)
		return
	}
	fmt.Println("ok", name)
}

type Creature interface {
	pointerName() string
}

type Monster struct{ name string }

func (m *Monster) pointerName() string {
	if m == nil {
		return "<nil>"
	}
	return m.name
}

func main() {
	typed := rt.CovSlice[Creature]([]*Monster{nil, {name: "imp"}})
	check("cov-slice-keeps-elements", len(typed) == 2 && typed[0] == (*Monster)(nil), fmt.Sprint(len(typed), typed[0] == nil), "2 true")
	check("cov-slice-keeps-typed-nil", typed[0] != nil, typed[0] == nil, false)
	check("cov-slice-dispatches-nil-receiver", typed[0].pointerName() == "<nil>", typed[0].pointerName(), "<nil>")
	check("cov-slice-dispatches-value", typed[1].pointerName() == "imp", typed[1].pointerName(), "imp")

	var none []*Monster
	check("cov-slice-nil-in-nil-out", rt.CovSlice[Creature](none) == nil, rt.CovSlice[Creature](none) == nil, true)
	empty := rt.CovSlice[Creature]([]*Monster{})
	check("cov-slice-empty-stays-non-nil", empty != nil && len(empty) == 0, fmt.Sprint(empty == nil, len(empty)), "false 0")

	in := []Creature{nil, &Monster{name: "orc"}}
	ifaces := rt.CovSlice[Creature](in)
	check("cov-slice-passes-interfaces-through", len(ifaces) == 2 && ifaces[0] == in[0] && ifaces[1] == in[1], fmt.Sprint(len(ifaces), ifaces[0] == nil), "2 true")
	check("cov-slice-dispatches-passthrough", ifaces[0] == nil && ifaces[1].pointerName() == "orc", ifaces[1].pointerName(), "orc")
	if failed {
		os.Exit(1)
	}
}
