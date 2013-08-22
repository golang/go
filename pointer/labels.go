package pointer

import (
	"fmt"
	"go/token"
	"strings"

	"code.google.com/p/go.tools/ssa"
)

// A Label is an abstract location or an instruction that allocates memory.
// A points-to set is (conceptually) a set of labels.
//
// (This is basically a pair of a Value that allocates an object and a
// subelement indicator within that object.)
//
// TODO(adonovan): local labels should include their context (CallGraphNode).
//
type Label struct {
	Value      ssa.Value
	subelement *fieldInfo // e.g. ".a.b[*].c"
}

func (l *Label) Pos() token.Pos {
	if l.Value != nil {
		return l.Value.Pos()
	}
	return token.NoPos
}

func (l *Label) String() string {
	var s string
	switch v := l.Value.(type) {
	case *ssa.Function, *ssa.Global:
		s = v.String()

	case *ssa.Const:
		s = v.Name()

	case *ssa.Alloc:
		s = v.Comment
		if s == "" {
			s = "alloc"
		}

	case *ssa.Call:
		// Currently only calls to append can allocate objects.
		if v.Call.Value.(*ssa.Builtin).Object().Name() != "append" {
			panic("unhandled *ssa.Call label: " + v.Name())
		}
		s = "append"

	case *ssa.MakeMap, *ssa.MakeChan, *ssa.MakeSlice, *ssa.Convert:
		s = strings.ToLower(strings.TrimPrefix(fmt.Sprintf("%T", v), "*ssa."))

	case *ssa.MakeInterface:
		// MakeInterface is usually implicit in Go source (so
		// Pos()==0), and interfaces objects may be allocated
		// synthetically (so no *MakeInterface data).
		s = "makeinterface:" + v.X.Type().String()

	default:
		panic(fmt.Sprintf("unhandled Label.Value type: %T", v))
	}

	return s + l.subelement.path()
}
