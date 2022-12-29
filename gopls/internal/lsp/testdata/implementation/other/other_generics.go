//go:build go1.18
// +build go1.18

package other

// -- generics (limited support) --

type GI[T any] interface { //@mark(GI, "GI"),implementations("GI", GenConc)
	F(int, string, T) //@mark(GIF, "F"),implementations("F", GenConcF)
}

type GIString GI[string] //@mark(GIString, "GIString"),implementations("GIString", GenConcString)

type GC[U any] int //@mark(GC, "GC"),implementations("GC", GenIface)

func (GC[V]) F(int, string, V) {} //@mark(GCF, "F"),implementations("F", GenIfaceF)
