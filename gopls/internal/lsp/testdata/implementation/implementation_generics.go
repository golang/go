//go:build go1.18
// +build go1.18

package implementation

// -- generics --

type GenIface[T any] interface { //@mark(GenIface, "GenIface"),implementations("GenIface", GC)
	F(int, string, T) //@mark(GenIfaceF, "F"),implementations("F", GCF)
}

type GenConc[U any] int //@mark(GenConc, "GenConc"),implementations("GenConc", GI)

func (GenConc[V]) F(int, string, V) {} //@mark(GenConcF, "F"),implementations("F", GIF)

type GenConcString struct{ GenConc[string] } //@mark(GenConcString, "GenConcString"),implementations(GenConcString, GIString)
