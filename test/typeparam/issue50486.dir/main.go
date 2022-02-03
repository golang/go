package main

import fp "goerror_fp"

func Fold[A, B any](zero B, a A, f func(B, A) B) B {
	return f(zero, a)
}

func main() {

	var v any = "hello"
	Fold(fp.Seq[any]{}, v, func(seq fp.Seq[any], v any) fp.Seq[any] {
		return seq.Append(v)
	})

}
