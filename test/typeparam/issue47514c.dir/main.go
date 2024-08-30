package main

import "./a"

func Do[T any](doer a.Doer[T]) {
	doer.Do()
}

func main() {
}
