package testy

type tt int //@rename("tt", "testyType")

func a() {
	foo := 42 //@rename("foo", "bar")
}
