// run

package main

var sp = ""

func f(name string, _ ...interface{}) int {
	print(sp, name)
	sp = " "
	return 0
}

var a = f("a", x)
var b = f("b", y)
var c = f("c", z)
var d = func() int {
	if false {
		_ = z
	}
	return f("d")
}()
var e = f("e")

var x int
var y int = 42
var z int = func() int { return 42 }()

func main() { println() }
