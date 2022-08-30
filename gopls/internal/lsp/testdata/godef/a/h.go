package a

func _() {
	type s struct {
		nested struct {
			// nested number
			number int64 //@mark(nestedNumber, "number")
		}
		nested2 []struct {
			// nested string
			str string //@mark(nestedString, "str")
		}
		x struct {
			x struct {
				x struct {
					x struct {
						x struct {
							// nested map
							m map[string]float64 //@mark(nestedMap, "m")
						}
					}
				}
			}
		}
	}

	var t s
	_ = t.nested.number  //@hoverdef("number", nestedNumber)
	_ = t.nested2[0].str //@hoverdef("str", nestedString)
	_ = t.x.x.x.x.x.m    //@hoverdef("m", nestedMap)
}

func _() {
	var s struct {
		// a field
		a int //@mark(structA, "a")
		// b nested struct
		b struct { //@mark(structB, "b")
			// c field of nested struct
			c int //@mark(structC, "c")
		}
	}
	_ = s.a   //@hoverdef("a", structA)
	_ = s.b   //@hoverdef("b", structB)
	_ = s.b.c //@hoverdef("c", structC)

	var arr []struct {
		// d field
		d int //@mark(arrD, "d")
		// e nested struct
		e struct { //@mark(arrE, "e")
			// f field of nested struct
			f int //@mark(arrF, "f")
		}
	}
	_ = arr[0].d   //@hoverdef("d", arrD)
	_ = arr[0].e   //@hoverdef("e", arrE)
	_ = arr[0].e.f //@hoverdef("f", arrF)

	var complex []struct {
		c <-chan map[string][]struct {
			// h field
			h int //@mark(complexH, "h")
			// i nested struct
			i struct { //@mark(complexI, "i")
				// j field of nested struct
				j int //@mark(complexJ, "j")
			}
		}
	}
	_ = (<-complex[0].c)["0"][0].h   //@hoverdef("h", complexH)
	_ = (<-complex[0].c)["0"][0].i   //@hoverdef("i", complexI)
	_ = (<-complex[0].c)["0"][0].i.j //@hoverdef("j", complexJ)

	var mapWithStructKey map[struct {
		// X key field
		x []string //@mark(mapStructKeyX, "x")
	}]int
	for k := range mapWithStructKey {
		_ = k.x //@hoverdef("x", mapStructKeyX)
	}

	var mapWithStructKeyAndValue map[struct {
		// Y key field
		y string //@mark(mapStructKeyY, "y")
	}]struct {
		// X value field
		x string //@mark(mapStructValueX, "x")
	}
	for k, v := range mapWithStructKeyAndValue {
		// TODO: we don't show docs for y field because both map key and value
		// are structs. And in this case, we parse only map value
		_ = k.y //@hoverdef("y", mapStructKeyY)
		_ = v.x //@hoverdef("x", mapStructValueX)
	}

	var i []map[string]interface {
		// open method comment
		open() error //@mark(openMethod, "open")
	}
	i[0]["1"].open() //@hoverdef("open", openMethod)
}

func _() {
	test := struct {
		// test description
		desc string //@mark(testDescription, "desc")
	}{}
	_ = test.desc //@hoverdef("desc", testDescription)

	for _, tt := range []struct {
		// test input
		in map[string][]struct { //@mark(testInput, "in")
			// test key
			key string //@mark(testInputKey, "key")
			// test value
			value interface{} //@mark(testInputValue, "value")
		}
		result struct {
			v <-chan struct {
				// expected test value
				value int //@mark(testResultValue, "value")
			}
		}
	}{} {
		_ = tt.in               //@hoverdef("in", testInput)
		_ = tt.in["0"][0].key   //@hoverdef("key", testInputKey)
		_ = tt.in["0"][0].value //@hoverdef("value", testInputValue)

		_ = (<-tt.result.v).value //@hoverdef("value", testResultValue)
	}
}

func _() {
	getPoints := func() []struct {
		// X coord
		x int //@mark(returnX, "x")
		// Y coord
		y int //@mark(returnY, "y")
	} {
		return nil
	}

	r := getPoints()
	r[0].x //@hoverdef("x", returnX)
	r[0].y //@hoverdef("y", returnY)
}
