package labels

func _() {
	goto F //@complete(" //", label1, label5)

Foo1: //@item(label1, "Foo1", "label", "const")
	for a, b := range []int{} {
	Foo2: //@item(label2, "Foo2", "label", "const")
		switch {
		case true:
			break F //@complete(" //", label2, label1)

			continue F //@complete(" //", label1)

			{
			FooUnjumpable:
			}

			goto F //@complete(" //", label1, label2, label4, label5)

			func() {
				goto F //@complete(" //", label3)

				break F //@complete(" //")

				continue F //@complete(" //")

			Foo3: //@item(label3, "Foo3", "label", "const")
			}()
		}

	Foo4: //@item(label4, "Foo4", "label", "const")
		switch interface{}(a).(type) {
		case int:
			break F //@complete(" //", label4, label1)
		}
	}

	break F //@complete(" //")

	continue F //@complete(" //")

Foo5: //@item(label5, "Foo5", "label", "const")
	for {
		break F //@complete(" //", label5)
	}

	return
}
