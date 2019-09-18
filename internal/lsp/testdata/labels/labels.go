package labels

func _() {
	goto F //@complete(" //", label1, label4)

Foo1: //@item(label1, "Foo1", "label", "const")
	for {
	Foo2: //@item(label2, "Foo2", "label", "const")
		switch {
		case true:
			break F //@complete(" //", label2, label1)

			continue F //@complete(" //", label1)

			{
			FooUnjumpable:
			}

			goto F //@complete(" //", label1, label2, label4)

			func() {
				goto F //@complete(" //", label3)

				break F //@complete(" //")

				continue F //@complete(" //")

			Foo3: //@item(label3, "Foo3", "label", "const")
			}()
		}
	}

	break F //@complete(" //")

	continue F //@complete(" //")

Foo4: //@item(label4, "Foo4", "label", "const")
	return
}
