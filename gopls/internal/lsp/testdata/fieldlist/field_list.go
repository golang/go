package fieldlist

var myInt int   //@item(flVar, "myInt", "int", "var")
type myType int //@item(flType, "myType", "int", "type")

func (my) _()    {} //@complete(") _", flType)
func (my my) _() {} //@complete(" my)"),complete(") _", flType)

func (myType) _() {} //@complete(") {", flType)

func (myType) _(my my) {} //@complete(" my)"),complete(") {", flType)

func (myType) _() my {} //@complete(" {", flType)

func (myType) _() (my my) {} //@complete(" my"),complete(") {", flType)

func _() {
	var _ struct {
		//@complete("", flType)
		m my //@complete(" my"),complete(" //", flType)
	}

	var _ interface {
		//@complete("", flType)
		m() my //@complete("("),complete(" //", flType)
	}
}
