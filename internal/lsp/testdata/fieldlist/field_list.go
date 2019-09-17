package fieldlist

var myInt int   //@item(flVar, "myInt", "int", "var")
type myType int //@item(flType, "myType", "int", "type")

func (my) _()    {} //@complete(") _", flType, flVar)
func (my my) _() {} //@complete(" my)"),complete(") _", flType, flVar)

func (myType) _() {} //@complete(") {", flType, flVar)

func (myType) _(my my) {} //@complete(" my)"),complete(") {", flType, flVar)

func (myType) _() my {} //@complete(" {", flType, flVar)

func (myType) _() (my my) {} //@complete(" my"),complete(") {", flType, flVar)

func _() {
	var _ struct {
		//@complete("", flType, flVar)
		m my //@complete(" my"),complete(" //", flType, flVar)
	}

	var _ interface {
		//@complete("", flType, flVar)
		m() my //@complete("("),complete(" //", flType, flVar)
	}
}
