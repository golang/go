package fillstruct

type StructA struct {
	unexportedIntField int
	ExportedIntField   int
	MapA               map[int]string
	Array              []int
	StructB
}

type StructA2 struct {
	B *StructB
}

type StructA3 struct {
	B StructB
}

func fill() {
	a := StructA{}  //@refactorrewrite("}", "Fill struct")
	b := StructA2{} //@refactorrewrite("}", "Fill struct")
	c := StructA3{} //@refactorrewrite("}", "Fill struct")
}
