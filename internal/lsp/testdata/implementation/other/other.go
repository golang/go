package other

type ImpP struct{} //@mark(OtherImpP, "ImpP")

func (*ImpP) Laugh() { //@mark(OtherLaughP, "Laugh")
}

type ImpS struct{} //@mark(OtherImpS, "ImpS")

func (ImpS) Laugh() { //@mark(OtherLaughS, "Laugh")
}

type ImpI interface {
	Laugh()
}

type Foo struct {
}

func (Foo) U() { //@mark(ImpU, "U")
}

type CryType int

const Sob CryType = 1

type Cryer interface {
	Cry(CryType) //@implementations("Cry", CryImpl)
}
