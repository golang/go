package other

type ImpP struct{} //@mark(OtherImpP, "ImpP")

func (*ImpP) Laugh() { //@mark(OtherLaughP, "Laugh")
}

type ImpS struct{} //@mark(OtherImpS, "ImpS")

func (ImpS) Laugh() { //@mark(OtherLaughS, "Laugh")
}

type ImpI interface { //@mark(OtherImpI, "ImpI")
	Laugh() //@mark(OtherLaughI, "Laugh")
}
