package implementation

import "golang.org/x/tools/internal/lsp/implementation/other"

type ImpP struct{} //@ImpP

func (*ImpP) Laugh() { //@mark(LaughP, "Laugh")
}

type ImpS struct{} //@ImpS

func (ImpS) Laugh() { //@mark(LaughS, "Laugh")
}

type ImpI interface {
	Laugh() //@implementations("Laugh", LaughP, OtherLaughP, LaughS, OtherLaughS)
}

type Laugher interface { //@implementations("Laugher", ImpP, OtherImpP, ImpS, OtherImpS)
	Laugh() //@implementations("Laugh", LaughP, OtherLaughP, LaughS, OtherLaughS)
}

type Foo struct {
	other.Foo
}

type U interface {
	U() //@implementations("U", ImpU)
}

type cryer int

func (cryer) Cry(other.CryType) {} //@mark(CryImpl, "Cry")
